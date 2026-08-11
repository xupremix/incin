use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{
    Expr, Ident, Lit, Token, Type,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    spanned::Spanned,
};

/// How one leaf expression was written, which is all that decides the
/// tensor's default dtype and whether the leaf gets an inserted `as` cast.
#[derive(Clone, PartialEq, Eq)]
enum LiteralKind {
    /// An unsuffixed integer literal, e.g. `2` or `-2`.
    Int,
    /// An unsuffixed float literal, e.g. `2.0` or `-2.0`.
    Float,
    /// A literal with an explicit dtype suffix, e.g. `2u8`, `2.0f64`.
    Suffixed(String),
    /// Anything else — a variable, a call, an index, ... `tensor!` never
    /// guesses a dtype from these and never casts them.
    Other,
}

struct Leaf {
    expr: Expr,
    kind: LiteralKind,
}

struct TensorInput {
    shape: Vec<usize>,
    leaves: Vec<Leaf>,
    dtype: Option<Type>,
    grad: Option<Type>,
}

/// A `key: value` pair after `tensor!`'s data. Both remaining clauses
/// (`dtype`, `grad`) name a *type*, and they are matched by key rather than
/// position, so either order works.
///
/// There is deliberately no `backend:` or `device:` clause. Placing a tensor
/// somewhere other than the default CPU backend is the allocation target's
/// job — see `incin_backends::target` — and an earlier revision of this macro
/// tried to infer a backend from the *token spelling* of a device expression
/// (`Wgpu::new(0)` → `Wgpu`), which could not see through a binding or an
/// alias and broke the moment a caller wrote `let d = Wgpu::new(0);`.
/// Inferring types from how an expression is spelled is not something a macro
/// should do, and the target API removes the need to.
struct Clause {
    key: Ident,
    value: Type,
}

impl Parse for Clause {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        Ok(Clause {
            key,
            value: input.parse()?,
        })
    }
}

impl Parse for TensorInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // `tensor![]` (no data at all) and `tensor![; dtype: f64]` (data
        // elided before a clause) both mean "empty, shape [0]", matching
        // `vec![]` and `torch.tensor([])`. Anything else must have at least
        // one top-level item to parse as `Expr`s.
        let items: Vec<Expr> = if input.is_empty() || input.peek(Token![;]) {
            Vec::new()
        } else {
            Punctuated::<Expr, Token![,]>::parse_separated_nonempty(input)?
                .into_iter()
                .collect()
        };

        let mut shape = Vec::new();
        let mut leaves = Vec::new();
        classify(&items, 0, &mut shape, &mut leaves)?;

        let mut dtype = None;
        let mut grad = None;
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
            let clauses: Punctuated<Clause, Token![,]> = Punctuated::parse_terminated(input)?;
            for clause in clauses {
                let key_name = clause.key.to_string();
                let slot_is_taken = match key_name.as_str() {
                    "dtype" => dtype.is_some(),
                    "grad" => grad.is_some(),
                    other => {
                        return Err(syn::Error::new_spanned(
                            &clause.key,
                            format!(
                                "unknown tensor! clause `{other}` (expected `dtype` or `grad`). \
                                 To place a tensor on a specific backend or device, allocate it \
                                 from a target instead: `Cpu.tensor(..)`, `Wgpu::new(0).tensor(..)` \
                                 (see `incin_backends::target`)."
                            ),
                        ));
                    }
                };
                if slot_is_taken {
                    return Err(syn::Error::new_spanned(
                        &clause.key,
                        format!("duplicate `{key_name}` clause in tensor!"),
                    ));
                }
                match key_name.as_str() {
                    "dtype" => dtype = Some(clause.value),
                    "grad" => {
                        validate_grad_type(&clause.value)?;
                        grad = Some(clause.value);
                    }
                    _ => unreachable!("the key was matched against the same names above"),
                }
            }
        }

        if !input.is_empty() {
            return Err(input.error("unexpected tokens after tensor!'s `; clauses`"));
        }

        Ok(TensorInput {
            shape,
            leaves,
            dtype,
            grad,
        })
    }
}

/// Walks one level of nesting, recording `shape[depth]` the first time it is
/// reached and requiring every later sibling at that depth to agree with it
/// (a ragged literal is a compile error, not a best-effort reshape). An
/// empty `items` is a valid, terminal 0-length dimension — `tensor![]` and
/// `tensor![[], []]` (shape `[0]`/`[2, 0]`) — matching `vec![]` and
/// `torch.tensor([])`; it just has no further nesting to validate or recurse
/// into, so it returns immediately after the length check below.
fn classify(
    items: &[Expr],
    depth: usize,
    shape: &mut Vec<usize>,
    leaves: &mut Vec<Leaf>,
) -> syn::Result<()> {
    match shape.get(depth) {
        Some(&expected) if expected != items.len() => {
            // `items` can itself be empty here (an empty sibling next to a
            // non-empty one), so there is no last element to anchor the
            // span on; call_site is the honest fallback.
            let span = items
                .last()
                .map_or_else(Span::call_site, syn::spanned::Spanned::span);
            return Err(syn::Error::new(
                span,
                format!(
                    "ragged tensor! literal: dimension {depth} has {} element(s) here but {expected} earlier",
                    items.len()
                ),
            ));
        }
        Some(_) => {}
        None => shape.push(items.len()),
    }

    if items.is_empty() {
        return Ok(());
    }

    let first_is_array = matches!(items[0], Expr::Array(_));
    if let Some(bad) = items
        .iter()
        .find(|item| matches!(item, Expr::Array(_)) != first_is_array)
    {
        return Err(syn::Error::new_spanned(
            bad,
            "tensor! rows must be uniformly nested: every element at this depth must either all be nested arrays or all be plain values",
        ));
    }

    if first_is_array {
        for item in items {
            let Expr::Array(array) = item else {
                unreachable!("checked uniformly above")
            };
            let inner: Vec<Expr> = array.elems.iter().cloned().collect();
            classify(&inner, depth + 1, shape, leaves)?;
        }
    } else {
        for item in items {
            leaves.push(Leaf {
                kind: classify_literal(item),
                expr: item.clone(),
            });
        }
    }
    Ok(())
}

fn classify_literal(expr: &Expr) -> LiteralKind {
    let target = match expr {
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => &*unary.expr,
        other => other,
    };
    let Expr::Lit(lit) = target else {
        return LiteralKind::Other;
    };
    match &lit.lit {
        Lit::Int(i) if i.suffix().is_empty() => LiteralKind::Int,
        Lit::Int(i) => LiteralKind::Suffixed(i.suffix().to_string()),
        Lit::Float(f) if f.suffix().is_empty() => LiteralKind::Float,
        Lit::Float(f) => LiteralKind::Suffixed(f.suffix().to_string()),
        _ => LiteralKind::Other,
    }
}

/// Picks `K`: an explicit `; dtype: T` clause wins outright; otherwise an
/// agreeing numeric-literal suffix across every leaf wins; otherwise
/// `i64` if every leaf is a bare integer literal, `f32` if any leaf is a
/// float literal or not a literal at all (`tensor.rs`'s module doc explains
/// why: an arbitrary expression's type is unknowable at macro-expansion
/// time, so this is a default, not a proof, and a real mismatch still
/// surfaces as `rustc`'s own type error at the `from_slice` call this
/// macro expands to).
fn resolve_dtype(leaves: &[Leaf], explicit: Option<Type>) -> syn::Result<Type> {
    if let Some(ty) = explicit {
        return Ok(ty);
    }

    let mut suffix: Option<(String, proc_macro2::Span)> = None;
    for leaf in leaves {
        if let LiteralKind::Suffixed(name) = &leaf.kind {
            match &suffix {
                None => suffix = Some((name.clone(), leaf.expr.span())),
                Some((prev, _)) if prev != name => {
                    return Err(syn::Error::new_spanned(
                        &leaf.expr,
                        format!(
                            "conflicting numeric literal suffixes in tensor!: found `{prev}` and `{name}` — add `; dtype: <Type>` to disambiguate"
                        ),
                    ));
                }
                _ => {}
            }
        }
    }
    if let Some((name, span)) = suffix {
        return syn::parse_str::<Type>(&name)
            .map_err(|_| syn::Error::new(span, format!("`{name}` is not a valid dtype suffix")));
    }

    // `.all()` on an empty iterator is vacuously true, which would otherwise
    // pick `i64` for `tensor![]`; an empty tensor has no literal to infer
    // from at all, so it gets the same `f32` default `torch.tensor([])` has.
    let all_int_literals =
        !leaves.is_empty() && leaves.iter().all(|leaf| leaf.kind == LiteralKind::Int);
    let name = if all_int_literals { "i64" } else { "f32" };
    Ok(syn::parse_str::<Type>(name).expect("`i64`/`f32` always parse as a type"))
}

/// `; grad:` only accepts `Grad`/`NoGrad` — the two `RequiresGrad` markers
/// with no runtime argument (`Arg = ()`), matching how `dtype:` is also a
/// compile-time-only clause. `Dyn` (runtime-toggled tracking)
/// takes a `bool` `Arg`, which would need a value-carrying clause — not
/// worth it for a marker most code sets once and never flips, so it is left
/// to a direct
/// `Tensor::<S, B, K, Dyn>::from_slice(&data, (.., runtime_bool))` call
/// instead of guessing at a value-carrying grammar for it here.
fn validate_grad_type(ty: &Type) -> syn::Result<()> {
    if let Type::Path(path) = ty
        && let Some(seg) = path.path.segments.last()
        && (seg.ident == "Grad" || seg.ident == "NoGrad")
    {
        return Ok(());
    }
    Err(syn::Error::new_spanned(
        ty,
        "tensor!'s `grad:` clause only supports `Grad` or `NoGrad` — the only \
         gradient-tracking markers with no runtime argument. For `Dyn` \
         (runtime-toggled tracking) construct the tensor directly instead: \
         `Tensor::<S, B, K, Dyn>::from_slice(&data, (.., runtime_bool))`",
    ))
}

fn cast_leaf(leaf: &Leaf, dtype: &Type) -> proc_macro2::TokenStream {
    let expr = &leaf.expr;
    if leaf.kind == LiteralKind::Other {
        // An arbitrary expression must already be `dtype`; casting it would
        // silently narrow a caller's own f64 (or otherwise mistyped) value
        // instead of surfacing the mismatch.
        quote! { #expr }
    } else {
        quote! { (#expr) as #dtype }
    }
}

pub(crate) fn tensor(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as TensorInput);

    let dtype = match resolve_dtype(&parsed.leaves, parsed.dtype) {
        Ok(ty) => ty,
        Err(err) => return err.to_compile_error().into(),
    };
    let dims = &parsed.shape;
    let cast_leaves = parsed.leaves.iter().map(|leaf| cast_leaf(leaf, &dtype));
    // `tensor!` always builds on the default CPU backend, whose device,
    // dtype and grad arguments are all `()`; the shape is fully static
    // because it came from the literal's own nesting. So there is exactly one
    // argument form and no `ArgInto` slot bookkeeping for a caller to get
    // wrong. Anything else — another backend, another device, a runtime
    // ordinal — is the allocation target's job.
    let args = quote! { () };

    // `G` is the 4th of `Tensor`'s 5 type params, after `K` — trailing
    // params with a `Default` (`G`, and `P` after it) can be omitted from
    // the turbofish entirely, so an absent `; grad:` clause just leaves `G`
    // out here and lets it fall back to `Tensor`'s own default (`Grad`)
    // rather than spelling it out redundantly.
    let grad = parsed.grad.map(|ty| quote! { , #ty });

    let expanded = quote! {
        ::incin::prelude::Tensor::<
            ::incin::prelude::s![#(#dims),*],
            ::incin::prelude::DefaultBackend,
            #dtype
            #grad,
        >::from_slice(&[#(#cast_leaves),*], #args)
    };
    expanded.into()
}
