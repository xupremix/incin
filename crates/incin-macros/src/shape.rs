use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

/// Dim.
enum Dim {
    /// Dyn.
    Dyn,
    /// Lit.
    Lit(syn::LitInt),
    /// Path.
    Path(syn::Path),
    /// A compile-time const dimension.
    ConstPath(syn::Path),
    /// A semantic axis tag paired with an independent extent specification.
    Named { tag: syn::Path, extent: Box<Dim> },
}

impl Parse for Dim {
    /// Parse.
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Token![const]) {
            let const_token = input.parse::<Token![const]>()?;
            if input.peek(syn::token::Brace) || input.peek(syn::token::Paren) {
                return Err(syn::Error::new_spanned(
                    const_token,
                    "dimension expressions like `const { ... }` or `const (...)` are not supported in s!",
                ));
            }
            let path = input.parse::<syn::Path>()?;
            if input.peek(Token![*])
                || input.peek(Token![+])
                || input.peek(Token![-])
                || input.peek(Token![/])
            {
                return Err(syn::Error::new(
                    input.span(),
                    "arithmetic expressions after `const` are not supported in s!",
                ));
            }
            return Ok(Dim::ConstPath(path));
        }
        // Keep the named form in the same dimension grammar as anonymous
        // dimensions.  This is deliberately a type-level pairing, not a
        // second runtime shape representation.
        if input.peek(syn::Ident) || input.peek(syn::token::SelfValue) {
            let fork = input.fork();
            if let Ok(tag) = fork.parse::<syn::Path>() {
                if fork.peek(Token![=]) {
                    let tag = input.parse::<syn::Path>()?;
                    input.parse::<Token![=]>()?;
                    let extent = input.parse::<Dim>()?;
                    return Ok(Dim::Named {
                        tag,
                        extent: Box::new(extent),
                    });
                }
                let _ = tag;
            }
        }
        if input.peek(Token![dyn]) {
            input.parse::<Token![dyn]>()?;
            return Ok(Dim::Dyn);
        }
        if input.peek(Token![_]) {
            input.parse::<Token![_]>()?;
            return Ok(Dim::Dyn);
        }
        if input.peek(syn::LitInt) {
            return Ok(Dim::Lit(input.parse::<syn::LitInt>()?));
        }

        Ok(Dim::Path(input.parse::<syn::Path>()?))
    }
}

enum ShapeInput {
    List(Vec<Dim>),
    Repeat { dim: Dim, count: usize },
    Tail(Vec<Dim>),
    Head(Vec<Dim>),
    Span { head: Vec<Dim>, tail: Vec<Dim> },
}

/// Number list.
struct NumberList {
    internal: bool,
    input: ShapeInput,
}

impl Parse for NumberList {
    /// Parse.
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut internal = false;
        if input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            internal = true;
        }

        if !input.peek(Token![..]) {
            let fork = input.fork();
            if fork.parse::<Dim>().is_ok() && fork.peek(Token![;]) {
                let dim: Dim = input.parse()?;
                input.parse::<Token![;]>()?;
                let count_lit: syn::LitInt = input.parse()?;
                let count: usize = count_lit.base10_parse()?;
                return Ok(NumberList {
                    internal,
                    input: ShapeInput::Repeat { dim, count },
                });
            }
        }

        let mut before_dotdot: Vec<Dim> = Vec::new();
        let mut after_dotdot: Vec<Dim> = Vec::new();
        let mut has_dotdot = false;

        while !input.is_empty() {
            if input.peek(Token![..]) {
                if has_dotdot {
                    return Err(syn::Error::new(
                        input.span(),
                        "Only a single '..' ellipsis is permitted in a shape",
                    ));
                }
                input.parse::<Token![..]>()?;
                has_dotdot = true;
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
                continue;
            }

            let dim: Dim = input.parse()?;
            if has_dotdot {
                after_dotdot.push(dim);
            } else {
                before_dotdot.push(dim);
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        if has_dotdot {
            if before_dotdot.is_empty() && !after_dotdot.is_empty() {
                Ok(NumberList {
                    internal,
                    input: ShapeInput::Tail(after_dotdot),
                })
            } else if !before_dotdot.is_empty() && after_dotdot.is_empty() {
                Ok(NumberList {
                    internal,
                    input: ShapeInput::Head(before_dotdot),
                })
            } else if !before_dotdot.is_empty() && !after_dotdot.is_empty() {
                Ok(NumberList {
                    internal,
                    input: ShapeInput::Span {
                        head: before_dotdot,
                        tail: after_dotdot,
                    },
                })
            } else {
                Ok(NumberList {
                    internal,
                    input: ShapeInput::Tail(Vec::new()),
                })
            }
        } else {
            Ok(NumberList {
                internal,
                input: ShapeInput::List(before_dotdot),
            })
        }
    }
}

pub(crate) fn lit_to_typenum(
    n: usize,
    path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    if n == 0 {
        return quote! { #path typenum::UTerm };
    }
    let bit = if n.is_multiple_of(2) {
        quote! { #path typenum::B0 }
    } else {
        quote! { #path typenum::B1 }
    };
    let rest = lit_to_typenum(n / 2, path);
    quote! { #path typenum::UInt<#rest, #bit> }
}

fn render_dim(elem: &Dim, path: &proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    match elem {
        Dim::Dyn => quote! { usize },
        Dim::Lit(lit_int) => {
            let val: usize = lit_int.base10_parse().unwrap_or(0);
            // Emit typenum's binary representation directly.  This is
            // O(log N) and deliberately avoids typenum's finite alias
            // catalogue (U0..U4096, etc.).
            lit_to_typenum(val, path)
        }
        Dim::Path(p) if p.is_ident("usize") => quote! { usize },
        Dim::Path(p) => quote! { #path NamedDim<#p, usize> },
        Dim::ConstPath(p) => quote! { #path ConstDim<{ #p }> },
        Dim::Named { tag, extent } => {
            let extent = render_dim(extent, path);
            quote! { #path NamedDim<#tag, #extent> }
        }
    }
}

pub(crate) fn shape(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as NumberList);
    let internal = parsed.internal;
    let path = if internal {
        quote! { crate::prelude:: }
    } else {
        quote! { ::incin::prelude:: }
    };
    let types_path = if internal {
        quote! { crate::types:: }
    } else {
        quote! { ::incin::types:: }
    };

    let build_cons_chain = |dims: &[proc_macro2::TokenStream]| -> proc_macro2::TokenStream {
        let mut chain = quote! { #types_path Nil };
        for d in dims.iter().rev() {
            chain = quote! { #types_path DimCons<#d, #chain> };
        }
        chain
    };

    match parsed.input {
        ShapeInput::List(list) => {
            let output: Vec<_> = list.iter().map(|dim| render_dim(dim, &path)).collect();
            build_cons_chain(&output)
        }
        ShapeInput::Repeat { dim, count } => {
            let rendered = render_dim(&dim, &path);
            let output: Vec<_> = (0..count).map(|_| rendered.clone()).collect();
            build_cons_chain(&output)
        }
        ShapeInput::Tail(list) => {
            let _ = list;
            quote! { #path Dyn }
        }
        ShapeInput::Head(list) => {
            let _ = list;
            quote! { #path Dyn }
        }
        ShapeInput::Span { head, tail } => {
            let _ = (head, tail);
            quote! { #path Dyn }
        }
    }
    .into()
}
