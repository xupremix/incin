# Macro internals

How the macros work behind their reference entries - the grammars they parse,
the hygiene strategy every expansion follows, and the test suite that keeps
all of it honest. The user-level documentation is
[the macro reference](./macros.md) and each macro's rustdoc; this chapter is
about the machinery in
[`crates/incin-macros/src/`](../../../crates/incin-macros/src/lib.rs).

## The inventory

| Macro | Kind | Implementation |
|---|---|---|
| `s![...]` | proc macro | `shape.rs` - shape *types* |
| `shape![...]` | proc macro | `shape_value.rs` - shape *values* for targets |
| `tensor![...]` | proc macro | `tensor.rs` - literal construction |
| `idx![...]` / `i![...]` | proc macros | `idx.rs` / `index_expr.rs` - type-level vs runtime indexing |
| `axis![...]` | proc macro | `axis.rs` - axis selectors |
| `#[module]` | attribute | `module.rs` - visitor derivation |
| `model!` / `import_model!` | proc macros | `safetensors.rs` + `onnx.rs` - compile-time import |
| `dim!(...)` | `macro_rules!` | lives in `incin-core/src/shapes/dim.rs`, not a proc macro |

The distributed family (`mesh!`, `placement!`, `parallel!`,
`#[distributed_main]`) shares the same conventions and is feature-gated.

## Hygiene: absolute paths, one documented exception

Every expansion names what it needs through `::incin::prelude::...` (or
`::incin::advanced::...`) *absolutely*, so it resolves against the crate
rather than against whatever the caller has in scope. `s!` additionally
carries an internal marker (`s![@ ...]`, spelled `crate::prelude::...`) for
use inside `incin` itself:

```rust,ignore
// crates/incin-macros/src/shape.rs
let path = if internal {
    quote! { crate::prelude:: }
} else {
    quote! { ::incin::prelude:: }
};
```

This is not a style preference; it is load-bearing, and the failure it
prevents used to happen:

```rust,ignore
// crates/incin-macros/tests/compile_pass/hygiene.rs (abridged)
mod incin {
    pub mod prelude {
        pub struct Decoy;
    }
}
/// `s!` emits `typenum::UInt`, `typenum::UTerm`, `typenum::B0`, `typenum::B1`.
mod typenum {
    pub struct UTerm;
    pub struct B0;
    ...
}
```

Before CI-005, expansions resolved against relative paths, so a caller with a
module named `incin` (or `typenum`) won, and the error surfaced as a
nonsensical message pointing at the caller's own invocation. The test plants
decoys for every emitted name and asserts both directions of hygiene: the
macro does not resolve against them, and it does not capture them either.
`rename.rs` covers the other axis - calling the crate under an alias
(`use ::incin as renamed;`) with no glob import at all.

The one thing absolute paths cannot survive is a *package* rename in the
caller's manifest (`incin_x = { package = "incin" }`): `::incin` then names a
crate that is not there. Resolving the real name would require reading the
caller's manifest at expansion time, which the macro policy in `PROPOSALS.md`
forbids - so that limitation is documented rather than worked around.

## `s!`: grammar and expansion

`s!` parses each dimension into one of five forms
(`crates/incin-macros/src/shape.rs`), then renders a right-folded
`DimCons<..., Nil>` chain:

```rust,ignore
enum Dim {
    Dyn,                          // `dyn` or `_`
    Lit(syn::LitInt),             // integer literal -> binary typenum
    Path(syn::Path),              // named tag -> NamedDim<path, usize>
    ConstPath(syn::Path),         // `const PATH` -> ConstDim<{ PATH }>
    Named { tag: syn::Path, extent: Box<Dim> },  // `tag = extent`
}
```

Rejections are grammar errors with named messages (`s_rejects_a_non_path_dim`,
`s_rejects_a_repeat_without_a_count`). The `..` ellipsis forms (`Head`,
`Tail`, `Span`) currently render to plain `Dyn` - the conservative choice,
since a partially-known chain would promise more rank proof than the parser
verified.

`shape!` (`shape_value.rs`) is the value-level counterpart whose static/runtime
split is *syntactic*: an integer literal is a static axis, anything else -
including a named `const` - is a runtime axis. That is deliberately a weaker
answer, never a wrong one, and negative or fractional dimensions are rejected
at expansion ("a shape! dimension cannot be negative") instead of surfacing as
a confusing `usize` mismatch later.

`tensor!` infers shape from nesting depth exactly like a Rust array literal,
and dtype in a fixed order: explicit `; dtype:` clause, then numeric-literal
suffixes when every suffixed leaf agrees, then `i64` if every leaf is a bare
integer literal (matching `torch.tensor`), else `f32`. A ragged literal is an
expansion error naming the offending dimension, never a best-effort reshape.
An earlier revision accepted a `device:` clause inferred from token spelling,
which could not see through `let d = Wgpu::new(0);`; the heuristic was removed
rather than patched, and allocation placement belongs to targets.

## `axis!`: two vocabularies, kept apart

`axis!` accepts expressions (including negative literals), bare named tags,
and `named <Tag>`:

```rust,ignore
// crates/incin-macros/src/axis.rs (parser)
if input.peek(syn::Ident) && input.peek2(syn::Ident) {
    let keyword: syn::Ident = input.parse()?;
    if keyword != "named" {
        return Err(...expected `named <AxisTag>`...);
    }
    return Ok(Self::Named(input.parse()?));
}
```

Numeric items expand to typed cursors built by recursion over magnitude -
`Here`, `Next::<...>`, and a `ReverseAxis::<...>` wrapper for negatives - so
positive and negative selectors carry separate compile-time proofs while
runtime normalization still checks both against the real rank
([target API](./target_api.md)). `i!` is deliberately a separate macro with
its own vocabulary (`..`, ranges, negative indices, `-1` inference); axis
selection never changes indexing rules.

## `#[module]`: versioned argument grammar

The struct-level arguments are parsed against a fixed vocabulary, unknown keys
rejected by name:

```rust,ignore
// crates/incin-macros/src/module.rs (abridged)
const STRUCT_ARGUMENTS: &[&str] = &[
    "internal", "no_stats", "no_parameters", "no_state",
    "no_named_layers", "no_shape_info", "no_train_mode", "no_to_device",
];
```

That list exists because of how this macro used to fail: arguments were read
with `attr.to_string().contains(..)`, so `#[module(no_such_argument)]`
expanded as if written `#[module]`, and substring matching even accepted
`not_internal` as `internal` - the failure mode where a typo silently changes
behavior. The expansion walks every field of the struct, delegating to fields
implementing the parameter/state visitors and recursing into nested modules;
plain fields are skipped. Generated code routes `Vec`/`format!` through
`::incin::__macro_support` so even stdlib prelude names stay out of the
caller's namespace. `no_*` flags disable individual generated capabilities
for forward-only or specialized modules.

## Compile-time import

`model!` / `import_model!` read `.onnx` graphs and `.safetensors` headers at
compile time and emit typed module structs. Support is intentionally partial
and **fail-closed**: initializers, unknown rank, control flow, custom domains,
and unsupported nodes produce macro-expansion diagnostics instead of fabricated
code or values ([experimental surfaces](./experimental.md)).

## CI-005: the suite that keeps all of this true

The macro policy requires every public macro to provide compile-pass,
compile-fail, hygiene, rename, and rustfmt tests. Until CI-005,
`crates/incin-macros` had no `tests/` directory at all - `cargo test` ran
nothing and exited zero doing it. The harness is
[`tests/macro_suite.rs`](../../../crates/incin-macros/tests/macro_suite.rs):

- `tests/compile_pass/*.rs` and `tests/compile_fail/*.rs` run through
  trybuild;
- each compile-fail case has a row in `expected_reasons()` pinning the exact
  diagnostic wording, because a macro rejection carries no error code - the
  message *is* the contract the user reads;
- adding a case without pinning its reason fails
  `compile_fail_cases_fail_for_their_stated_reason`, a guard added after
  four cases elsewhere were found rotted into passing while asserting
  nothing;
- `hygiene.rs` and `rename.rs` are the decoy suites described above;
  `rustfmt_fixture.rs` pins expansion output formatting.

The same pattern guards `mesh_*`, `placement_*`, `parallel_attrs*`, and
`tensor_compile_fail` directories. When you extend any macro's grammar, the
expected workflow is: new pass case, new fail case with its pinned reason,
then the implementation - the suite is written so skipping the first two
steps fails visibly.
