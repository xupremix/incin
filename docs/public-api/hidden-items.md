# Hidden public items

`#[doc(hidden)]` is reserved for compatibility and compile-time plumbing that
must be reachable by macro expansion or a trait implementation but should not
be presented as normal consumer API. It is not a substitute for `pub(crate)`.

The current inventory has four reviewed groups:

- macro expansion support: `incin`, `incin-core`, and `incin-macros` support
  modules and exported helper macros;
- type-level proof helpers: shape and placement traits used by public typed
  operations;
- backend contract plumbing: hidden associated methods and unavailable
  variants required by backend implementations;
- compatibility spellings: legacy tensor operation entry points retained for
  source compatibility while the documented spelling is preferred.

The source inventory is:

- `crates/incin-backends/src/cpu/var.rs`
- `crates/incin-backends/src/dispatch.rs`
- `crates/incin-backends/src/lib.rs`
- `crates/incin-backends/src/target.rs`
- `crates/incin-core/src/dist/placement.rs`
- `crates/incin-core/src/exec/catalog.rs`
- `crates/incin-core/src/exec/proof.rs`
- `crates/incin-core/src/lib.rs`
- `crates/incin-core/src/nn/module.rs`
- `crates/incin-core/src/shapes/dim.rs`
- `crates/incin-core/src/shapes/idx.rs`
- `crates/incin-core/src/shapes/proof.rs`
- `crates/incin-core/src/shapes/rank.rs`
- `crates/incin-core/src/shapes/shape.rs`
- `crates/incin-core/src/tensor/backend/execute.rs`
- `crates/incin-core/src/tensor/ops/manipulation.rs`
- `crates/incin-core/src/tensor/ops/reduce.rs`
- `crates/incin-core/src/tensor/ops/unary.rs`
- `crates/incin-macros/src/lib.rs`
- `crates/incin/src/lib.rs`

The inventory is checked by `tools/check-hidden-items.py`. When an item becomes
ordinary consumer API, remove `#[doc(hidden)]` and document it normally. When
an item becomes implementation-only, make it private instead of adding another
hidden export.
