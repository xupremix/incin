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

## Occurrence review

Every hidden attribute is reviewed at its source location. The line number is
checked mechanically so a new hidden export cannot be added without an
inventory update. The reviewed items are compatibility or compile-time
plumbing; they are not part of the normal consumer facade.

| Source location | Classification |
| --- | --- |
| `crates/incin/src/lib.rs:246` | compatibility or compile-time plumbing |
| `crates/incin-backends/src/cpu/var.rs:95` | compatibility or compile-time plumbing |
| `crates/incin-backends/src/cpu/var.rs:109` | compatibility or compile-time plumbing |
| `crates/incin-backends/src/dispatch.rs:87` | compatibility or compile-time plumbing |
| `crates/incin-backends/src/dispatch.rs:136` | compatibility or compile-time plumbing |
| `crates/incin-backends/src/dispatch.rs:153` | compatibility or compile-time plumbing |
| `crates/incin-backends/src/lib.rs:73` | compatibility or compile-time plumbing |
| `crates/incin-backends/src/target.rs:350` | compatibility or compile-time plumbing |
| `crates/incin-backends/src/target.rs:559` | compatibility or compile-time plumbing |
| `crates/incin-backends/src/target.rs:578` | compatibility or compile-time plumbing |
| `crates/incin-backends/src/target.rs:606` | compatibility or compile-time plumbing |
| `crates/incin-core/src/dist/placement.rs:90` | compatibility or compile-time plumbing |
| `crates/incin-core/src/dist/placement.rs:100` | compatibility or compile-time plumbing |
| `crates/incin-core/src/dist/placement.rs:104` | compatibility or compile-time plumbing |
| `crates/incin-core/src/dist/placement.rs:108` | compatibility or compile-time plumbing |
| `crates/incin-core/src/dist/placement.rs:132` | compatibility or compile-time plumbing |
| `crates/incin-core/src/dist/placement.rs:170` | compatibility or compile-time plumbing |
| `crates/incin-core/src/exec/catalog.rs:1610` | compatibility or compile-time plumbing |
| `crates/incin-core/src/exec/proof.rs:239` | compatibility or compile-time plumbing |
| `crates/incin-core/src/lib.rs:49` | compatibility or compile-time plumbing |
| `crates/incin-core/src/nn/module.rs:692` | compatibility or compile-time plumbing |
| `crates/incin-core/src/nn/module.rs:700` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/dim.rs:179` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/dim.rs:192` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/dim.rs:195` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/dim.rs:198` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/dim.rs:396` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/idx.rs:113` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/idx.rs:282` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/proof.rs:34` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/rank.rs:45` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/shape.rs:13` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/shape.rs:259` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/shape.rs:318` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/shape.rs:488` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/shape.rs:510` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/shape.rs:585` | compatibility or compile-time plumbing |
| `crates/incin-core/src/shapes/shape.rs:639` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/backend/execute.rs:41` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/manipulation.rs:1117` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/manipulation.rs:1832` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2011` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2070` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2255` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2353` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2426` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2519` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/reduce.rs:195` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/reduce.rs:323` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/reduce.rs:459` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/reduce.rs:478` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/reduce.rs:697` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/reduce.rs:721` | compatibility or compile-time plumbing |
| `crates/incin-core/src/tensor/ops/unary.rs:173` | compatibility or compile-time plumbing |
| `crates/incin-macros/src/lib.rs:288` | compatibility or compile-time plumbing |
