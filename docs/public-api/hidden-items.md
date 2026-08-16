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
inventory update. The classification is deliberately narrow:

- A: required macro expansion ABI. The item must remain reachable from code
  generated in a consumer crate.
- B: deliberate extension or compatibility ABI. The item is a documented
  implementation hook for backend authors or a retained source-compatible
  spelling, not ordinary facade API.
- C: implementation machinery. The item is reachable because a public typed
  contract needs it, but it is not an intended extension point or compatibility
  spelling.

Every row records why making the item private would break a supported
expansion, backend implementation, compatibility path, or type-level proof.

| Source location | Class | Reason |
| --- | --- | --- |
| `crates/incin/src/lib.rs:246` | A | Macro expansions need stable access to facade allocation and backend support. |
| `crates/incin-backends/src/cpu/var.rs:95` | C | Test-only fault injection owns its scoped guard; it is not consumer API. |
| `crates/incin-backends/src/cpu/var.rs:109` | C | Test-only fault injection is exported only for deterministic rollback tests. |
| `crates/incin-backends/src/dispatch.rs:87` | C | The unavailable storage variant keeps the backend-neutral enum total when Metal is disabled. |
| `crates/incin-backends/src/dispatch.rs:136` | C | The unavailable variable variant keeps dispatch types total across feature sets. |
| `crates/incin-backends/src/dispatch.rs:153` | C | The unavailable gradient variant keeps dispatch types total across feature sets. |
| `crates/incin-backends/src/lib.rs:73` | C | Test utilities are compiled only for deterministic backend failure tests. |
| `crates/incin-backends/src/target.rs:350` | B | Backend implementors use the hidden allocation hook behind the public target contract. |
| `crates/incin-backends/src/target.rs:559` | B | Backend implementors use the hidden storage-finalization hook behind constructors. |
| `crates/incin-backends/src/target.rs:578` | B | Backend implementors use canonical zero-operand dispatch for generated operations. |
| `crates/incin-backends/src/target.rs:606` | B | Backend implementors use canonical creation dispatch for generated operations. |
| `crates/incin-core/src/dist/placement.rs:90` | B | Placement implementors convert checked runtime placement metadata. |
| `crates/incin-core/src/dist/placement.rs:100` | B | Placement implementors provide mesh rank cardinality used by validation. |
| `crates/incin-core/src/dist/placement.rs:104` | B | Placement implementors provide shard degree used by validation. |
| `crates/incin-core/src/dist/placement.rs:108` | B | Placement implementors provide pipeline degree used by validation. |
| `crates/incin-core/src/dist/placement.rs:132` | C | Ranked placement carries proof-backed rank metadata without exposing construction. |
| `crates/incin-core/src/dist/placement.rs:170` | C | Dynamic placement carries runtime metadata without exposing construction. |
| `crates/incin-core/src/exec/catalog.rs:1610` | C | Descriptor lowering selects transforms internally; callers cannot forge this enum. |
| `crates/incin-core/src/exec/proof.rs:239` | C | Paranoid validation is an internal hook for auditing an already sealed proof. |
| `crates/incin-core/src/lib.rs:49` | A | Macros expanded inside the crate need the private support re-exports. |
| `crates/incin-core/src/nn/module.rs:692` | A | The exported sequence macro is a compatibility name used by generated code. |
| `crates/incin-core/src/nn/module.rs:700` | A | The exported sequence macro is a compatibility name used by generated code. |
| `crates/incin-core/src/shapes/dim.rs:179` | C | Broadcast type selection is solver machinery for public shape proofs. |
| `crates/incin-core/src/shapes/dim.rs:192` | C | Static broadcast aliases are solver machinery for public shape proofs. |
| `crates/incin-core/src/shapes/dim.rs:195` | C | Static broadcast aliases are solver machinery for public shape proofs. |
| `crates/incin-core/src/shapes/dim.rs:198` | C | Static broadcast aliases are solver machinery for public shape proofs. |
| `crates/incin-core/src/shapes/dim.rs:396` | A | The dimension declaration macro must resolve from generated downstream code. |
| `crates/incin-core/src/shapes/idx.rs:113` | C | Reshape target collection is type-level proof machinery, not an extension point. |
| `crates/incin-core/src/shapes/idx.rs:282` | C | Slice target collection is type-level proof machinery, not an extension point. |
| `crates/incin-core/src/shapes/proof.rs:34` | C | Proof-level selection supports typed validation without exposing implementation details. |
| `crates/incin-core/src/shapes/rank.rs:45` | C | Shape rank computation is solver machinery for public shape proofs. |
| `crates/incin-core/src/shapes/shape.rs:13` | C | Forward cursor normalization prevents solver ambiguity in public shapes. |
| `crates/incin-core/src/shapes/shape.rs:259` | C | Last-dimension selection is solver machinery for typed shape operations. |
| `crates/incin-core/src/shapes/shape.rs:318` | C | Last-dimension removal is solver machinery for typed shape operations. |
| `crates/incin-core/src/shapes/shape.rs:488` | C | Structural cursor swapping is solver machinery for typed shape operations. |
| `crates/incin-core/src/shapes/shape.rs:510` | C | Structural cursor swapping is solver machinery for typed shape operations. |
| `crates/incin-core/src/shapes/shape.rs:585` | C | Prefix flattening is solver machinery for typed shape operations. |
| `crates/incin-core/src/shapes/shape.rs:639` | C | Suffix flattening is solver machinery for typed shape operations. |
| `crates/incin-core/src/tensor/backend/execute.rs:41` | B | Backend authors provide the storage-erasure hook used by execution validation. |
| `crates/incin-core/src/tensor/ops/manipulation.rs:1117` | B | The explicit typed reshape spelling remains for source compatibility. |
| `crates/incin-core/src/tensor/ops/manipulation.rs:1832` | C | Runtime transpose lowers the canonical selector API. |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2011` | C | Runtime flatten lowers the canonical selector API. |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2070` | B | The signed flatten spelling remains for source compatibility. |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2255` | B | The legacy concatenation spelling remains for source compatibility. |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2353` | B | The legacy signed-axis concatenation spelling remains for source compatibility. |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2426` | C | Structural stacking lowers the canonical stack API. |
| `crates/incin-core/src/tensor/ops/manipulation.rs:2519` | B | The legacy dynamic stacking spelling remains for source compatibility. |
| `crates/incin-core/src/tensor/ops/reduce.rs:195` | C | Structural reduction lowers typed reduction proofs. |
| `crates/incin-core/src/tensor/ops/reduce.rs:323` | C | Structural keep-dimension reduction lowers typed proofs. |
| `crates/incin-core/src/tensor/ops/reduce.rs:459` | C | Runtime ranked reduction lowers the canonical reduction API. |
| `crates/incin-core/src/tensor/ops/reduce.rs:478` | C | Runtime ranked keep-dimension reduction lowers the canonical API. |
| `crates/incin-core/src/tensor/ops/reduce.rs:697` | C | Runtime argmax lowers the canonical reduction API. |
| `crates/incin-core/src/tensor/ops/reduce.rs:721` | C | Runtime argmin lowers the canonical reduction API. |
| `crates/incin-core/src/tensor/ops/unary.rs:173` | B | The old negation spelling remains as a documented compatibility path. |
| `crates/incin-macros/src/lib.rs:288` | A | The exported helper macro is the expansion entry point for layer arguments. |
