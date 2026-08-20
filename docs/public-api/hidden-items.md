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
- `crates/incin-backends/src/target/ext.rs`
- `crates/incin-core/src/dist/placement.rs`
- `crates/incin-core/src/exec/catalog/shape_transform.rs`
- `crates/incin-core/src/exec/proof.rs`
- `crates/incin-core/src/lib.rs`
- `crates/incin-core/src/nn/module.rs`
- `crates/incin-core/src/shapes/dim.rs`
- `crates/incin-core/src/shapes/idx.rs`
- `crates/incin-core/src/shapes/proof.rs`
- `crates/incin-core/src/shapes/rank.rs`
- `crates/incin-core/src/shapes/shape.rs`
- `crates/incin-core/src/tensor/backend/execute.rs`
- `crates/incin-core/src/tensor/ops/manipulation/concat.rs`
- `crates/incin-core/src/tensor/ops/manipulation/reshape.rs`
- `crates/incin-core/src/tensor/ops/manipulation/transpose.rs`
- `crates/incin-core/src/tensor/ops/reduce.rs`
- `crates/incin-core/src/tensor/ops/unary.rs`
- `crates/incin-macros/src/lib.rs`
- `crates/incin/src/lib.rs`

The inventory is checked by `tools/check-hidden-items.py`. When an item becomes
ordinary consumer API, remove `#[doc(hidden)]` and document it normally. When
an item becomes implementation-only, make it private instead of adding another
hidden export.

## Occurrence review

Every hidden attribute is reviewed by item identity. The item path is
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
| `crates/incin/src/lib.rs::__macro_support` | A | Macro expansions need stable access to facade allocation and backend support. |
| `crates/incin-backends/src/cpu/var.rs::AssignFailureGuard` | C | Test-only fault injection owns its scoped guard; it is not consumer API. |
| `crates/incin-backends/src/cpu/var.rs::fail_assign_on` | C | Test-only fault injection is exported only for deterministic rollback tests. |
| `crates/incin-backends/src/dispatch.rs::DispatchStorage::Unavailable` | C | The unavailable storage variant keeps the backend-neutral enum total when Metal is disabled. |
| `crates/incin-backends/src/dispatch.rs::DispatchVar::Unavailable` | C | The unavailable variable variant keeps dispatch types total across feature sets. |
| `crates/incin-backends/src/dispatch.rs::DispatchGrads::Unavailable` | C | The unavailable gradient variant keeps dispatch types total across feature sets. |
| `crates/incin-backends/src/lib.rs::test_utils` | C | Test utilities are compiled only for deterministic backend failure tests. |
| `crates/incin-backends/src/target/ext.rs::allocate_row_major` | B | Backend implementors use the hidden allocation hook behind the public target contract. |
| `crates/incin-backends/src/target/ext.rs::finish` | B | Backend implementors use the hidden storage-finalization hook behind constructors. |
| `crates/incin-backends/src/target/ext.rs::generated_canonical` | B | Backend implementors use canonical zero-operand dispatch for generated operations. |
| `crates/incin-backends/src/target/ext.rs::canonical_creation` | B | Backend implementors use canonical creation dispatch for generated operations. |
| `crates/incin-core/src/dist/placement.rs::try_from_incin` | B | Placement implementors convert checked runtime placement metadata. |
| `crates/incin-core/src/dist/placement.rs::RANKS` | B | Placement implementors provide mesh rank cardinality used by validation. |
| `crates/incin-core/src/dist/placement.rs::SHARD_DEGREE` | B | Placement implementors provide shard degree used by validation. |
| `crates/incin-core/src/dist/placement.rs::PIPELINE_DEGREE` | B | Placement implementors provide pipeline degree used by validation. |
| `crates/incin-core/src/dist/placement.rs::RankedPlacement` | C | Ranked placement carries proof-backed rank metadata without exposing construction. |
| `crates/incin-core/src/dist/placement.rs::DynamicPlacement` | C | Dynamic placement carries runtime metadata without exposing construction. |
| `crates/incin-core/src/exec/catalog/shape_transform.rs::ShapeTransform` | C | Descriptor lowering selects transforms internally; callers cannot forge this enum. |
| `crates/incin-core/src/exec/proof.rs::__audit_or_panic` | C | Paranoid validation is an internal hook for auditing an already sealed proof. |
| `crates/incin-core/src/lib.rs::__macro_support` | A | Macros expanded inside the crate need the private support re-exports. |
| `crates/incin-core/src/nn/module.rs::seq_ty` | A | The exported sequence macro is a compatibility name used by generated code. |
| `crates/incin-core/src/nn/module.rs::seq_type` | A | The exported sequence macro is a compatibility name used by generated code. |
| `crates/incin-core/src/shapes/dim.rs::BroadcastChoice` | C | Broadcast type selection is solver machinery for public shape proofs. |
| `crates/incin-core/src/shapes/dim.rs::BroadcastSameNat` | C | Static broadcast aliases are solver machinery for public shape proofs. |
| `crates/incin-core/src/shapes/dim.rs::BroadcastRightNat` | C | Static broadcast aliases are solver machinery for public shape proofs. |
| `crates/incin-core/src/shapes/dim.rs::BroadcastStaticNat` | C | Static broadcast aliases are solver machinery for public shape proofs. |
| `crates/incin-core/src/shapes/dim.rs::__incin_dim_declare` | A | The dimension declaration macro must resolve from generated downstream code. |
| `crates/incin-core/src/shapes/idx.rs::ReshapeTargetSpec` | C | Reshape target collection is type-level proof machinery, not an extension point. |
| `crates/incin-core/src/shapes/idx.rs::SliceSpec` | C | Slice target collection is type-level proof machinery, not an extension point. |
| `crates/incin-core/src/shapes/proof.rs::of_ranked` | C | Proof-level selection supports typed validation without exposing implementation details. |
| `crates/incin-core/src/shapes/rank.rs::ShapeRank` | C | Shape rank computation is solver machinery for public shape proofs. |
| `crates/incin-core/src/shapes/shape.rs::ForwardCursor` | C | Forward cursor normalization prevents solver ambiguity in public shapes. |
| `crates/incin-core/src/shapes/shape.rs::LastDim` | C | Last-dimension selection is solver machinery for typed shape operations. |
| `crates/incin-core/src/shapes/shape.rs::RemoveLastDim` | C | Last-dimension removal is solver machinery for typed shape operations. |
| `crates/incin-core/src/shapes/shape.rs::SwapLastWith` | C | Structural cursor swapping is solver machinery for typed shape operations. |
| `crates/incin-core/src/shapes/shape.rs::SwapFirstWith` | C | Structural cursor swapping is solver machinery for typed shape operations. |
| `crates/incin-core/src/shapes/shape.rs::FlattenPrefix` | C | Prefix flattening is solver machinery for typed shape operations. |
| `crates/incin-core/src/shapes/shape.rs::FlattenSuffix` | C | Suffix flattening is solver machinery for typed shape operations. |
| `crates/incin-core/src/tensor/backend/execute.rs::execution_storage` | B | Backend authors provide the storage-erasure hook used by execution validation. |
| `crates/incin-core/src/tensor/ops/manipulation/reshape.rs::reshape_typed` | B | The explicit typed reshape spelling remains for source compatibility. |
| `crates/incin-core/src/tensor/ops/manipulation/transpose.rs::transpose_runtime` | C | Runtime transpose lowers the canonical selector API. |
| `crates/incin-core/src/tensor/ops/manipulation/reshape.rs::flatten_runtime` | C | Runtime flatten lowers the canonical selector API. |
| `crates/incin-core/src/tensor/ops/manipulation/reshape.rs::flatten_range` | B | The signed flatten spelling remains for source compatibility. |
| `crates/incin-core/src/tensor/ops/manipulation/concat.rs::try_concat` | B | The legacy concatenation spelling remains for source compatibility. |
| `crates/incin-core/src/tensor/ops/manipulation/concat.rs::concat_axis` | B | The legacy signed-axis concatenation spelling remains for source compatibility. |
| `crates/incin-core/src/tensor/ops/manipulation/concat.rs::stack_structural` | C | Structural stacking lowers the canonical stack API. |
| `crates/incin-core/src/tensor/ops/manipulation/concat.rs::try_stack` | B | The legacy dynamic stacking spelling remains for source compatibility. |
| `crates/incin-core/src/tensor/ops/reduce.rs::sum_at` | C | Structural reduction lowers typed reduction proofs. |
| `crates/incin-core/src/tensor/ops/reduce.rs::sum_keepdim_at` | C | Structural keep-dimension reduction lowers typed proofs. |
| `crates/incin-core/src/tensor/ops/reduce.rs::sum_runtime_ranked` | C | Runtime ranked reduction lowers the canonical reduction API. |
| `crates/incin-core/src/tensor/ops/reduce.rs::sum_keepdim_runtime_ranked` | C | Runtime ranked keep-dimension reduction lowers the canonical API. |
| `crates/incin-core/src/tensor/ops/reduce.rs::argmax_runtime` | C | Runtime argmax lowers the canonical reduction API. |
| `crates/incin-core/src/tensor/ops/reduce.rs::argmin_runtime` | C | Runtime argmin lowers the canonical reduction API. |
| `crates/incin-core/src/tensor/ops/unary.rs::neg` | B | The old negation spelling remains as a documented compatibility path. |
| `crates/incin-macros/src/lib.rs::impl_layer_args` | A | The exported helper macro is the expansion entry point for layer arguments. |
