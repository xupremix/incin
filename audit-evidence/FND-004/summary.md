# FND-004 - Canonical operation semantics and descriptors

**Status: DONE**

**Commit: `93beef10b01ffba4f67d5ac08cb636922b8ff09c`**
(parent `2eb7fe7200b2bdfd713ef1c50419a182dc696e02`, FND-003)

The acceptance gate was run twice: on the pre-commit worktree (`*-gate.txt`) and
again on the commit above (`*-committed.txt`). Both reproduce **1348 passed, 0
failed, 1 ignored**. `cargo public-api -p incin` reports 756 items on the
committed hash, identical to the FND-003 baseline: FND-004 does not change the
stable facade. `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps`
exits 0 on the committed hash.

Freezes one canonical semantic identity per stable operation, gives each a
concrete typed descriptor, and couples that identity to capability reporting and
backend execution admission. It does **not** migrate execution; that is FND-005.

## Result

| Acceptance criterion | Verdict | Evidence |
|---|---|---|
| Every public semantic operation appears exactly once in the catalog | PASS | `test-results/test-operation-inventory-gate.txt`, `operation-inventory.md` |
| All 142 legacy operation-family methods have a reviewed descriptor mapping | PASS | `old-trait-to-descriptor.md`, machine-checked by `operation_inventory` |
| Descriptors preserve every attribute eager execution requires | PASS | `test-results/test-core-exec-gate.txt` |
| Capability docs generated from the same source | PASS | `test-results/test-generated-docs-gate.txt`, `docs/capabilities.md` |
| Capture retains descriptor semantics without backend storage | PASS | `test-results/test-core-gate.txt` (descriptor-schema and catalog suites) |
| No output shape/dtype/device is fabricated | PASS | `conformance-summary.md` §4 |
| Per-operand (not per-operation) rank validation | PASS | `conformance-summary.md` §2 |
| Family rows cannot make an exact query supported | PASS | `conformance-summary.md` §5 |
| Advertised layouts execute; unadvertised layouts return typed refusals | PASS | `test-results/test-capability-matrix-gate.txt` |
| Workspace suite passes | PASS | `test-results/test-workspace-gate.txt` |
| Metal feature compiles | PASS | `test-results/check-backend-metal-gate.txt` |
| Workspace formatter clean | **BLOCKED** | pre-existing drift; see `known-limitations.md` |

## Test counts

Reproduced from this checkout only. No historical count is reused.

| Suite | Result |
|---|---|
| `cargo test --workspace` | **1348 passed, 0 failed, 1 ignored** |
| `cargo test --doc --workspace` | 78 passed, 0 failed |
| `cargo test -p incin-core` | 461 passed, 0 failed |
| `cargo test -p incin-backends --features cpu` | 379 passed, 0 failed |
| `cargo test -p incin` | 119 passed, 0 failed |

The single ignored test is `every_generated_cuda_row_matches_real_execution_on_hardware`,
which requires a CUDA device.

## What is now frozen

1. **`crates/incin-core/src/operation_catalog.rs`**  -  the single declaration of
   174 exact operations. A macro-callback table consumed by the diagnostic
   identity, the descriptor generator, and the inventory. There is no second
   hand-maintained operation list.
2. **`OperationKind::<Exact>`**  -  exact identities, distinct from broad
   families. `is_exact()` separates them and a family is never a capability
   identity.
3. **`Descriptor<O>`**  -  one concrete, non-interchangeable Rust type per
   operation, carrying a typed attribute schema (`Conv2dAttributes`,
   `LayerNormAttributes`, `AdamWAttributes`, ...) rather than a string map.
   Fields are private; a compile-fail fixture proves they cannot be forged.
4. **`ValidatedInvocation<O>`**  -  opaque proof that input and output metadata
   were validated without touching storage.
5. **`CapturedDescriptor`**  -  storage-free serialization with identity and
   schema outside the payload, so a wrong-type decode fails closed.
6. **`operand_ranks`**  -  per-role rank contracts overriding the primary
   operand's window.
7. **Capability resolution**  -  exact-identity matching only; family fallback is
   removed from `CapabilityRegistry`.

## Blockers resolved in this task

The previous FND-004 attempt did not pass its gate. Its logs are retained in
`test-results/*-final.txt` and `*-rerun.txt`; they are **superseded**, not
deleted, and several of them record failures.

1. **Workspace tests failed.** Root cause was not what the earlier notes
   assumed. Strided reshape and matmul descriptor execution work correctly and
   the CPU capability rows advertising them are truthful  -  both strided tests
   pass. The real failures were assertions still expecting broad **family**
   identities after the executors began reporting **exact** ones. Fixed by
   correcting the assertions and, more importantly, by making WGPU and Metal
   report exact identities too, which they did not. No capability row was
   narrowed, because no unsupported claim was found.
2. **Metal `E0425`.** No longer reproduces; the feature compiles. Compilation
   only  -  no Metal hardware claim.
3. **Per-operation rank validation.** Replaced with per-operand roles. This was
   a live defect: a biased `Conv2dExact` could not validate at all.
4. **Empty operation inventory.** `operation-surface-inventory.txt` was
   zero-length. Replaced by a machine-checked test that parses the operation
   traits from source and fails when a method gains no catalog row, plus the
   generated `operation-inventory.md`.
5. **Profile overgeneralization.** Audited; the exceptions found are fixed and
   regression-tested (`conformance-summary.md` §3). Removed the unreachable
   `DTypeRule::BooleanResult`.
6. **Output inference.** Shape verification now fails closed on known inputs
   instead of accepting caller-supplied metadata.
7. **Capability truth.** Exact rows are coupled to executor admission across all
   four backends; unadvertised layouts and family fallback are regression-tested.

## Commands

Every command, working directory, timestamp, exit code and output path is in
`commands.log`. The gate was run twice: once on the pre-commit worktree and once
on the committed hash, so the evidence is not tied to an unspecified diff.

## Next

FND-005  -  migrate CPU eager execution to this descriptor contract.
