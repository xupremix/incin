# FND-005 - Migrate CPU eager execution to the durable contract

**Status: PARTIAL**

FND-005's completion condition is that stable CPU tensor methods no longer rely
on the operation-family supertrait architecture. That condition is **not met**.
What this task delivered is the execution architecture that condition depends
on, plus the first 24 of 174 operations migrated onto it and verified.

Nothing here should be read as "the CPU is done". The generated
`cpu-migration-status.md` states the count, is derived from the registrations
rather than written by hand, and a test fails if it drifts.

## Commits

| Hash | Subject |
|---|---|
| `43fd02f` | `feat(fnd-005): establish the canonical CPU execution path` |
| `b2088fa` | `feat(fnd-005): migrate the CPU reduction and spatial families` |
| `2b0fb2d` | `test(fnd-005): verify canonical CPU gradients against finite differences and the legacy path` |

Parent: `c538539` (FND-004). No earlier commit was amended, squashed or
recreated.

## What FND-004 left, and what this fixed

FND-004 froze 174 exact identities, one typed `Descriptor<O>` each, and one
exact capability row each. It had **no production consumer**: outside the test
suite, nothing built a `Descriptor<O>` or ran one.
`ValidatedInvocation::validate` was reachable only from tests. The stable tensor
surface still called the operation-family traits directly, so the frozen
semantics constrained nothing at runtime.

`incin_core::exec::dispatch` is that consumer. `dispatch::execute::<O, B>` is
the single path from a typed operation to native execution, and each property
FND-005 requires of it is structural rather than conventional:

| Required property | How it holds |
|---|---|
| Support is explicit | `B: Execute<Descriptor<O>>` is a compile-time fact, and the exact capability row is queried before launch |
| Validation precedes execution | The descriptor is validated against the operands' real storage metadata |
| No silent unsupported | There is no default method to fall through; refusals are typed `UnsupportedReason` |
| Outputs are derived | The caller passes attributes and operands only - there is no output argument to get wrong |
| Dispatch uses the exact registry | `CapabilityQuery` is keyed on `O::ID`, and families cannot satisfy it |
| Capture retains the descriptor | The value handed to the backend is the value a compiler would record |

## Result

| Acceptance criterion | Verdict | Evidence |
|---|---|---|
| A production path exists from a canonical operation to native execution | PASS | `exec/dispatch.rs`, `test-results/test-canonical-cpu.txt` |
| Descriptor validation runs before execution | PASS | `validation_runs_before_the_backend_is_reached` |
| Output metadata is derived, never accepted | PASS | `the_canonical_path_derives_output_metadata_from_the_inputs` |
| Every advertised CPU identity has an executor | PASS | compile-time proof in `cpu::canonical`; negative-tested |
| Migrated operations match the legacy path exactly | PASS | `test-results/test-canonical-cpu.txt` (forward), `canonical_and_legacy_gradients_are_identical` (backward) |
| Gradients match finite differences | PASS | four checks in `cpu::canonical::tests`; see `known-limitations.md` for what they do and do not resolve |
| Capability output is generated from the implementations | PASS | one declaration feeds the rows, the legacy executors and the canonical executors |
| Non-CPU backends preserve compilation without broadened claims | PASS | `test-results/check-backend-*.txt`; WGPU training rows are `f32` only |
| **Stable CPU tensor methods no longer use the supertraits** | **NOT MET** | `Backend` still requires all nine; 850 references across 75 files |
| **The whole stable CPU surface is migrated** | **NOT MET** | 24 of 174; `cpu-migration-status.md` |
| Workspace suite passes | PASS | `test-results/test-workspace.txt` |
| Workspace formatter clean | **BLOCKED** | pre-existing drift; see `known-limitations.md` |

## Migrated in this task

24 exact identities, each with its own `Execute<Descriptor<op::X>>`:

- pointwise binary: `add`, `sub`, `mul`, `div`
- views: `reshape`, `broadcast_as`
- matmul: `matmul`
- reductions: `sum_all`, `mean_all`, `max_all`, `min_all`, `prod_all`,
  `sum_dim`, `mean_dim`, `max_dim`, `min_dim`, `prod_dim`, `sum_keepdim`,
  `mean_keepdim`, `max_keepdim`, `min_keepdim`
- spatial: `conv2d`, `max_pool2d`, `avg_pool2d`

For the pointwise, view and matmul families the kernel body was **moved** to a
free function that both the canonical executor and the legacy trait method
call, so there is one implementation rather than two that must agree. The
reduction and spatial executors still reach the bodies through `ReductionOps`
and `ModuleOps`; that is the migration's temporary compatibility adapter, it is
private to `cpu::canonical`, and it is the only remaining call from the
canonical path into the legacy families.

## Defects found and fixed

1. **The exact view rows were narrower than the family rows they replace.**
   `ReshapeExact` and `BroadcastAs` were registered with `training = false` on
   every backend, while the legacy `Reshape`/`Broadcast` family rows carried
   both a training and a non-training row. But the CPU view kernels record a
   gradient - `reshape_storage` pushes a tape entry unconditionally. A training
   reshape therefore resolved through the family row and would have stopped
   resolving the moment FND-005 removed it. Fixed by generating the training
   half of each exact row, restricted to each backend's own float dtypes so no
   support claim is broadened; WGPU's training rows are `f32` only, matching its
   non-training rows.

2. **An anisotropic convolution window had no honest answer.** The descriptor
   carries one extent per spatial axis; the routed CPU kernel takes one for
   both. The canonical executor refuses the mismatch explicitly rather than
   applying the first axis' extent to both.

## Test counts

Reproduced from the committed revision only. No historical count is reused.

| Suite | Result |
|---|---|
| `cargo test --workspace` | **1372 passed, 0 failed, 1 ignored** |
| `cargo test --doc --workspace` | 78 passed, 0 failed |
| `cargo test -p incin-core` | 461 passed, 0 failed |
| `cargo test -p incin-backends --features cpu` | 403 passed, 0 failed, 1 ignored |
| `cargo test -p incin` | 119 passed, 0 failed |
| `cargo test -p incin-backends --features cpu --test canonical_cpu` | 16 passed, 0 failed |

The single ignored test is
`every_generated_cuda_row_matches_real_execution_on_hardware`, which requires a
CUDA device.

FND-004 recorded 1348. The 24 added here are the 16 canonical CPU conformance
tests, the 5 gradient tests in `cpu::canonical::tests`, and the 3
migration-status tests.

`cargo public-api -p incin` reports **756 items, unchanged** from the FND-003 and
FND-004 baseline.

## What remains for FND-005

Sequenced, with the dependency that makes the order necessary:

1. **Migrate the remaining 150 operations onto `Execute<Descriptor<op::X>>`**,
   moving each kernel body down as the pointwise family already did. Until the
   whole surface is migrated, step 2 cannot start, because a tensor method
   cannot depend on a capability that does not exist.
2. **Remove the nine operation-family supertraits from `Backend`** and give
   each stable tensor method a bound naming only the capability it uses. This
   is source-breaking for every backend implementation and changes the `incin`
   facade; it is the step that actually ends the dual architecture.
3. **Delete the broad family capability rows** (`Pointwise`, `Reduction`,
   `Reshape`, `MatMul`, `Conv2d`, `Pool2d`, `Storage`, `Fill`, `Random`,
   `Normalization`, `Broadcast`) and the grouped `Execute<MatMulSpec>` adapters.
4. **Delete the compatibility adapter** in `cpu::canonical` and the
   `the_migration_is_recorded_as_incomplete` test, which is written to fail once
   the catalog is fully migrated so the completion claim must be a deliberate
   edit.

## Commands

Every command, working directory, timestamp, exit code and output path is in
`commands.log`.
