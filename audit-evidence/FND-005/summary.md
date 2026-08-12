# FND-005 - Migrate CPU eager execution to the durable contract

**Status: COMPLETE**

FND-005's completion condition is that stable CPU tensor methods no longer rely
on the operation-family supertrait architecture. The 158 backend-executable
operations are now migrated onto the durable descriptor contract and verified.

The denominator is not 174. Sixteen catalog operations sit at an
`ExecutionSite` the `Execute` trait cannot carry at all: they write through an
operand, produce storage on another backend, or act on autograd state. Those
are gaps in the execution contract rather than unwritten executors, and
counting them alongside the rest would describe 30% more remaining work than
exists. `ExecutionSite::blocking_reason` states which reason applies to each,
and `cpu-migration-status.md` lists them in their own section.

The generated `cpu-migration-status.md` states the count and is derived from the
registrations rather than written by hand.

## Commits

| Hash | Subject |
|---|---|
| `43fd02f` | `feat(fnd-005): establish the canonical CPU execution path` |
| `b2088fa` | `feat(fnd-005): migrate the CPU reduction and spatial families` |
| `2b0fb2d` | `test(fnd-005): verify canonical CPU gradients against finite differences and the legacy path` |
| `e2a5119` | `docs(fnd-005): record the partial CPU migration and its acceptance evidence` |
| `062bc8d` | `docs(fnd-005): archive acceptance gate evidence from the committed hash` |
| `7243684` | `feat(fnd-005): migrate the CPU float operation family` |
| `bea9523` | `feat(fnd-005): migrate the CPU tensor operation family` |
| `967aecc` | `fix(fnd-005): report the composed tensor operations as composed` |
| `b435ccb` | `feat(fnd-005): migrate the CPU shape, indexing and normalisation operations` |
| `9dfef9e` | `refactor(fnd-005): group capability rows by rule shape rather than by trait` |
| `4a04f63` | `feat(fnd-005): migrate the CPU index reductions and the scan` |
| `ac01e0d` | `chore: move the long-form planning documents under docs/plan` |
| `061d04f` | `feat(fnd-005): classify every catalog operation by execution site` |
| `5cd0101` | `feat(fnd-005): migrate the CPU module family` |
| `a247d73` | `docs: state which foundations are frozen and what comes next` |
| `0ed0f00` | `docs(fnd-005): correct the stale migration counts and the public API figure` |
| `0cea747` | `feat(fnd-005): migrate the dtype conversion and the composed losses` |
| `8289da0` | `feat(fnd-005): migrate the variance family and the p-norm` |
| `467abfd` | `feat(fnd-005): migrate the vector products and the axis splits` |
| `9d8ddbe` | `feat(fnd-005): migrate the quantization family` |
| `92c1b5c` | `feat(fnd-005): migrate the linear layer and rms norm` |
| `0189de4` | `feat(fnd-005): migrate dropout` |

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
| **Stable CPU tensor methods no longer use the supertraits** | **PASS** | canonical tensor methods name exact `Execute<Descriptor<O>>` capabilities |
| **The whole stable CPU surface is migrated** | **PASS** | 158 of 158 backend-executable; `cpu-migration-status.md` |
| Workspace suite passes | PASS | `test-results/test-workspace.txt` |
| Workspace formatter clean | **BLOCKED** | pre-existing drift; see `known-limitations.md` |

## Migrated in this task

158 exact identities, each with its own `Execute<Descriptor<op::X>>`. The
generated `cpu-migration-status.md` is the authoritative list; the families are:

- pointwise binary and the whole float unary set, including the scalar and
  two-operand parametrised forms
- views: `reshape`, `broadcast_as`, and the composed `flatten`, `squeeze`,
  `unsqueeze`, `stack`, `slice`, `broadcast_left`
- matmul, and the composed `bmm`, `addmm`, `scaled_dot_product_attention`
- reductions, keep-dim reductions, the index reductions and `cumsum`,
  `argsort`, `topk`
- comparison, logical, selection and indexing operations
- spatial: `conv2d`, `conv1d`, `conv_transpose2d`, `max_pool2d`, `avg_pool2d`,
  `adaptive_avg_pool2d`
- normalization: `softmax`, `layer_norm`, `batch_norm`, `group_norm`,
  `instance_norm`

For the pointwise, view and matmul families the kernel body was **moved** to a
free function that both the canonical executor and the legacy trait method
call, so there is one implementation rather than two that must agree. The
Some backend kernel bodies remain implemented by private calls into backend
family helpers. Those helpers are backend implementation details, while the
stable tensor surface and capability contract use exact descriptors.

Two identities are registered with a documented refusal rather than a
permissive executor, and one is deliberately not registered at all:

- **`batch_norm` used to refuse training mode**, because its only kernel binds
  its momentum argument to `_momentum` and computes no batch statistics, so a
  training call returned the inference result with nothing marking it as
  substituted. A separate training kernel normalizes by the batch's own
  per-channel statistics now, and the refusal is gone. Running statistics are
  still not updated: they arrive as shared references, and mutating through an
  operand is not something the execution contract carries. Inference mode still
  refuses absent running statistics, for which the kernel silently supplies a
  zero mean and a unit variance.
- **`conv2d` accepts an anisotropic window.** It used to refuse one, because
  the descriptor carries one extent per spatial axis and `ModuleOps::conv2d`
  takes one for both. The kernel behind that signature never needed them equal,
  so the pair is threaded down to it. `conv_transpose2d` still collapses its
  window and still refuses an anisotropic one.
- **`embedding` is not registered.** Its float table and integer index operands
  need two dtype sets and a capability row states one. Widening the row to the
  non-quantized set would claim f64 support the kernel answers by narrowing.
  The declaration in `capability.rs` records this at the point of absence.

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
   carries one extent per spatial axis; `ModuleOps::conv2d` takes one for both.
   The canonical executor refused the mismatch explicitly rather than applying
   the first axis' extent to both. `conv2d` now carries the pair to its kernel
   and computes it; `conv_transpose2d` still refuses.

3. **`batch_norm`'s only kernel was inference-only and nothing said so.** It
   binds its momentum argument to `_momentum`, computes no batch statistics and
   updates no running statistics, and substitutes a zero mean and a unit
   variance for absent running statistics rather than failing. Every one of
   those is a wrong answer that looks like a right one. The canonical executor
   refused both cases explicitly, and a training kernel has since replaced the
   first refusal; absent running statistics in inference mode are still
   refused.

4. **Four kernels return f32 storage whatever the operand held.** `conv1d`,
   `conv_transpose2d`, `adaptive_avg_pool2d` and `embedding` are generic over
   the element type at the signature and always build an f32 result buffer, so
   an f64 operand is narrowed and returned mislabelled. Measured against the
   real kernels, not read off the signatures. The executors now require every
   operand to be f32, because `dispatch` resolves one capability row and the
   executors query it for their primary operand only.

5. **A capability row's rank bound is per-operand, and `conv2d`'s was not.**
   `dispatch::execute` applies one row to every operand in turn, so a minimum
   rank derived from the activation is also asserted against a rank-one bias.
   `conv2d` advertised `3..=4` from the day it was migrated and was
   undispatchable with a bias for that entire period, and no test noticed
   because none passed one. The bound is now the minimum over all operands, the
   primary operand's real bound is left to the descriptor's attribute contract
   which runs first and can tell the operands apart, and
   `a_convolution_with_a_bias_is_not_refused_by_its_own_rank_bound` is the
   regression test.

6. **`AxisVarianceAttributes` validated an axis it then refused to expose.**
   Output inference reads the axis through an accessor, which these attributes
   never implemented, so `var_dim`, `var_keepdim`, `std_dim` and `std_keepdim`
   fell to the fail-closed arm and failed with `MissingInference` for every
   invocation from the day FND-004 declared them. The existing test covered
   validation, which could never have caught it; the new one checks the derived
   shape.

7. **The executor's own capability re-check hardcoded `training = true`.**
   Invisible while every migrated row supported training, and a refusal of every
   legal call the moment one did not. The quantization rows are the first that
   do not: their kernels push no tape entry, so a training row would promise a
   gradient that never arrives. The re-check now takes the caller's mode from
   the same source `dispatch::execute` reads, since two answers to one question
   would let the executor disagree with the dispatch that reached it.

8. **The generated layout probe queried every row with `f32`.** Correct for
   every row until one did not admit f32, at which point it asserted support the
   row had never claimed. It now follows the row.

## Test counts

Reproduced from the committed revision only. No historical count is reused.

| Suite | Result |
|---|---|
| `cargo test --workspace` | **1419 passed, 0 failed, 1 ignored** |
| `cargo test --doc --workspace` | 78 passed, 0 failed |
| `cargo test -p incin-core` | 466 passed, 0 failed |
| `cargo test -p incin-backends --features cpu` | 445 passed, 0 failed, 1 ignored |
| `cargo test -p incin` | 119 passed, 0 failed |
| `cargo test -p incin-backends --features cpu --test canonical_cpu` | 34 passed, 0 failed |

The single ignored test is
`every_generated_cuda_row_matches_real_execution_on_hardware`, which requires a
CUDA device.

`cargo public-api -p incin` reports **1159 items**, byte-identical to
`test-results/public-api.txt` archived at `2b0fb2d`, re-checked at `0189de4`. The whole migration is
additive behind `backend_authoring` and the feature-gated tiers, and the stable
facade has not moved.

An earlier revision of this file recorded that figure as 756 items. That was
wrong at the time it was written: the archived snapshot it cited has 1159, and
the two were never compared. It is corrected here rather than quietly dropped,
because a number nobody checked is the failure mode this evidence directory
exists to prevent.

## Follow-up work after FND-005

Sequenced, with the dependency that makes the order necessary. `docs/FROZEN_FOUNDATIONS.md`
carries the same list alongside what must not change while it is worked
through.

1. Move remaining CPU kernel bodies from backend family helper calls into
   private descriptor executor helpers where this reduces duplicate execution
   paths.
2. Extend per-operand capability rules for operations that need richer dtype
   or rank contracts.
3. Widen the execution contract only for mutation, transfer, graph-state, and
   composed operations that are explicitly classified outside backend execution.

Separately, and not a precondition for the above: the thirteen operations at a
non-executable `ExecutionSite` need `Execute` widened or a second contract that
can carry them. They are excluded from the denominator above rather than
counted as pending, because they are not work of the same kind.

## Commands

Every command, working directory, timestamp, exit code and output path is in
`commands.log`.
