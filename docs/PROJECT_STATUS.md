# Project Status

This document reports only what is supported by current source inspection or
archived command output. A feature can appear in source without being a stable,
verified product capability.

## Status vocabulary

- **Dynamically verified**: exercised successfully by an archived command from
  the identified checkout.
- **Implemented but unverified**: a substantive implementation exists, but the
  current remediation has not completed its required validation matrix.
- **Partial**: only a documented subset is implemented; unsupported cases fail
  explicitly.
- **Structural prototype**: types and graph structures exist without a complete
  executable product path.
- **Intentionally unsupported**: the path rejects requests rather than
  fabricating results or returning success.
- **Hardware-blocked**: verification requires hardware or platform libraries
  unavailable in the current environment.

## Current classification

| Subsystem | Implemented behavior | Known gaps | Public tier | Evidence | Next dependency |
|---|---|---|---|---|---|
| Core feature builds | **Complete and dynamically verified** for the archived no-default and `std` checks | The broader foundation sequence remains active | Stable core dependency | FND-000 check outputs | FND-001 facade |
| `incin` CPU feature build | **Complete and dynamically verified** for the archived CPU check and package tests | This does not prove the canonical descriptor architecture | Stable end-user API | FND-000 CPU check and test outputs | FND-004, then FND-005 |
| Workspace suite | **Dynamically verified** by the post-containment workspace run; no historical aggregate count is reused | Formatting remains non-clean under the current rustfmt baseline | Workspace validation | `fnd000-test-workspace-after-fixes.txt` | Per-task validation |
| Stable public facade | **Complete and dynamically verified** for the FND-001 allow-list and isolated consumer contracts | Semver comparison tooling is blocked by its forced all-feature rustdoc build; see FND-001 evidence | Stable root/prelude plus explicit `backend_authoring`, `experimental`, and feature-gated `test_utils` tiers | FND-001 public API, compile-contract, feature-matrix, Clippy, test, and rustdoc outputs | FND-002 invariant opacity |
| Invariant-bearing values and allocation arithmetic | **Complete and dynamically verified** for the FND-002 opacity, checked-construction, serialization, feature, compile-contract, Clippy, package, workspace, doctest, and rustdoc gates | The workspace-wide formatting baseline still reports pre-existing drift outside the task diff; accelerator execution remains hardware-blocked | Stable values plus backend-authoring/experimental internals | `audit-evidence/FND-002/` | FND-003 typed failure and rollback contracts |
| Typed failures, scalar conversion, and optimizer rollback | **Complete and dynamically verified** for FND-003 | Legacy free-form compatibility variants remain but are not used for new foundation paths; operator outputs are intentionally source-breaking `Result` values | Stable root/prelude plus backend contracts | `audit-evidence/FND-003/` | FND-004 operation semantics |
| Canonical operation semantics and descriptors | **Complete and dynamically verified** for FND-004: 174 exact identities declared once, typed `Descriptor<O>` per operation, per-operand rank contracts, fail-closed output inference, and exact-identity capability resolution | Execution is not migrated; this task freezes semantics only | Backend-authoring contract plus generated docs | `audit-evidence/FND-004/` | FND-005 CPU migration |
| Canonical execution path | **Complete and dynamically verified** for FND-005: `exec::dispatch` validates against real storage metadata, queries the exact capability row, derives output metadata, and dispatches to `Execute<Descriptor<O>>` | Reaching it is opt-in; the stable tensor surface does not yet use it | Backend-authoring internals | `audit-evidence/FND-005/` | Remaining FND-005 migration |
| CPU eager tensor execution | **Partially migrated**: 154 of the 161 backend-executable catalog operations execute canonically, each verified for forward and gradient parity against the legacy path | Stable tensor methods still depend on the legacy operation-family traits, and `Backend` still requires all nine as supertraits | Stable CPU surface | `audit-evidence/FND-005/cpu-migration-status.md` | Remaining FND-005 migration |
| Typed descriptor execution | **Partial** descriptor validation and execution | Every advertised CPU identity has an executor, proved at compile time; the other 23 backend-executable operations are reachable only through legacy traits, and 13 more sit at an `ExecutionSite` the `Execute` trait cannot carry at all | Backend-authoring/experimental internals | `audit-evidence/FND-005/summary.md` | Remaining FND-005 migration |
| Compiled execution | **Structural prototype** for capture, plans, and artifacts | No validated executable/run path | `experimental::compiled`, opt-in `compiled` feature | Containment test and compiled feature check | Deferred compiled CPU vertical slice |
| Constant folding and weight prepacking | **Intentionally unsupported** with typed errors | No transformations are implemented | `experimental::compiled` | `fnd000-test-compiled-containment.txt` | Deferred until canonical CPU descriptors |
| ONNX macro import | **Partial** stateless eager expansion | No initializers, control flow, custom domains, attributes, or broad opset coverage | `experimental::{model, import_model}` | Macro unit tests and FND-001 facade contracts | Real ONNX initializer/state loading (deferred) |
| ONNX initializer/state loading | **Intentionally unsupported** | Real byte/dtype/state loading is absent | No product surface | Macro fail-closed tests | Real ONNX initializer/state loading |
| Data pipeline | **Partial and dynamically verified** for non-zero batch construction and worker iteration | Broader lifecycle, resource, download, and integrity work is deferred | Preview/data APIs | FND-003 loader tests | Data-pipeline reliability (deferred) |
| Distributed execution | **Structural prototype** | No broad stable multi-node execution claim | Experimental/feature-gated | Source inspection only | Deferred until local semantics stabilize |
| CUDA, WGPU, Metal, and Candle execution | **Dynamically verified** for feature compilation; WGPU default-workspace tests ran in the available adapter environment | CUDA and Metal hardware execution was not run; no hardware availability is inferred from compilation | Feature-gated backend surfaces | FND-003 feature checks and workspace suite | Canonical CPU contract first |

## Active foundation sequence

FND-000 through FND-005 are executed in dependency order. A later foundation
task is not started until the prior task's acceptance gate is truthfully met.
FND-000 through FND-004 have passed their archived acceptance gates. **FND-005
is active and PARTIAL.** Its completion condition - that stable CPU tensor
methods no longer rely on the operation-family supertrait architecture - is not
met: `Backend` still requires all nine supertraits, and 154 of the 161
backend-executable catalog operations have a canonical CPU executor. Each of
the seven remaining is blocked by a limit of the descriptor or capability
contract rather than by nobody having written it, and
`cpu-migration-status.md` names which limit stops which operation.
`audit-evidence/FND-005/summary.md` records what was delivered and what
remains, and `audit-evidence/FND-005/cpu-migration-status.md` is generated from
the registrations so the migrated count cannot be overstated by hand.

The denominator is the backend-executable subset rather than the whole catalog.
Thirteen operations sit at an `ExecutionSite` that `Execute` cannot carry: they
write through an operand, produce storage on another backend, or act on
autograd state. Those need the execution contract changed before an executor
could exist, so counting them as pending migrations would overstate the
remaining work by roughly 30%. `ExecutionSite::blocking_reason` states which
reason applies to each.

`docs/FROZEN_FOUNDATIONS.md` names the parts of the architecture that are
finished and should not be rewritten while that work proceeds, and orders the
remaining steps by what blocks what.

The workspace suite at `3e9609e` reports **1433 passed, 0 failed, 1
ignored**. The ignored case requires a CUDA device. No historical aggregate
count is reused.

## CPU correctness pass

Separate from the migration, and finished. An audit of every dtype refusal and
every `Unsupported` site in the CPU backend turned up seven defects, four of
which returned a wrong answer with no error attached. All seven are fixed, each
with a test that fails against the previous code.

| Defect | Class |
|---|---|
| `group_norm`/`instance_norm` took statistics across the whole flattened batch instead of per sample | wrong answer above batch size 1, and every prior test used batch 1 |
| `matmul` and the axis extrema wrote `f32` whatever they read | wrong result dtype; the cause of `scaled_dot_product_attention` answering in `f32` for every operand |
| `argmax`, `argmin`, `argsort` and `topk` ignored their index dtype parameter | wrong result dtype, and `Tensor::argmax` could not succeed at all |
| `to_scalar`/`to_vec1` compared byte width rather than dtype | reinterpreted the bits, so `1.0f32` read as `u32` returned `1065353216` |
| `adamw_step` refused every dtype but `f32` | valid request refused |
| `batch_norm` refused training mode | valid request refused; blocked convolutional training |
| `conv2d` refused an anisotropic window | valid request refused |

Two refusals are deliberate and remain. `conv_transpose2d` still collapses its
window, because its `output_padding` is a fourth per-axis pair and belongs in
its own change. Training-mode `batch_norm` still does not update running
statistics, because they arrive as shared references and mutating through an
operand is a site the execution contract does not carry.

The Q8_0 refusals across storage, reductions, the tape, quantization and
creation were measured and left alone. A packed block format has no scalar
arithmetic or gradient identity without a dequantization step, so those are
boundaries rather than gaps.

The three GPU backends are the open coverage gap rather than a correctness one.
Each declares roughly 37 `TensorOps` methods unsupported, including every
comparison, `where_cond`, `masked_fill`, `gather`, `scatter`, `index_select`
and `scaled_dot_product_attention`, plus about half the elementwise catalog.

The FND-004 evidence records 16 formatter-drifted files; the actual count at
that commit was 22, and is 20 now. See `audit-evidence/FND-005/known-limitations.md`
for the correction. No drifted file is one either task changed.
