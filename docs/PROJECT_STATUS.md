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
| CPU eager tensor execution | **Implemented but not yet verified against the canonical contract** | Stable tensor methods still depend on legacy operation-family traits | Stable CPU surface | Source audit and CPU package tests | FND-004, then FND-005 |
| Typed descriptor execution | **Partial** descriptor validation and execution | Descriptor coverage is incomplete and CPU adapters call legacy traits | Backend-authoring/experimental internals | Descriptor tests and source audit | FND-004 |
| Compiled execution | **Structural prototype** for capture, plans, and artifacts | No validated executable/run path | `experimental::compiled`, opt-in `compiled` feature | Containment test and compiled feature check | Deferred compiled CPU vertical slice |
| Constant folding and weight prepacking | **Intentionally unsupported** with typed errors | No transformations are implemented | `experimental::compiled` | `fnd000-test-compiled-containment.txt` | Deferred until canonical CPU descriptors |
| ONNX macro import | **Partial** stateless eager expansion | No initializers, control flow, custom domains, attributes, or broad opset coverage | `experimental::{model, import_model}` | Macro unit tests and FND-001 facade contracts | Real ONNX initializer/state loading (deferred) |
| ONNX initializer/state loading | **Intentionally unsupported** | Real byte/dtype/state loading is absent | No product surface | Macro fail-closed tests | Real ONNX initializer/state loading |
| Data pipeline | **Implemented but not verified by this foundation task** | Worker lifecycle, validation, resource, and integrity work is deferred | Preview/data APIs | Existing source only; no completion claim | Data-pipeline reliability (deferred) |
| Distributed execution | **Structural prototype** | No broad stable multi-node execution claim | Experimental/feature-gated | Source inspection only | Deferred until local semantics stabilize |
| CUDA, WGPU, and Metal execution | **Hardware-blocked** for this run | Required devices/platform libraries were not validated | Feature-gated backend surfaces | No hardware evidence claimed | Canonical CPU contract first |

## Active foundation sequence

FND-000 through FND-005 are executed in dependency order. A later foundation
task is not started until the prior task's acceptance gate is truthfully met.
FND-000 through FND-002 have passed their archived acceptance gates; FND-003 is
the next active task.
