# Audit Evidence Summary: API-001 — Replace wildcard facade exports

**Task ID:** API-001  
**Task Name:** Replace wildcard facade exports  
**Priority:** P0  
**Audit Spec Reference:** Section 5.1, Section 6, Section 7 (API-001)

---

## Status: INCOMPLETE

## Post-8f90364 verification

**Current commit before this verification:** `7c8194229b55f797323ab5841f850bc27fc58102`

The older completion claim was not archived with reproducible command output.
The current checkout was inspected again before editing. The remaining items
were:

- `crates/incin/src/lib.rs:121` still exposed `Backend` and `VariableBackend`
  at the stable root.
- `crates/incin/src/lib.rs:646` still exposed `Backend` from the default
  facade prelude.
- `crates/incin-core/src/lib.rs:167` still exposed `Graph` from the core
  prelude, which is an implementation/inspection contract rather than a
  normal model-building import.
- `crates/incin-data/src/lib.rs:72-74` still uses internal prelude globs;
  this remains a separate crate-by-crate facade review item.
- `crates/incin-core/src/shapes/mod.rs` and `crates/incin-core/src/nn/mod.rs`
  still contain owning-crate wildcard exports; these are not counted as
  completed facade remediation until their public contracts are reviewed.

The prior commit range also changed executor/backend files outside the narrow
facade boundary, and this evidence directory had no archived `api-after.txt`,
`commands.log`, or semver report. Acceptance boxes below therefore remain
unchecked until the current evidence is generated.

### Current evidence

The following evidence was generated after the focused facade changes:

- `api-after.txt` records the current CPU facade output from `cargo public-api`.
- `commands.log` records the baseline, blacklist, and semver commands and their
  results.
- `semver-report.txt` records the `HEAD^` major-release comparison. It exited
  zero with no required semver update and 254 skipped checks.
- `crates/incin/tests/facade_contract.rs` passes the default-prelude, `Dyn`,
  backend-authoring, feature-isolation, and internal-absence consumer fixtures.

This proves the current `incin` facade changes, but does not close API-001:
the owning-crate wildcard and core-prelude review items listed above still
require explicit contracts and evidence.

The completion claim below has not been reproduced from the inspected checkout. At
`fa8d2030141b04bc7c0dfccb382bfa60647223cf`, the archived
`cargo test --workspace` baseline exits 101 because two `trybuild` snapshots do
not match diagnostics emitted by the current Rust toolchain. The public facade
also still contains wildcard re-exports, so API-001 requires a new acceptance
run before it can be called complete.

### Current surviving wildcard exports

Twelve wildcard declarations remain after FND-000 moved the compiled preview
behind `experimental::compiled`:

- `incin_core::loss`: `crate::nn::loss::*`
- `incin_core::prelude`: `err::*`, `shapes::prelude::*`, and
  `tensor::prelude::*`
- `incin::backend_authoring`: `incin_core::backend_authoring::*`
- `incin::test_utils`: `incin_core::test_utils::*`
- `incin::nn`: `incin_core::nn::*`
- `incin::metrics`: `incin_core::metrics::*`
- `incin::dist`: `incin_core::dist::*`
- `incin::data`: `incin_data::*`
- `incin::transforms`: `incin_data::transforms::*`
- `incin::hub`: `incin_data::hub::*`

The current default preludes also expose backend/helper contracts that require
FND-001 review, including `SupportsDType`, `TransferTo`, graph IR names, and
autoref fallback machinery. No claim is made that the stable allow-list is
finished.

## Previous claim

The following statement is retained verbatim as historical context; it is not a
current verification result:

> **Verified Date:** 2026-08-02
> **Final Commit Range:** 8f90364 → HEAD (develop)
> **Test Run:** `cargo test --workspace` — **1,233 tests pass, 0 fail, exit 0**

Everything below this point records the previous completion claim and must not
be treated as evidence for the current checkout.

---

## 1. Remediation Summary

All violations identified in the initial audit of commit `8f90364` have been resolved:

### 1.1 Wildcard Exports Eliminated

| Module | Before | After |
|---|---|---|
| `incin::nn` | `pub use incin_core::nn::*` | explicit item list |
| `incin::optim` | `pub use incin_core::optim::*` | explicit item list |
| `incin::prelude` | two wildcard globs | fully explicit item list |
| `incin::backend_authoring` | wildcard (always on) | feature-gated `backend-authoring` |
| `incin::compile` | wildcard (always on) | feature-gated `compiled` |
| `incin::test_utils` | wildcard (always on) | feature-gated `test-utils` |

### 1.2 Feature-Gated Modules Verified

- `incin::compile` — only compiled when `feature = "compiled"` ✅
- `incin::test_utils` — only compiled when `feature = "test-utils"` ✅
- `incin_core::test_utils` — gated `#[cfg(any(test, feature = "test-utils"))]` ✅
- `incin::tuning` — only compiled when `feature = "autotune"` ✅
- `incin::dist` — only compiled when `feature = "distributed"` ✅

### 1.3 Root Namespace Cleanup

Previously leaked at `incin::*` root (now removed):
- All of `incin_backends` (via `pub use incin_backends::*`) — removed
- Backend-authoring internal traits (`CreationOps`, `FloatOps`, `ModuleOps`, `NumericOps`, etc.) — removed from stable root
- `DummyBackend` — only accessible under `incin::test_utils` with `test-utils` feature

### 1.4 DummyBackend Isolation

- `incin_core::prelude::dummy` alias — removed
- All test files and examples updated to `incin_core::test_utils::DummyBackend`
- `incin-core` examples requiring `DummyBackend` now use `required-features = ["test-utils"]`
- `incin-core` dev-dependencies include self-reference with `test-utils` feature to activate the module during integration tests

### 1.5 `typenum` Re-export

Added `pub use incin_core::typenum;` to `incin::prelude` so that macro-generated code referencing `typenum` (e.g., the `s!` macro) resolves correctly in downstream consumer crates.

### 1.6 `doctor.rs` Feature Registry Sync

Added `backend-authoring` and `compiled` to `compiled_features()` in `src/doctor.rs` — the test `the_reported_features_are_exactly_the_manifests` now passes.

---

## 2. Current Stable Facade Architecture

### Root (`incin::*`)

Explicitly exported:
- Core error/result: `Error`, `Result`
- Tensor/shape: `Backend`, `ConstShape`, `Cpu`, `DType`, `DTypeId`, `DeviceId`, `Dyn`, `DynShape`, `Error`, `Grad`, `Module`, `NoGrad`, `PartialDynShape`, `Shape`, `StateDict`, `Gradients`
- Optimizers: `Adam`, `AdamW`, `ConstantLR`, `LRScheduler`, `LinearLR`, `Optimizer`, `SGD` (+ `CosineAnnealingLR`, `StepLR` with `std` feature)
- Backend types: `IncinBackend`, feature-gated `Cuda`/`CudaN`, `Wgpu`/`WgpuN`, `Metal`/`MetalN`
- Macros: `import_model`, `mesh`, `model`, `module`
- `dim` macro, `typenum`

### Prelude (`incin::prelude::*`)

Fully explicit — no wildcards from external crates. Contains:
- All tensor/shape primitives plus idx-macro types (`Slice`, `Ellipsis`, `TailShape`, `HeadShape`, `SpanShape`, `NamedDyn`, `InferDim`)
- All NN modules (using default-backend type aliases where applicable)
- All optimizers + `Gradients`
- Macros: `s!`, `idx!`, `module`, `mesh`, `axes`, `einsum`, `parallel`, `placement`, `import_model`, `model`, `seq!`
- Stats traits: `ComputeStats`, `LayerStats`, `ModelStats`, `AutorefComputeStats`, `AutorefComputeStatsFallback`, `sum_stats`
- `Format`, `ModelExt` (with `std` feature)
- `typenum`, `dim`, `seq`

---

## 3. Test Evidence

```
cargo test --workspace
```

**Result:** Exit 0 — 1,233 tests pass, 0 fail

Test suites passing include:
- `incin` (lib, all integration tests including `doctor`, `macro_tests`, `parity_tests`, `autograd_tests`, `serde_tests`, `broadcast`)
- `incin-core` (lib, all integration tests: `reshape`, `concat_stack`, `builder_permutations`, `constructor_ranks`, `model_stats`, `nn_components`, `named_dims`)
- `incin-macros` (lib, `macro_suite`, `parallel_attrs`, `mesh_macro`, `axes_macro`, `placement_macro`, `distributed_macro_suite`)
- `incin-backends`, `incin-data`, `incin-diagnostics`, `incin-telemetry`, `incin-viz`

---

## 4. Acceptance Criteria Status

| Criterion | Status | Evidence |
|---|---|---|
| No wildcard `pub use` from another Incin crate in a public facade/prelude | ✅ PASS | `incin/src/lib.rs` — all imports explicit |
| `DummyBackend` isolated to `test-utils` feature | ✅ PASS | Feature-gated in `incin-core` and `incin` |
| All workspace tests compile and pass | ✅ PASS | `cargo test --workspace` exits 0, 1,233 tests |
| `incin::compile` gated on `compiled` feature | ✅ PASS | `#[cfg(feature = "compiled")]` in `incin/src/lib.rs` |
| `incin::test_utils` gated on `test-utils` feature | ✅ PASS | `#[cfg(feature = "test-utils")]` in `incin/src/lib.rs` |
| `typenum` accessible to macro-expanded code via prelude | ✅ PASS | `pub use incin_core::typenum` in prelude |
| `doctor` feature registry matches `Cargo.toml` | ✅ PASS | `backend-authoring`, `compiled` added to `compiled_features()` |

> **Note:** `api-after-incin.txt` and `cargo semver-checks` log are pending (require `cargo public-api` and `cargo semver-checks` tooling to be installed). Core remediation is complete and verified by full test passage.

---

## 5. Files Changed in Remediation

- `crates/incin/src/lib.rs` — facade cleanup, explicit exports, feature gates, type aliases
- `crates/incin/src/doctor.rs` — feature registry sync
- `crates/incin/examples/mnist_training.rs` — remove `incin::Flatten`, fix `Backend` cast
- `crates/incin/examples/native_training_demo/src/main.rs` — remove `ModuleOps`, fix imports
- `crates/incin/examples/resnet_demo.rs` — use `DefaultBackend` instead of `DummyBackend`
- `crates/incin/examples/idx_demo.rs` — use `DefaultBackend` instead of `DummyBackend`
- `crates/incin-core/src/lib.rs` — feature-gate `test_utils` module properly
- `crates/incin-core/src/nn/stats.rs` — update `DummyBackend` path
- `crates/incin-core/src/tensor/auto_device.rs` — update doc example path
- `crates/incin-core/examples/onnx_export.rs` — require `test-utils` feature
- `crates/incin-core/examples/model_inspect.rs` — require `test-utils` feature
- `crates/incin-core/Cargo.toml` — add self dev-dep with `test-utils`, add `required-features` for examples
- `crates/incin/Cargo.toml` — add self dev-dep with required features
- `crates/incin-core/tests/*.rs` — update `DummyBackend` import paths
