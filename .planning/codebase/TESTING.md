# Testing Patterns

**Analysis Date:** 2026-07-09

## Test Framework

**Runner:**
- Standard `cargo test` (built-in Rust test harness) across the whole workspace. No custom test runner (e.g. `nextest`) configured in CI.
- `trybuild` (dev-dependency of `kindle-core`) is used for compile-fail testing — see `crates/kindle-core/tests/compile_tests.rs`.

**Assertion Library:**
- Standard library assertions only: `assert_eq!`, `assert!`, `assert!(x.is_nan())`, no third-party assertion crate (no `pretty_assertions`, no `assert_matches`).

**Run Commands:**
```bash
cargo test --workspace --all-targets   # Run all unit + integration tests (as run in CI)
cargo fmt --all -- --check             # Formatting check (CI gate, run before tests)
cargo clippy --workspace --all-targets -- -D warnings   # Lint gate (CI gate, run before tests)
cargo build --examples --workspace     # Build examples (final CI step)
```
Defined in `.github/workflows/ci.yml` (job `Test Suite`, single Ubuntu runner, `dtolnay/rust-toolchain@stable`). There is no separate coverage-collection step or coverage tool configured.

## Test File Organization

**Location:**
- Integration tests live in each crate's `tests/` directory, one file per feature area (not co-located with `src/`):
  - `crates/kindle/tests/`: `broadcast.rs`, `layers.rs`, `optim_tests.rs`, `nn_tests.rs`, `serde_tests.rs`, `tensor_ops.rs`, `autograd_tests.rs`, `onnx_import.rs`, `macro_tests.rs`, `data_tests.rs`
  - `crates/kindle-core/tests/`: `builder_permutations.rs`, `reshape.rs`, `compile_tests.rs`, `concat_stack.rs`, plus `compile_fail/*.rs` fixtures
  - `crates/kindle-backends/tests/`: `ndarray.rs`, `ops.rs`
- Unit tests are also embedded in `src/` files as `#[cfg(test)] mod tests { ... }` blocks for focused, low-level logic (e.g. `crates/kindle-core/src/err.rs`, `crates/kindle-core/src/tensor/device.rs`, `crates/kindle-core/src/shapes/reshape.rs`, `crates/kindle-core/src/shapes/shape.rs`, `crates/kindle-core/src/tensor/ops/loss.rs`, `crates/kindle-core/src/shapes/idx.rs`, `crates/kindle-core/src/tensor/base.rs`, `crates/kindle-backends/src/lib.rs`).
- Total `#[test]` function count across `crates/*/src` and `crates/*/tests`: 79.

**Naming:**
- Integration test functions: `test_<area>_<scenario>`, e.g. `test_reshape_static_success`, `test_try_reshape_dynamic`, `test_unary_abs`, `test_unary_relu`, `test_unary_gelu`, `test_unary_softmax` (`crates/kindle/tests/tensor_ops.rs`, `crates/kindle-core/tests/reshape.rs`).
- Compile-fail fixture files are named after the exact invalid scenario they assert (`device_mismatch.rs`, `conv2d_invalid_shape.rs`, `macro_idx_invalid.rs`) rather than `test_*`.

**Structure:**
```
crates/
├── kindle/tests/            # High-level, user-facing API integration tests (uses kindle::prelude)
├── kindle-core/tests/       # Core tensor/shape/macro integration tests (uses kindle_core::prelude)
│   └── compile_fail/        # trybuild negative-compilation fixtures
└── kindle-backends/tests/   # Backend-specific (ndarray) operation tests
```

## Test Structure

**Suite Organization:**
```rust
// crates/kindle-core/tests/reshape.rs
use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_core::tensor::device::Cpu;
use kindle_macros::s;
use typenum::{U2, U3, U6};

#[test]
fn test_reshape_static_success() {
    let t = Tensor::<s![U2, U3], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
    let reshaped = t.reshape::<s![U6]>(((),)).unwrap();
    assert_eq!(reshaped.dims(), [6]);
}
```
Larger integration test files (e.g. `crates/kindle/tests/tensor_ops.rs`) are organized into `// ---- N.N Section Name ----` banner comments grouping related tests (Unary Operations, etc.), with each test annotated by an inline comment describing the permutation matrix it exercises (e.g. `// permutations: positive, negative, zero, very small numbers, very large numbers, NaN, Inf`).

**Patterns:**
- No shared setup/teardown fixtures or test harness struct — each test constructs its own `Tensor` inline via `from_slice`/`zeros`.
- Tests frequently return `Result<()>` and use `?` for fallible tensor construction/ops rather than `.unwrap()`, keeping error paths visible while still surfacing panics via `#[test]` failure on `Err`.
- A local helper function is sometimes defined at the top of a test file to reduce repetition, e.g. `to_vec(t: &Tensor<Dyn, CpuBackend>) -> Vec<f32>` in `crates/kindle/tests/tensor_ops.rs`, which extracts tensor data through `.inner().flatten_all().unwrap().to_vec1::<f32>().unwrap()`.
- Numeric assertions on floating point results use epsilon-tolerant comparisons (`assert!((r[1] - expected).abs() < 1e-4)`) rather than `assert_eq!`, given backend floating point nondeterminism.

## Mocking

**Framework:** No mocking framework (no `mockall`, no `mockito`). The codebase instead uses a dedicated `DummyBackend<Elem, Device>` type (`kindle_core::tensor::backend::dummy::DummyBackend`) as a lightweight stand-in tensor backend for shape/type-level tests that don't need real numeric computation — see `crates/kindle-core/tests/reshape.rs` and `builder_permutations.rs`.

**Patterns:**
```rust
// Use DummyBackend when only shape/type behavior is under test, not numeric correctness
let t = Tensor::<s![U2, U3], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
```
For real numeric behavior, tests use the crate's actual `DefaultBackend` type alias (`crates/kindle/tests/tensor_ops.rs`: `type CpuBackend = DefaultBackend;`) backed by the real `ndarray`/`candle` backend implementation rather than a mock.

**What to Mock:**
- Use `DummyBackend` only for pure shape/type-system verification (reshape validity, builder permutations, static-shape macro expansion) where actual numeric computation is irrelevant.

**What NOT to Mock:**
- Numeric operation correctness (unary ops, softmax, matmul, loss functions, etc.) is always tested against the real backend (`DefaultBackend` / `CpuBackend`), never a dummy/mocked backend — see `crates/kindle/tests/tensor_ops.rs`.

## Fixtures and Factories

**Test Data:**
- No shared fixture files or factory crate; tensors are built inline per test via `Tensor::<Shape, Backend>::from_slice(&[...], ())` or `::zeros(...)`.
- Real ONNX model fixtures exist at the repo root and `test_models/` for import/export integration tests: `crates/kindle/resnet18.onnx` (+ `.kindle_meta`), `test_models/advanced.onnx`, `test_models/if.onnx`, and a serialized weights fixture `rnn_model.safetensors`, consumed by tests like `crates/kindle/tests/onnx_import.rs` and `crates/kindle/tests/serde_tests.rs`.

**Location:**
- Inline in test functions; binary/model fixtures live at repo root / `test_models/` rather than under a `fixtures/` or `testdata/` directory.

## Coverage

**Requirements:** No coverage tool or enforced threshold configured (no `cargo-tarpaulin`, `cargo-llvm-cov`, or `grcov` in CI).

**View Coverage:**
```bash
# Not configured — no coverage command available in this repo.
```

## Test Types

**Unit Tests:**
- Embedded `#[cfg(test)] mod tests` blocks inside `src/` files for isolated logic close to implementation (error formatting, shape arithmetic, device equality) — see `crates/kindle-core/src/err.rs`.

**Integration Tests:**
- Majority of test coverage: black-box tests against each crate's public `prelude` API in `tests/*.rs`, covering tensor ops, NN layers, optimizers, autograd, serialization (safetensors), ONNX import, data loading, and proc-macro-generated code.

**E2E Tests:**
- Not used in the traditional web-app sense. The closest equivalent is ONNX model round-trip import/export tests (`crates/kindle/tests/onnx_import.rs`) that load real `.onnx` files end-to-end through the framework.

**Compile-Fail Tests:**
- `trybuild`-based negative compilation tests assert that invalid usage (shape mismatches, device mismatches, invalid macro arguments) fails to compile with the expected diagnostics. Driven by a single `#[test] fn compile_fail()` in `crates/kindle-core/tests/compile_tests.rs` which globs `tests/compile_fail/*.rs`. This is the primary mechanism for testing the type-level shape-safety guarantees of the framework — new static-shape-safety invariants should get a corresponding fixture here.

## Common Patterns

**Async Testing:**
- Not applicable — the codebase is synchronous (no `tokio`/`async-std` test attributes observed).

**Error Testing:**
```rust
// crates/kindle-core/src/err.rs
#[test]
fn test_error_formatting() {
    let err = Error::OutOfMemory { device: "CUDA:0".to_string() };
    let formatted = alloc::format!("{}", err);
    assert_eq!(formatted, "Out of Memory error on device: CUDA:0");
}
```
Error variant `Display` output is asserted directly via string equality rather than pattern-matching the enum, ensuring user-facing error messages stay stable.

**Result-returning tests:**
```rust
#[test]
fn test_unary_abs() -> Result<()> {
    let t = Tensor::<s![7], CpuBackend>::from_slice(&[1.0, -1.0, 0.0, 1e-30, -1e30, f32::NAN, f32::INFINITY], ())?;
    let r = to_vec(&t.abs()?.into_dyn());
    assert_eq!(r[0], 1.0);
    Ok(())
}
```
Preferred over `.unwrap()`-heavy tests when the crate's `Result` alias is in scope; construction/op failures surface as a normal test failure via the `?` operator and the `Termination` impl for `Result`.

---

*Testing analysis: 2026-07-09*
