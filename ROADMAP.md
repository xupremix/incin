# Kindle — Release Roadmap

> **Goal:** Ship stable `0.1.0` crates to crates.io with a public API surface
> that can evolve without breaking changes in any `0.x` patch release, and
> with a clearly documented semver contract for the `1.0` boundary.
>
> **Guiding principle:** Every `pub` item is a promise. Anything that is not
> deliberately part of the public API must be `pub(crate)` or `pub(super)`.
> Adding a function later is semver-compatible; removing or changing one is not.

---

## Current State

| Crate | Tests | Status |
|-------|-------|--------|
| `kindle-core` | 265 passing | ✅ Solid |
| `kindle-native` | 11+5+4+1+2 passing | ✅ Solid |
| `kindle-wgpu` | 53 passing, **1 failing** | ⚠ One bug |
| `kindle-backends` | (links to candle) | ⚠ Partial |
| `kindle-macros` | — | ⚠ Undocumented |
| `kindle-data` | — | ⚠ Untested |
| `kindle` (facade) | Bus error in linker | ❌ Broken |
| `kindle-viz` | — | 🔲 Prototype |
| `kindle-telemetry` | — | 🔲 Prototype |

---

## Blocker Issues (must fix before any release)

### B-1 — `kindle-wgpu`: one failing test (`test_adamw_step`)
GPU/CPU race in `dispatch.rs::run_pipeline` — no `device.poll` after submit.
**Fix:** Add `state.device.poll(wgpu::Maintain::Wait)` at the end of `run_pipeline`.

### B-2 — `kindle` facade crate: linker bus-error in test suite
`cargo test -p kindle` crashes the linker (`rust-lld` SIGBUS / Bus error).
Likely a circular dependency or an object file size issue from pulling in both
`candle` and the rest of the workspace. Must be reproducibly buildable and
testable before publishing.

### B-3 — Accidental `pub` API leakage — implementation details are public
Multiple crates expose internals that were never intended to be stable API:

| Crate | Leaking symbol | Problem |
|-------|---------------|---------|
| `kindle-wgpu` | `dispatch_binary`, `dispatch_unary`, `dispatch_scalar`, `dispatch_reduce_all`, `dispatch_softmax`, `dispatch_transpose`, `dispatch_im2col`, `dispatch_adamw` | Raw shader dispatch helpers — internal glue, not public API |
| `kindle-wgpu` | `get_or_create_pipeline`, `WgpuDeviceState`, `get_device_state` | Pipeline cache internals — must not be stable |
| `kindle-wgpu` | `WgpuBuffer`, `WgpuStorage.buffer`, `WgpuStorage.shape`, `WgpuStorage.strides` | Storage fields are pub — users can construct invalid states |
| `kindle-native` | `TapeEntry`, `tape::push` | Autograd tape internals |
| `kindle-native` | `scatter_into_zeros`, `contiguous_strides`, `is_contiguous`, `broadcast_shape` | Utility fns used only inside the crate |
| `kindle-native` | `NativeCudaDispatcher`, `NativeMetalDispatcher`, `cuda_cache` module | GPU dispatch internals |
| `kindle-native` | All of `pub mod creation`, `pub mod ops`, `pub mod stride`, `pub mod tape`, `pub mod var` | Entire implementation modules are `pub` — should be `pub(crate)` |

**Fix pattern for every case:**
- Change `pub mod X` → `pub(crate) mod X` for impl modules.
- Change `pub fn dispatch_*` / `pub fn get_or_create_pipeline` / `pub fn get_device_state` → `pub(crate)`.
- Change `pub struct WgpuBuffer { pub buffer, pub size }` → non-pub fields with constructor / accessor methods.
- Only re-export what a downstream crate genuinely needs: `WgpuBackend`, `NativeBackend`.

- [x] **B-4: Backend FloatElem generic consistency**
  - **Issue:** `CandleBackend<T, D>` and `NdarrayBackend<T, D>` ignore their own `T` generic and hardcode `type FloatElem = f32`. This is semantically wrong and will silently ignore any `CandleBackend<f64, _>` instantiation.
  - **Status:** Documented as a known limitation in 0.1.0 (to avoid breaking backwards compatibility with hardcoded internal types).
- [x] **B-5: DummyBackend shape calculation bugs**
  - **Issue:** `DummyBackend` conv/pool ops return `t.clone()` (unchanged shape) instead of computing the real output shape from kernel/stride/padding/dilation. Any user relying on `DummyBackend` for shape testing will get silent wrong answers.
  - **Status:** Fixed shape calculations in `conv1d`, `conv2d`, `conv_transpose2d`, `max_pool2d`, and `avg_pool2d`.
- [x] **B-6: Compile-fail tests exercise the wrong failure**
  - **Issue:** `kindle-core/tests/compile_fail/stack_static_mismatch.rs` and `concat_static_mismatch.rs` are missing `use kindle_macros::s;` so they fail to compile because of an unknown macro, not the intended shape-mismatch check.
  - **Status:** Fixed macro imports and updated `.stderr` snapshots.

---

## API Stability Contract (must define before release)

### What is stable in `0.1.0`

| Symbol | Crate | Stable? |
|--------|-------|---------|
| `Backend` trait + all sub-traits (`FloatOps`, `NumericOps`, ...) | `kindle-core` | ✅ Yes — this is the extension point |
| `Tensor<S, B, K, D, G>` type and its inherent methods | `kindle-core` | ✅ Yes |
| `s![]`, `idx![]` macros | `kindle-macros` | ✅ Yes |
| `NativeBackend<T, D>` | `kindle-native` | ✅ Yes (the struct only) |
| `WgpuBackend<T, D>` | `kindle-wgpu` | ✅ Yes (the struct only) |
| `Error` enum variants | `kindle-core` | ✅ Yes (non-exhaustive + `#[non_exhaustive]` needed) |
| `nn::Linear`, `Conv2d`, `LayerNorm`, etc. | `kindle-core` | ✅ Yes |
| `dispatch_*` functions | `kindle-wgpu` | ❌ No — `pub(crate)` only |
| `WgpuBuffer`, `WgpuStorage` fields | `kindle-wgpu` | ❌ No — fields must be private |
| `NativeStorage`, `NativeVar` fields | `kindle-native` | ❌ No — fields must be private |
| Internal modules (`tape`, `stride`, `creation`, `ops`, `var`) | `kindle-native` | ❌ No — `pub(crate)` |

### `#[non_exhaustive]` is required on
- `Error` enum (adding new variants must not break match arms downstream)
- `KindleDType` enum
- `KindleDevice` struct

### Semver implications
Adding a new associated type or method to the `Backend` trait is a **breaking change** (implementors must add it). Use a strategy:
1. All new methods on `Backend` must have a default impl returning `Err(Error::UnsupportedBackendOperation {...})`.
2. Document clearly which methods are required vs optional.
3. Keep a `BackendVersion` associated const so implementations can opt into capability discovery.

---

## Documentation Requirements

Every public item must have a `///` doc comment. Current state:

| Crate | Missing doc coverage |
|-------|---------------------|
| `kindle-core` | Backend trait has doc comments; individual methods mostly do not |
| `kindle-native` | No doc comments on any `pub` items |
| `kindle-wgpu` | No doc comments on any `pub` items |
| `kindle-macros` | `s![]` and `idx![]` have no `#[doc]` examples |

**Minimum bar for `0.1.0`:** Every `pub` item has at least a one-line `///` comment. Every crate has a `//!` module-level doc describing its purpose. Every non-trivial public type has a usage example in the doc.

---

## Testing Requirements

| Type | Current | Target for `0.1.0` |
|------|---------|---------------------|
| Unit tests (per-op) | 265+ in native, 53 in wgpu | All ops covered, 0 failures |
| Integration parity tests | None | `NativeBackend` vs ground-truth for every op |
| Cross-backend parity | None | `NativeBackend` ≈ `WgpuBackend` to 1e-4 for all common ops |
| Compile-fail shape tests | Partially broken (B-6) | All shape-mismatch tests exercise the correct error |
| Doc tests | None | At least one `///` example per public type |

---

## Crate-by-Crate Remaining Work

### `kindle-core` (the contract)
- [ ] Mark `Error` as `#[non_exhaustive]`
- [ ] Mark `KindleDType` as `#[non_exhaustive]`
- [ ] Add default impls for all non-critical `Backend` methods
- [ ] Fix compile-fail tests (B-6)
- [ ] Fix `DummyBackend` conv/pool shape math (B-5)
- [ ] Add `///` doc to all trait methods in `backend.rs`
- [ ] Write `#[doc = include_str!("../../README.md")]` on crate root

### `kindle-native`
- [ ] Fix `FloatElem` not driven by `T` (B-4, already fixed in native — confirm)
- [x] Change all implementation modules to `pub(crate)` (B-3)
- [x] Make `NativeStorage` / `NativeVar` fields private with constructors (B-3)
- [ ] Add `///` doc to `NativeBackend` struct
- [x] Run `cargo clippy -p kindle-native -- -D warnings`
- [ ] Fix 58 `#[ignore]`d tests — either implement them or delete them

### `kindle-wgpu`
- [x] Fix `test_adamw_step` (B-1) — add `device.poll` in `run_pipeline`
- [x] Change all `dispatch.rs` functions to `pub(crate)` (B-3)
- [x] Change `get_or_create_pipeline` / `get_device_state` to `pub(crate)` (B-3)
- [x] Make `WgpuBuffer`, `WgpuStorage`, `WgpuDeviceState` fields private (B-3)
- [x] Change `pub mod dispatch`, `pub mod pipeline`, `pub mod device`, `pub mod storage` to `pub(crate)` in `lib.rs`
- [ ] Implement `conv_transpose1d`, `conv_transpose2d` (3 `unimplemented!()` remaining)
- [ ] Add GPU matmul path to `conv2d` (remove CPU fallback)
- [x] Add numerical parity tests against `NativeBackend`
- [ ] Add `///` doc to `WgpuBackend`

### `kindle-backends` (legacy candle/ndarray wrappers)
- [ ] Fix `NdarrayBackend` — most ops return `UnsupportedBackendOperation`
- [ ] Fix `BurnBackend` — does not compile at all with `--features burn`
- [ ] Fix `CandleBackend` `FloatElem` hardcoding (B-4)
- [ ] Decide: publish as `0.1.0-alpha` with explicit "not all ops implemented" notice, or defer to `0.2.0`

### `kindle-macros`
- [ ] Add `#[doc]` examples to `s![]`, `idx![]`, `module`
- [ ] Remove dead proc-macro code paths (if any)
- [ ] Run `cargo clippy -p kindle-macros -- -D warnings`

### `kindle-data`
- [ ] Add integration tests
- [ ] Document public API

### `kindle` (facade)
- [x] Fix linker crash in test suite (B-2)
- [ ] Audit re-exports: only re-export what belongs in the public prelude
- [ ] Remove direct `anyhow` dependency (leaked through `dev-dependencies`)

### `kindle-viz` / `kindle-telemetry` / `kindle-viz-plugin-api`
- [ ] Not ready for `0.1.0` — either exclude from workspace publish or publish as `0.1.0-alpha.1`

---

## Repository Hygiene

- [ ] Delete scratch files from root: `diagnostic_test.rs`, `dummy_ops.rs`, `scratch.rs`, `scratch2.rs`, `combine.py`, `fix_*.py`, `impl_ops.py`, `replace_ops.py`, `rewrite_elementwise.py`, `remove_traits.py`
- [ ] Move planning docs to `.planning/` or `docs/`: `FUTURE_ROADMAP.md`, `GPU_ROADMAP.md`, `NATIVE_BACKEND_PLAN.md`, `NO_STD_MIGRATION.md`, `PLAN.md`, `DESIGN.md`, `TODO.md`
- [ ] Delete `rnn_model.safetensors` from root (large binary, should be in test fixtures or git-ignored)
- [ ] Add `publish = false` to `kindle-viz` / `kindle-telemetry` / `kindle-viz-plugin-api` until they are ready
- [ ] Add `[workspace.metadata.release]` or similar to control which crates get published
- [ ] Update `README.md`: it still says "wraps candle and burn" — that is no longer the primary story
- [ ] Add `CHANGELOG.md` following Keep-a-Changelog format
- [ ] Add `CONTRIBUTING.md`
- [ ] Ensure `LICENSE_MIT` and `LICENSE_APACHE` are correct and that `license = "MIT OR Apache-2.0"` matches
- [ ] Add `.cargo/config.toml` with `[net] offline = false` and a `[build]` target if needed for CI
- [ ] Add GitHub Actions CI: `cargo check --workspace`, `cargo test -p kindle-core -p kindle-native -p kindle-wgpu`, `cargo clippy --workspace -- -D warnings`

---

## Feature Flag Audit

Before publishing, every feature must be tested in isolation:

```bash
cargo check -p kindle-core --no-default-features
cargo check -p kindle-core --all-features
cargo check -p kindle-native --no-default-features
cargo check -p kindle-native --features cuda    # needs CUDA env, CI-gated
cargo check -p kindle-wgpu --no-default-features
cargo check -p kindle --no-default-features
cargo check -p kindle --features candle
cargo check -p kindle --features ndarray
```

Known gap: `kindle --features burn` does not compile.

---

## Publish Order (dependency graph)

```
kindle-macros   (no workspace deps)
    ↓
kindle-core     (dep: kindle-macros)
    ↓
kindle-native   (dep: kindle-core)
kindle-wgpu     (dep: kindle-core)
kindle-backends (dep: kindle-core)
    ↓
kindle          (dep: kindle-core, kindle-macros, kindle-backends, kindle-data)
kindle-data     (dep: none from workspace, could publish independently)
```

Publish `kindle-viz`, `kindle-telemetry`, `kindle-viz-plugin-api` after `kindle` is stable.

---

## Version Strategy

| Milestone | Version | Meaning |
|-----------|---------|---------|
| Internal only | `0.1.0-alpha.1` | Fixes B-1..B-6, API cleanup, no public announcement |
| Beta | `0.1.0-beta.1` | All tests pass, full docs, CI green |
| Release | `0.1.0` | crates.io publish, README updated, announcement |
| First breaking change | `0.2.0` | Semver minor — breaking only in `0.x` land |
| Stable API | `1.0.0` | `Backend` trait frozen, no breaking changes without major bump |

---

## Summary Checklist (ordered by priority)

1. **B-1** Fix `test_adamw_step` (GPU poll race)
2. **B-2** Fix `kindle` facade linker crash
3. **B-3** Scope all internal symbols to `pub(crate)` across all crates
4. **B-4** Fix `FloatElem` in legacy backends
5. **B-5** Fix `DummyBackend` conv/pool shape math
6. **B-6** Fix compile-fail test imports
7. Add `#[non_exhaustive]` to `Error`, `KindleDType`
8. Add `///` docs to all `pub` items
9. Add `Backend` method default impls for non-critical ops
10. Add parity integration tests
11. Clean up root-level scratch files and planning docs
12. Add `publish = false` to not-ready crates
13. Add CI workflow
14. Write `CHANGELOG.md`
