# Changelog

All notable changes to the Kindle framework will be documented in this file.

## [Unreleased] - Current Sprint
### Added
- **Macros:** Real compiled doctests for `idx![]`, `s![]`, and `#[module]` in `kindle-macros`.
- **Docs:** Comprehensive doc comments replacing auto-generated stubs across all core abstractions and backend traits.
- **Autograd Coverage:** Complete backward pass implementations (autograd wiring) for `WgpuBackend` (elementwise operations, activations, matmul, `layer_norm`, `batch_norm`, and `ReductionOps`) and `CudaBackend`.
- **Testing:** `DataLoader` unit tests in `kindle-data` (previously zero coverage).
- **Tooling:** New agent architecture workflows, `.agents` encapsulation rules, and rigorous `STATE.md`/`ROADMAP.md` sprint planning.

### Fixed
- **API Visibility:** Closed `pub(crate)` API leaks in `cuda`/`cpu`/`wgpu` backends and `kindle-core` (Issue B-3).
- **Feature Flags:** Fixed wildcard dependencies in `kindle-data` and missing test features across the workspace. Restored the `DefaultBackend` trait and removed dead `candle` feature configs.
- **CPU Backend:** Correctly gated `cpu::ops::elementwise` components under the `cpu` feature flag instead of `cuda` (Issue C-8).
- **Core Operations:** Solved numerous critical bugs from Phase 0 codebase audit, including shape inference errors and backward implementation panics (C-1, C-2, C-5, C-6, C-7).
- **Cargo:** Complete repo-wide `cargo fmt` pass and zero-warning `clippy --workspace` enforcement.

### Changed
- **Architecture:** Restructured backend dispatch and tape recording systems, moving legacy inline CUDA kernels to distinct `.cu` files for cleaner separation.

## [0.1.0-alpha.1] - Backend Refactoring Sprint
### Changed
- **Backend Crates:** Moved `native`, `wgpu`, and `cuda` backends into their own distinct crate (`kindle-backends`), standardizing trait bounds (`NumericOps`, `ModuleOps`, `ReductionOps`, etc.) across devices.
- **WGPU Migration:** Transitioned core components, backends, and app libraries from Metal to WGPU for unified cross-platform execution.
- **Legacy Removal:** Deleted obsolete, dead-code `ndarray` and `burn` compatibility wrappers.

### Added
- Complete WGPU convolution implementations and telemetry tracking features.

