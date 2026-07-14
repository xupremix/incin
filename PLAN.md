# Phase 1: Full CUDA Implementation using cudarc

## Phase Goals
A high-performance CUDA backend exists in `kindle-native` that supports dynamic kernel compilation (NVRTC) and features state-of-the-art fused kernels (MatMul+Activation, MHA, AdamW). The `wgpu` path is retained as a fallback.

## Architecture & Implementation Choices
1. **Target Architecture**: We will use the `cudarc` crate to safely interface with the CUDA Driver API and NVRTC for dynamic PTX compilation.
2. **Buffer Management**: `CudaStorage` will hold `cudarc::driver::CudaSlice<T>`.
3. **Execution**: We will construct a `CudaBackend` that implements `kindle_core::tensor::backend::Backend`.
4. **Kernels**: We will prioritize fused kernels for performance, writing CUDA C++ code as strings compiled at runtime by `nvrtc`.

## Tasks
- [ ] Add `cudarc` dependency to `kindle-native` under a `cuda` feature flag.
- [ ] Implement `CudaStorage` wrapping `CudaSlice`.
- [ ] Implement `CudaBackend` core struct (Device selection, stream management).
- [ ] Write NVRTC dynamic compilation wrapper.
- [ ] Implement fused kernel for MatMul + GeLU.
- [ ] Implement fused kernel for MHA (FlashAttention style).
- [ ] Implement fused kernel for AdamW optimizer step.
- [ ] Integrate `CudaBackend` with all trait bounds in `kindle-core/src/tensor/backend.rs`.
- [ ] Run test suites comparing `CudaBackend` output to `CpuBackend`.

## Verification
- Unit test all fused kernels against their un-fused equivalents.
- Ensure the `cuda` feature flag isolates dependencies cleanly.
- Verify `cargo test --features cuda` passes perfectly.
