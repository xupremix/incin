#85 Batched GEMM via cuBLASLt — M

> **Status (2026-09-22):** Landed as `4023ceed`. Plain `MatMulExact`
> requests try cuBLASLt first and fall back to NVRTC on any non-fit;
> epilogue requests (bias/relu/gelu) fail closed. Forced
> `CUBLAS_COMPUTE_32F`, row-major via `Cᵀ = Bᵀ Aᵀ`. Six host policy tests
> pass; six value tests are `#[ignore = "requires CUDA hardware"]` —
> compile-verified only, no runtime numerics claim. Coarse matmul row
> widened `F32_ONLY` → `FLOAT_DTYPES`. Dispatch lives in
> `cuda/ops/cublaslt.rs`; capability regenerated through `INCIN_DOCS`.
Finding: batched_matmul today = broadcast + per-slice narrow/matmul loop + concat (B*H launches, materialized copies); NVRTC tiled f32 kernel BM=128; cuda-vendor feature has ZERO call sites; cudarc 0.19.8 ships CudaBlasLT/MatmulConfig unused; autotune infra exists with no GEMM call site.
Recommendation: route matmul/bmm/addmm through cuBLASLt behind cuda-vendor (strided-batch, bias epilogue, heuristic algo first, autotune follow-up); NVRTC fallback; refuse non-contiguous batch strides pre-launch; flip composed->native rows.
Risk: 32MiB workspace/handle; driver/toolkit heuristic variance; FP4 needs Blackwell descriptors.
Unblocks: #95, #90 fast path.
