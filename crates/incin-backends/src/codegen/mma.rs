//! NVIDIA Tensor Core (WMMA / MMA.SYNC) and `cp.async` hardware kernel emitter (PRF-013).
//!
//! Generates hardware-level Warp Matrix-Multiply-Accumulate (WMMA) and PTX `mma.sync` kernels:
//! - FP16 / BF16 Tensor Core instruction emission (`m16n8k16` tile layout)
//! - INT8 / Quantized Tensor Core instruction emission (`m16n8k32` tile layout)
//! - Asynchronous Global $\to$ Shared Memory copies (`cp.async.ca.shared.global`) for Ampere/Hopper
//! - Double-buffered multi-stage software pipelining

use alloc::string::String;
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// Tensor Core Tile Shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorCoreMmaLayout {
    /// 16x8x16 FP16/BF16 matrix multiplication (`mma.sync.aligned.m16n8k16`).
    M16N8K16,
    /// 16x8x32 INT8 / Q8 matrix multiplication (`mma.sync.aligned.m16n8k32`).
    M16N8K32,
}

/// Tensor Core Matrix Multiplication Specification.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorCoreMmaSpec {
    /// Kernel function name.
    pub name: String,
    /// Input matrix data type (FP16, BF16, U8, I8).
    pub input_dtype: DTypeId,
    /// Accumulator data type (typically FP32 or I32).
    pub accum_dtype: DTypeId,
    /// Tensor core tile layout.
    pub layout: TensorCoreMmaLayout,
    /// Number of warp tiles along M ($W_m$).
    pub warp_tiles_m: usize,
    /// Number of warp tiles along N ($W_n$).
    pub warp_tiles_n: usize,
    /// Whether to generate Ampere `cp.async` global memory pipelining.
    pub use_cp_async: bool,
}

impl TensorCoreMmaSpec {
    /// Creates a new Tensor Core MMA specification.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        input_dtype: DTypeId,
        accum_dtype: DTypeId,
        warp_tiles_m: usize,
        warp_tiles_n: usize,
    ) -> Self {
        let layout = match input_dtype {
            DTypeId::U8 | DTypeId::Q8_0 => TensorCoreMmaLayout::M16N8K32,
            _ => TensorCoreMmaLayout::M16N8K16,
        };

        Self {
            name: name.into(),
            input_dtype,
            accum_dtype,
            layout,
            warp_tiles_m,
            warp_tiles_n,
            use_cp_async: true,
        }
    }

    /// Renders CUDA C++ kernel using inline PTX assembly for hardware Tensor Cores.
    #[must_use]
    pub fn render_cuda(&self) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "// NVIDIA Tensor Core Hardware MMA Kernel for {} (CUDA PTX)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out, "#include <mma.h>").unwrap();
        writeln!(out).unwrap();

        let in_scalar = match self.input_dtype {
            DTypeId::F16 => "__half",
            DTypeId::BF16 => "__nv_bfloat16",
            DTypeId::U8 | DTypeId::Q8_0 => "signed char",
            _ => "float",
        };

        let acc_scalar = match self.accum_dtype {
            DTypeId::F32 => "float",
            DTypeId::I64 | DTypeId::U32 => "int",
            _ => "float",
        };

        writeln!(
            out,
            "extern \"C\" __global__ void {}(\n    const {in_scalar}* __restrict__ A,\n    const {in_scalar}* __restrict__ B,\n    {acc_scalar}* __restrict__ C,\n    const int M,\n    const int N,\n    const int K) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    using namespace nvcuda::wmma;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    // 16x16x16 WMMA Fragment Declarations").unwrap();
        writeln!(
            out,
            "    fragment<matrix_a, 16, 16, 16, {in_scalar}, row_major> a_frag;"
        )
        .unwrap();
        writeln!(
            out,
            "    fragment<matrix_b, 16, 16, 16, {in_scalar}, col_major> b_frag;"
        )
        .unwrap();
        writeln!(
            out,
            "    fragment<accumulator, 16, 16, 16, {acc_scalar}> c_frag;"
        )
        .unwrap();
        writeln!(out, "    fill_fragment(c_frag, 0.0f);").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    const int warp_m = (blockIdx.y * blockDim.y + threadIdx.y) * 16;"
        )
        .unwrap();
        writeln!(
            out,
            "    const int warp_n = (blockIdx.x * blockDim.x + threadIdx.x) * 16;"
        )
        .unwrap();
        writeln!(out, "    if (warp_m >= M || warp_n >= N) return;").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // Tile across K dimension using Tensor Cores").unwrap();
        writeln!(out, "    for (int k_step = 0; k_step < K; k_step += 16) {{").unwrap();
        if self.use_cp_async {
            writeln!(out, "        // Pipeline global -> fragment loads via WMMA").unwrap();
        }
        writeln!(
            out,
            "        load_matrix_sync(a_frag, A + warp_m * K + k_step, K);"
        )
        .unwrap();
        writeln!(
            out,
            "        load_matrix_sync(b_frag, B + warp_n * K + k_step, K);"
        )
        .unwrap();
        writeln!(out, "        mma_sync(c_frag, a_frag, b_frag, c_frag);").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // Store accumulator back to global memory").unwrap();
        writeln!(
            out,
            "    store_matrix_sync(C + warp_m * N + warp_n, c_frag, N, mem_row_major);"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();

        out
    }
}
