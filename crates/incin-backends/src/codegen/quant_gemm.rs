//! Quantized Matrix-Vector (GEMV) and Matrix-Matrix (GEMM) hardware kernel generator (PRF-017).
//!
//! High-throughput quantized execution generator for LLM decoding and inference:
//! - Q8_0 (8-bit integer weights with FP16 scale block per 32 elements)
//! - W8A16 (8-bit integer weights with FP16/BF16 activations)
//! - Register-level on-the-fly dequantization with zero intermediate buffer allocation
//! - Warp-shuffle reduction accumulation across quantized blocks

use alloc::string::String;
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// Quantized GEMV / GEMM kernel specification.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantGemmSpec {
    /// Kernel function name.
    pub name: String,
    /// Activation data type (FP32, FP16, BF16).
    pub act_dtype: DTypeId,
    /// Quantized weight format.
    pub weight_dtype: DTypeId,
    /// Quantization block size (e.g. 32 for Q8_0).
    pub block_size: usize,
}

impl QuantGemmSpec {
    /// Creates a new Q8_0 quantized GEMV specification.
    #[must_use]
    pub fn q8_0_gemv(name: impl Into<String>, act_dtype: DTypeId) -> Self {
        Self {
            name: name.into(),
            act_dtype,
            weight_dtype: DTypeId::Q8_0,
            block_size: 32,
        }
    }

    /// Renders high-throughput quantized matrix-vector multiplication (GEMV) in CUDA C++.
    #[must_use]
    pub fn render_cuda_gemv(&self) -> String {
        let mut out = String::new();
        let act_scalar = match self.act_dtype {
            DTypeId::F32 => "float",
            DTypeId::F16 => "__half",
            DTypeId::BF16 => "__nv_bfloat16",
            _ => "float",
        };

        writeln!(
            out,
            "// High-Performance Quantized GEMV (Q8_0) Kernel for {} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "// 32-element Q8_0 block structure (34 bytes: 1 f16 scale + 32 i8 values)"
        )
        .unwrap();
        writeln!(
            out,
            "struct __align__(2) BlockQ8_0 {{\n    __half scale;\n    signed char qs[32];\n}};"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "extern \"C\" __global__ void {}(\n    const BlockQ8_0* __restrict__ W,\n    const {act_scalar}* __restrict__ X,\n    {act_scalar}* __restrict__ Y,\n    const int M,\n    const int K) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    const int row = blockIdx.x;").unwrap();
        writeln!(out, "    const int tid = threadIdx.x;").unwrap();
        writeln!(out, "    if (row >= M) return;").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    const int num_blocks_per_row = K / 32;").unwrap();
        writeln!(
            out,
            "    const BlockQ8_0* w_row = W + row * num_blocks_per_row;"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    float sum = 0.0f;").unwrap();
        writeln!(
            out,
            "    // Each thread processes a stride of 32-element blocks"
        )
        .unwrap();
        writeln!(
            out,
            "    for (int b = tid; b < num_blocks_per_row; b += blockDim.x) {{"
        )
        .unwrap();
        writeln!(out, "        const BlockQ8_0 block = w_row[b];").unwrap();
        writeln!(
            out,
            "        const float scale_val = __half2float(block.scale);"
        )
        .unwrap();
        writeln!(out, "        const {act_scalar}* x_block = X + b * 32;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "        float block_dot = 0.0f;").unwrap();
        writeln!(out, "        #pragma unroll").unwrap();
        writeln!(out, "        for (int i = 0; i < 32; ++i) {{").unwrap();
        writeln!(
            out,
            "            const float w_dequant = static_cast<float>(block.qs[i]) * scale_val;"
        )
        .unwrap();
        writeln!(
            out,
            "            const float x_val = static_cast<float>(x_block[i]);"
        )
        .unwrap();
        writeln!(out, "            block_dot += w_dequant * x_val;").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "        sum += block_dot;").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // Warp-level tree reduction for row result").unwrap();
        writeln!(out, "    for (int offset = 16; offset > 0; offset /= 2) {{").unwrap();
        writeln!(
            out,
            "        sum += __shfl_down_sync(0xffffffff, sum, offset);"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    if (tid == 0) {{").unwrap();
        writeln!(out, "        Y[row] = static_cast<{act_scalar}>(sum);").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }
}
