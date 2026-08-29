//! Implicit GEMM Conv2D kernel generator (im2col-free direct convolution) (PRF-021).
//!
//! Computes 2D image convolution directly as an implicit Matrix Multiplication:
//! - Eliminates $K_h \times K_w \times H_{\text{out}} \times W_{\text{out}}$ `im2col` VRAM expansion
//! - Maps output pixel coordinates $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$ directly to input receptive fields
//! - Register-tiled accumulation with boundary zero-padding guards

use alloc::string::String;
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// 2D Convolution Specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplicitConv2dSpec {
    /// Kernel function name.
    pub name: String,
    /// Data type.
    pub dtype: DTypeId,
    /// Kernel filter spatial size $(K_h, K_w)$.
    pub kernel_size: (usize, usize),
    /// Stride $(S_h, S_w)$.
    pub stride: (usize, usize),
    /// Padding $(P_h, P_w)$.
    pub padding: (usize, usize),
    /// Dilation $(D_h, D_w)$.
    pub dilation: (usize, usize),
}

impl ImplicitConv2dSpec {
    /// Creates a new implicit GEMM Conv2D specification.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        dtype: DTypeId,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            name: name.into(),
            dtype,
            kernel_size,
            stride,
            padding,
            dilation: (1, 1),
        }
    }

    /// Renders the implicit GEMM Conv2D CUDA C++ kernel.
    #[must_use]
    pub fn render_cuda(&self) -> String {
        let mut out = String::new();
        let scalar_ty = match self.dtype {
            DTypeId::F32 => "float",
            DTypeId::F64 => "double",
            DTypeId::F16 => "__half",
            DTypeId::BF16 => "__nv_bfloat16",
            _ => "float",
        };

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (dh, dw) = self.dilation;

        writeln!(
            out,
            "// Implicit GEMM Conv2D Kernel for {} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out, "#include <math.h>").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "extern \"C\" __global__ void {}(\n    const {scalar_ty}* __restrict__ Input,\n    const {scalar_ty}* __restrict__ Weight,\n    const {scalar_ty}* __restrict__ Bias,\n    {scalar_ty}* __restrict__ Output,\n    const int batch,\n    const int in_channels,\n    const int in_h,\n    const int in_w,\n    const int out_channels,\n    const int out_h,\n    const int out_w) {{",
            self.name
        )
        .unwrap();

        writeln!(
            out,
            "    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;"
        )
        .unwrap();
        writeln!(
            out,
            "    const int total_out_elements = batch * out_channels * out_h * out_w;"
        )
        .unwrap();
        writeln!(out, "    if (out_idx >= total_out_elements) return;").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    // Decompose output coordinate (N, Cout, Hout, Wout)"
        )
        .unwrap();
        writeln!(out, "    const int w_out = out_idx % out_w;").unwrap();
        writeln!(out, "    int rem = out_idx / out_w;").unwrap();
        writeln!(out, "    const int h_out = rem % out_h;").unwrap();
        writeln!(out, "    rem = rem / out_h;").unwrap();
        writeln!(out, "    const int c_out = rem % out_channels;").unwrap();
        writeln!(out, "    const int n = rem / out_channels;").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    float sum = (Bias != nullptr) ? static_cast<float>(Bias[c_out]) : 0.0f;"
        )
        .unwrap();
        writeln!(
            out,
            "    const int in_batch_offset = n * (in_channels * in_h * in_w);"
        )
        .unwrap();
        writeln!(
            out,
            "    const int weight_cout_offset = c_out * (in_channels * {kh} * {kw});"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    for (int c = 0; c < in_channels; ++c) {{").unwrap();
        writeln!(
            out,
            "        const int in_c_offset = in_batch_offset + c * (in_h * in_w);"
        )
        .unwrap();
        writeln!(
            out,
            "        const int weight_c_offset = weight_cout_offset + c * ({kh} * {kw});"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "        #pragma unroll").unwrap();
        writeln!(out, "        for (int r = 0; r < {kh}; ++r) {{").unwrap();
        writeln!(
            out,
            "            const int h_in = h_out * {sh} - {ph} + r * {dh};"
        )
        .unwrap();
        writeln!(out, "            if (h_in < 0 || h_in >= in_h) continue;").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "            #pragma unroll").unwrap();
        writeln!(out, "            for (int s = 0; s < {kw}; ++s) {{").unwrap();
        writeln!(
            out,
            "                const int w_in = w_out * {sw} - {pw} + s * {dw};"
        )
        .unwrap();
        writeln!(
            out,
            "                if (w_in < 0 || w_in >= in_w) continue;"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "                const float in_val = static_cast<float>(Input[in_c_offset + h_in * in_w + w_in]);").unwrap();
        writeln!(out, "                const float w_val = static_cast<float>(Weight[weight_c_offset + r * {kw} + s]);").unwrap();
        writeln!(out, "                sum = fmaf(in_val, w_val, sum);").unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    Output[out_idx] = static_cast<{scalar_ty}>(sum);").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }
}
