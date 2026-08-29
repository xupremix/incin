//! Rotary Position Embedding (RoPE) fused forward and backward kernel generator (PRF-014).
//!
//! Generates fast in-place and out-of-place rotary embedding kernels for CUDA C++, WebGPU (WGSL), and Metal (MSL):
//! - Direct coordinate pair rotation $[x_{2i}, x_{2i+1}] \to [x_{2i} \cos \theta - x_{2i+1} \sin \theta, x_{2i} \sin \theta + x_{2i+1} \cos \theta]$
//! - High-speed trigonometric caching using precomputed frequency tables or runtime fast $\sin / \cos$
//! - Analytical backward gradient computation for input $dx$ backpropagation

use alloc::string::String;
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// Rotary Position Embedding Specification.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeSpec {
    /// Kernel function name.
    pub name: String,
    /// Data type.
    pub dtype: DTypeId,
    /// Rotary dimension (must be even, e.g. 64, 128).
    pub rotary_dim: usize,
    /// Base frequency constant $\theta$ (typically 10000.0 or 500000.0 for LLaMA-style models).
    pub theta_base: f64,
    /// Whether cosine/sine table is precomputed and passed as a tensor.
    pub use_precomputed_freqs: bool,
}

impl RopeSpec {
    /// Creates a new RoPE kernel specification.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        dtype: DTypeId,
        rotary_dim: usize,
        theta_base: f64,
        use_precomputed_freqs: bool,
    ) -> Self {
        assert!(
            rotary_dim.is_multiple_of(2),
            "rotary dimension must be even"
        );
        Self {
            name: name.into(),
            dtype,
            rotary_dim,
            theta_base,
            use_precomputed_freqs,
        }
    }

    /// Renders the forward RoPE CUDA C++ kernel.
    #[must_use]
    pub fn render_cuda_forward(&self) -> String {
        let mut out = String::new();
        let scalar_ty = match self.dtype {
            DTypeId::F32 => "float",
            DTypeId::F64 => "double",
            DTypeId::F16 => "__half",
            DTypeId::BF16 => "__nv_bfloat16",
            _ => "float",
        };

        writeln!(
            out,
            "// Fused Rotary Position Embedding (RoPE) Forward Kernel for {} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out, "#include <math.h>").unwrap();
        writeln!(out).unwrap();

        let freqs_params = if self.use_precomputed_freqs {
            "const float* __restrict__ cos_table,\n    const float* __restrict__ sin_table,"
        } else {
            "const float theta_base,"
        };

        writeln!(
            out,
            "extern \"C\" __global__ void {}_forward(\n    const {scalar_ty}* __restrict__ X,\n    {scalar_ty}* __restrict__ Y,\n    {freqs_params}\n    const int batch_heads,\n    const int seq_len,\n    const int head_dim) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    const int bh_idx = blockIdx.z;").unwrap();
        writeln!(out, "    const int seq_idx = blockIdx.y;").unwrap();
        writeln!(
            out,
            "    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;"
        )
        .unwrap();
        writeln!(out, "    const int half_rotary = {};", self.rotary_dim / 2).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    if (bh_idx >= batch_heads || seq_idx >= seq_len || pair_idx >= half_rotary) return;").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    const int offset = (bh_idx * seq_len + seq_idx) * head_dim;"
        )
        .unwrap();
        writeln!(out, "    const int idx0 = offset + pair_idx * 2;").unwrap();
        writeln!(out, "    const int idx1 = offset + pair_idx * 2 + 1;").unwrap();
        writeln!(out).unwrap();

        if self.use_precomputed_freqs {
            writeln!(
                out,
                "    const int freq_offset = seq_idx * half_rotary + pair_idx;"
            )
            .unwrap();
            writeln!(out, "    const float cos_v = cos_table[freq_offset];").unwrap();
            writeln!(out, "    const float sin_v = sin_table[freq_offset];").unwrap();
        } else {
            writeln!(out, "    const float exponent = static_cast<float>(pair_idx * 2) / static_cast<float>({});", self.rotary_dim).unwrap();
            writeln!(
                out,
                "    const float freq = 1.0f / powf(theta_base, exponent);"
            )
            .unwrap();
            writeln!(
                out,
                "    const float angle = static_cast<float>(seq_idx) * freq;"
            )
            .unwrap();
            writeln!(out, "    float sin_v, cos_v;").unwrap();
            writeln!(out, "    sincosf(angle, &sin_v, &cos_v);").unwrap();
        }

        writeln!(out).unwrap();
        writeln!(out, "    const float x0 = static_cast<float>(X[idx0]);").unwrap();
        writeln!(out, "    const float x1 = static_cast<float>(X[idx1]);").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    // 2D Givens rotation").unwrap();
        writeln!(out, "    const float y0 = x0 * cos_v - x1 * sin_v;").unwrap();
        writeln!(out, "    const float y1 = x0 * sin_v + x1 * cos_v;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    Y[idx0] = static_cast<{scalar_ty}>(y0);").unwrap();
        writeln!(out, "    Y[idx1] = static_cast<{scalar_ty}>(y1);").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }

    /// Renders the backward RoPE CUDA C++ kernel.
    #[must_use]
    pub fn render_cuda_backward(&self) -> String {
        let mut out = String::new();
        let scalar_ty = match self.dtype {
            DTypeId::F32 => "float",
            DTypeId::F64 => "double",
            DTypeId::F16 => "__half",
            DTypeId::BF16 => "__nv_bfloat16",
            _ => "float",
        };

        writeln!(
            out,
            "// Fused Rotary Position Embedding (RoPE) Backward Kernel for {} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out, "#include <math.h>").unwrap();
        writeln!(out).unwrap();

        let freqs_params = if self.use_precomputed_freqs {
            "const float* __restrict__ cos_table,\n    const float* __restrict__ sin_table,"
        } else {
            "const float theta_base,"
        };

        writeln!(
            out,
            "extern \"C\" __global__ void {}_backward(\n    const {scalar_ty}* __restrict__ dY,\n    {scalar_ty}* __restrict__ dX,\n    {freqs_params}\n    const int batch_heads,\n    const int seq_len,\n    const int head_dim) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    const int bh_idx = blockIdx.z;").unwrap();
        writeln!(out, "    const int seq_idx = blockIdx.y;").unwrap();
        writeln!(
            out,
            "    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;"
        )
        .unwrap();
        writeln!(out, "    const int half_rotary = {};", self.rotary_dim / 2).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    if (bh_idx >= batch_heads || seq_idx >= seq_len || pair_idx >= half_rotary) return;").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    const int offset = (bh_idx * seq_len + seq_idx) * head_dim;"
        )
        .unwrap();
        writeln!(out, "    const int idx0 = offset + pair_idx * 2;").unwrap();
        writeln!(out, "    const int idx1 = offset + pair_idx * 2 + 1;").unwrap();
        writeln!(out).unwrap();

        if self.use_precomputed_freqs {
            writeln!(
                out,
                "    const int freq_offset = seq_idx * half_rotary + pair_idx;"
            )
            .unwrap();
            writeln!(out, "    const float cos_v = cos_table[freq_offset];").unwrap();
            writeln!(out, "    const float sin_v = sin_table[freq_offset];").unwrap();
        } else {
            writeln!(out, "    const float exponent = static_cast<float>(pair_idx * 2) / static_cast<float>({});", self.rotary_dim).unwrap();
            writeln!(
                out,
                "    const float freq = 1.0f / powf(theta_base, exponent);"
            )
            .unwrap();
            writeln!(
                out,
                "    const float angle = static_cast<float>(seq_idx) * freq;"
            )
            .unwrap();
            writeln!(out, "    float sin_v, cos_v;").unwrap();
            writeln!(out, "    sincosf(angle, &sin_v, &cos_v);").unwrap();
        }

        writeln!(out).unwrap();
        writeln!(out, "    const float dy0 = static_cast<float>(dY[idx0]);").unwrap();
        writeln!(out, "    const float dy1 = static_cast<float>(dY[idx1]);").unwrap();
        writeln!(out).unwrap();
        writeln!(
            out,
            "    // Inverse rotation for adjoint: [cos, sin; -sin, cos]"
        )
        .unwrap();
        writeln!(out, "    const float dx0 = dy0 * cos_v + dy1 * sin_v;").unwrap();
        writeln!(out, "    const float dx1 = -dy0 * sin_v + dy1 * cos_v;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    dX[idx0] = static_cast<{scalar_ty}>(dx0);").unwrap();
        writeln!(out, "    dX[idx1] = static_cast<{scalar_ty}>(dx1);").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }
}
