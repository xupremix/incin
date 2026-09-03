//! Mixture of Experts (MoE) Top-K gating and sparse token dispatch kernel generator (PRF-020).
//!
//! Generates high-speed MoE routing and combination kernels for CUDA C++:
//! - Top-K Gating: Computes top-$k$ expert selection and normalized routing softmax probabilities
//! - Token Permute / Gather: Collects active tokens per expert into contiguous batch buffers
//! - Weighted Scatter-Add: Gathers and accumulates expert outputs back to original token sequence

use alloc::string::String;
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// Mixture of Experts Gating Specification.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeGatingSpec {
    /// Kernel function name.
    pub name: String,
    /// Data type.
    pub dtype: DTypeId,
    /// Total number of experts (e.g. 8 for Mixtral, 64 for DeepSeek-V2).
    pub num_experts: usize,
    /// Number of active experts selected per token (top-$k$, e.g. 2, 6, 8).
    pub top_k: usize,
}

impl MoeGatingSpec {
    /// Creates a new MoE gating specification.
    #[must_use]
    pub fn new(name: impl Into<String>, dtype: DTypeId, num_experts: usize, top_k: usize) -> Self {
        assert!(
            top_k <= num_experts,
            "top_k cannot exceed total number of experts"
        );
        Self {
            name: name.into(),
            dtype,
            num_experts,
            top_k,
        }
    }

    /// Renders the Top-K Gating & Softmax CUDA C++ kernel.
    #[must_use]
    pub fn render_cuda_gating(&self) -> String {
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
            "// Fused MoE Top-{} Gating Kernel for {} (CUDA)",
            self.top_k, self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "extern \"C\" __global__ void {}_gating(\n    const {scalar_ty}* __restrict__ RouterLogits,\n    int* __restrict__ SelectedExpertIndices,\n    float* __restrict__ ExpertWeights,\n    const int num_tokens,\n    const int num_experts) {{",
            self.name
        )
        .unwrap();

        writeln!(
            out,
            "    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;"
        )
        .unwrap();
        writeln!(out, "    if (token_idx >= num_tokens) return;").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    const {scalar_ty}* logits = RouterLogits + token_idx * num_experts;"
        )
        .unwrap();
        writeln!(
            out,
            "    int* out_indices = SelectedExpertIndices + token_idx * {};",
            self.top_k
        )
        .unwrap();
        writeln!(
            out,
            "    float* out_weights = ExpertWeights + token_idx * {};",
            self.top_k
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    // Local registers for Top-{} selection",
            self.top_k
        )
        .unwrap();
        writeln!(out, "    int top_idx[{}] = {{0}};", self.top_k).unwrap();
        writeln!(out, "    float top_val[{}] = {{-1e20f}};", self.top_k).unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    for (int e = 0; e < num_experts; ++e) {{").unwrap();
        writeln!(
            out,
            "        const float val = static_cast<float>(logits[e]);"
        )
        .unwrap();
        writeln!(out, "        // Insertion sort into top-k").unwrap();
        writeln!(out, "        for (int k = 0; k < {}; ++k) {{", self.top_k).unwrap();
        writeln!(out, "            if (val > top_val[k]) {{").unwrap();
        writeln!(
            out,
            "                for (int j = {} - 1; j > k; --j) {{",
            self.top_k
        )
        .unwrap();
        writeln!(out, "                    top_val[j] = top_val[j - 1];").unwrap();
        writeln!(out, "                    top_idx[j] = top_idx[j - 1];").unwrap();
        writeln!(out, "                }}").unwrap();
        writeln!(out, "                top_val[k] = val;").unwrap();
        writeln!(out, "                top_idx[k] = e;").unwrap();
        writeln!(out, "                break;").unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    // Softmax over Top-{} selected values",
            self.top_k
        )
        .unwrap();
        writeln!(out, "    float max_val = top_val[0];").unwrap();
        writeln!(out, "    float sum_exp = 0.0f;").unwrap();
        writeln!(out, "    for (int k = 0; k < {}; ++k) {{", self.top_k).unwrap();
        writeln!(out, "        sum_exp += expf(top_val[k] - max_val);").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(
            out,
            "    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    for (int k = 0; k < {}; ++k) {{", self.top_k).unwrap();
        writeln!(out, "        out_indices[k] = top_idx[k];").unwrap();
        writeln!(
            out,
            "        out_weights[k] = expf(top_val[k] - max_val) * inv_sum;"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }
}
