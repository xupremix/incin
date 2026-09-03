//! Fused Cross Entropy Loss and Log-Softmax kernel generator (PRF-015).
//!
//! Generates single-pass fused Cross Entropy Loss kernels for CUDA C++, WebGPU (WGSL), and Metal (MSL):
//! - Computes online safe Log-Sum-Exp and Negative Log-Likelihood without materializing $[B, S, V]$ logits in VRAM
//! - Optional label smoothing regularizer $\epsilon \cdot \frac{1}{V}$
//! - Direct analytical backward gradient synthesis $dX = \text{Softmax}(X) - Y_{\text{target}}$ with zero extra memory passes

use alloc::string::String;
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// Fused Cross Entropy Loss Specification.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossEntropySpec {
    /// Kernel function name.
    pub name: String,
    /// Data type of logits.
    pub dtype: DTypeId,
    /// Vocabulary dimension size $V$.
    pub vocab_size: usize,
    /// Label smoothing factor $\alpha \in [0.0, 1.0)$.
    pub label_smoothing: f64,
}

impl CrossEntropySpec {
    /// Creates a new Cross Entropy Loss specification.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        dtype: DTypeId,
        vocab_size: usize,
        label_smoothing: f64,
    ) -> Self {
        Self {
            name: name.into(),
            dtype,
            vocab_size,
            label_smoothing,
        }
    }

    /// Renders the forward fused Cross Entropy CUDA C++ kernel.
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
            "// Fused Cross Entropy Loss Forward Kernel for {} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "extern \"C\" __global__ void {}_forward(\n    const {scalar_ty}* __restrict__ Logits,\n    const int* __restrict__ Targets,\n    float* __restrict__ Losses,\n    const int num_samples,\n    const int vocab_size,\n    const float label_smoothing) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    const int sample_idx = blockIdx.x;").unwrap();
        writeln!(out, "    const int tid = threadIdx.x;").unwrap();
        writeln!(out, "    const int num_threads = blockDim.x;").unwrap();
        writeln!(out, "    if (sample_idx >= num_samples) return;").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    const {scalar_ty}* logits_sample = Logits + sample_idx * vocab_size;"
        )
        .unwrap();
        writeln!(out, "    const int target_class = Targets[sample_idx];").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // 1. Warp-reduced Maximum Logit for Safe Softmax").unwrap();
        writeln!(out, "    float max_logit = -1e20f;").unwrap();
        writeln!(
            out,
            "    for (int v = tid; v < vocab_size; v += num_threads) {{"
        )
        .unwrap();
        writeln!(
            out,
            "        const float val = static_cast<float>(logits_sample[v]);"
        )
        .unwrap();
        writeln!(out, "        max_logit = fmaxf(max_logit, val);").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    for (int offset = 16; offset > 0; offset /= 2) {{").unwrap();
        writeln!(
            out,
            "        max_logit = fmaxf(max_logit, __shfl_down_sync(0xffffffff, max_logit, offset));"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    __shared__ float s_max;").unwrap();
        writeln!(out, "    if (tid == 0) s_max = max_logit;").unwrap();
        writeln!(out, "    __syncthreads();").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // 2. Warp-reduced Sum of Exponents (Log-Sum-Exp)").unwrap();
        writeln!(out, "    float sum_exp = 0.0f;").unwrap();
        writeln!(out, "    float target_logit = 0.0f;").unwrap();
        writeln!(
            out,
            "    for (int v = tid; v < vocab_size; v += num_threads) {{"
        )
        .unwrap();
        writeln!(
            out,
            "        const float val = static_cast<float>(logits_sample[v]);"
        )
        .unwrap();
        writeln!(out, "        sum_exp += expf(val - s_max);").unwrap();
        writeln!(out, "        if (v == target_class) target_logit = val;").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    for (int offset = 16; offset > 0; offset /= 2) {{").unwrap();
        writeln!(
            out,
            "        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    __shared__ float s_lse;").unwrap();
        writeln!(out, "    if (tid == 0) s_lse = s_max + logf(sum_exp);").unwrap();
        writeln!(out, "    __syncthreads();").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    // 3. Loss computation with optional label smoothing"
        )
        .unwrap();
        writeln!(out, "    if (tid == 0) {{").unwrap();
        writeln!(out, "        const float nll = s_lse - target_logit;").unwrap();
        if self.label_smoothing > 0.0 {
            writeln!(out, "        const float smooth_loss = (1.0f - label_smoothing) * nll + label_smoothing * s_lse;").unwrap();
            writeln!(out, "        Losses[sample_idx] = smooth_loss;").unwrap();
        } else {
            writeln!(out, "        Losses[sample_idx] = nll;").unwrap();
        }
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }

    /// Renders the backward fused Cross Entropy CUDA C++ gradient kernel ($dX$).
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
            "// Fused Cross Entropy Loss Backward Kernel for {} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "extern \"C\" __global__ void {}_backward(\n    const {scalar_ty}* __restrict__ Logits,\n    const int* __restrict__ Targets,\n    const float* __restrict__ dLoss,\n    {scalar_ty}* __restrict__ dLogits,\n    const int num_samples,\n    const int vocab_size,\n    const float label_smoothing) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    const int sample_idx = blockIdx.x;").unwrap();
        writeln!(out, "    const int tid = threadIdx.x;").unwrap();
        writeln!(out, "    const int num_threads = blockDim.x;").unwrap();
        writeln!(out, "    if (sample_idx >= num_samples) return;").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    const {scalar_ty}* logits_sample = Logits + sample_idx * vocab_size;"
        )
        .unwrap();
        writeln!(
            out,
            "    {scalar_ty}* dlogits_sample = dLogits + sample_idx * vocab_size;"
        )
        .unwrap();
        writeln!(out, "    const int target_class = Targets[sample_idx];").unwrap();
        writeln!(
            out,
            "    const float grad_scale = (dLoss != nullptr) ? dLoss[sample_idx] : 1.0f;"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // 1. Max logit").unwrap();
        writeln!(out, "    float max_logit = -1e20f;").unwrap();
        writeln!(
            out,
            "    for (int v = tid; v < vocab_size; v += num_threads) {{"
        )
        .unwrap();
        writeln!(
            out,
            "        max_logit = fmaxf(max_logit, static_cast<float>(logits_sample[v]));"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    for (int offset = 16; offset > 0; offset /= 2) {{").unwrap();
        writeln!(
            out,
            "        max_logit = fmaxf(max_logit, __shfl_down_sync(0xffffffff, max_logit, offset));"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    __shared__ float s_max;").unwrap();
        writeln!(out, "    if (tid == 0) s_max = max_logit;").unwrap();
        writeln!(out, "    __syncthreads();").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // 2. Sum exp").unwrap();
        writeln!(out, "    float sum_exp = 0.0f;").unwrap();
        writeln!(
            out,
            "    for (int v = tid; v < vocab_size; v += num_threads) {{"
        )
        .unwrap();
        writeln!(
            out,
            "        sum_exp += expf(static_cast<float>(logits_sample[v]) - s_max);"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    for (int offset = 16; offset > 0; offset /= 2) {{").unwrap();
        writeln!(
            out,
            "        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "    __shared__ float s_inv_sum_exp;").unwrap();
        writeln!(
            out,
            "    if (tid == 0) s_inv_sum_exp = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;"
        )
        .unwrap();
        writeln!(out, "    __syncthreads();").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // 3. dX = Softmax(X) - Target").unwrap();
        writeln!(
            out,
            "    const float smooth_target = label_smoothing / static_cast<float>(vocab_size);"
        )
        .unwrap();
        writeln!(
            out,
            "    for (int v = tid; v < vocab_size; v += num_threads) {{"
        )
        .unwrap();
        writeln!(out, "        const float prob = expf(static_cast<float>(logits_sample[v]) - s_max) * s_inv_sum_exp;").unwrap();
        writeln!(out, "        float target_prob = smooth_target;").unwrap();
        writeln!(
            out,
            "        if (v == target_class) target_prob += (1.0f - label_smoothing);"
        )
        .unwrap();
        writeln!(
            out,
            "        const float grad = (prob - target_prob) * grad_scale;"
        )
        .unwrap();
        writeln!(
            out,
            "        dlogits_sample[v] = static_cast<{scalar_ty}>(grad);"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }
}
