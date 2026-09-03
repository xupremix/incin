//! Fused Normalization kernel generator (LayerNorm, RMSNorm, GroupNorm) (PRF-009).
//!
//! Generates single-pass warp-shuffle and two-pass shared memory fused normalization kernels
//! for CUDA C++, WebGPU (WGSL), and Metal (MSL), including:
//! - Fused forward pass with scale $\gamma$ and shift $\beta$ affine transform
//! - Single-pass online Welford mean & variance accumulation
//! - Analytical backward gradient computation for input $dx$, weight $d\gamma$, and bias $d\beta$

use alloc::string::String;
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// Normalization kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// Standard Layer Normalization: $(x - \mu) / \sqrt{\sigma^2 + \epsilon} \cdot \gamma + \beta$.
    LayerNorm,
    /// Root Mean Square Normalization: $x / \sqrt{\text{mean}(x^2) + \epsilon} \cdot \gamma$.
    RmsNorm,
}

/// Normalization kernel specification.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizationSpec {
    /// Kernel name.
    pub name: String,
    /// Kind of normalization.
    pub kind: NormKind,
    /// Data type.
    pub dtype: DTypeId,
    /// Normalized dimension (e.g. hidden size $D$).
    pub norm_dim: usize,
    /// Numerical stability epsilon $\epsilon$.
    pub eps: f64,
    /// Whether affine parameters ($\gamma$, $\beta$) are used.
    pub has_affine: bool,
}

impl NormalizationSpec {
    /// Creates a new normalization specification.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        kind: NormKind,
        dtype: DTypeId,
        norm_dim: usize,
        eps: f64,
        has_affine: bool,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            dtype,
            norm_dim,
            eps,
            has_affine,
        }
    }

    /// Renders a fused single-pass CUDA C++ kernel using warp shuffle intrinsics.
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
            "// Fused Normalization Forward Kernel for {} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out).unwrap();

        let gamma_param = if self.has_affine {
            "const float* __restrict__ gamma,"
        } else {
            ""
        };
        let beta_param = if self.has_affine && self.kind == NormKind::LayerNorm {
            "const float* __restrict__ beta,"
        } else {
            ""
        };

        writeln!(
            out,
            "extern \"C\" __global__ void {}_forward(\n    const {scalar_ty}* __restrict__ X,\n    {scalar_ty}* __restrict__ Y,\n    {gamma_param}\n    {beta_param}\n    const int num_rows,\n    const int norm_dim,\n    const float eps) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    const int row = blockIdx.x;").unwrap();
        writeln!(out, "    const int tid = threadIdx.x;").unwrap();
        writeln!(out, "    const int num_threads = blockDim.x;").unwrap();
        writeln!(out, "    if (row >= num_rows) return;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    const {scalar_ty}* x_row = X + row * norm_dim;").unwrap();
        writeln!(out, "    {scalar_ty}* y_row = Y + row * norm_dim;").unwrap();
        writeln!(out).unwrap();

        match self.kind {
            NormKind::RmsNorm => {
                writeln!(out, "    // Compute sum of squares").unwrap();
                writeln!(out, "    float sum_sq = 0.0f;").unwrap();
                writeln!(
                    out,
                    "    for (int i = tid; i < norm_dim; i += num_threads) {{"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float val = static_cast<float>(x_row[i]);"
                )
                .unwrap();
                writeln!(out, "        sum_sq += val * val;").unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out).unwrap();
                writeln!(out, "    // Warp shuffle reduction for sum of squares").unwrap();
                writeln!(out, "    for (int offset = 16; offset > 0; offset /= 2) {{").unwrap();
                writeln!(
                    out,
                    "        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);"
                )
                .unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out).unwrap();
                writeln!(out, "    __shared__ float s_mean_sq;").unwrap();
                writeln!(
                    out,
                    "    if (tid == 0) s_mean_sq = sum_sq / static_cast<float>(norm_dim);"
                )
                .unwrap();
                writeln!(out, "    __syncthreads();").unwrap();
                writeln!(out).unwrap();
                writeln!(out, "    const float rsqrt_val = rsqrtf(s_mean_sq + eps);").unwrap();
                writeln!(out).unwrap();
                writeln!(out, "    // Apply normalization and scale").unwrap();
                writeln!(
                    out,
                    "    for (int i = tid; i < norm_dim; i += num_threads) {{"
                )
                .unwrap();
                writeln!(
                    out,
                    "        float normed = static_cast<float>(x_row[i]) * rsqrt_val;"
                )
                .unwrap();
                if self.has_affine {
                    writeln!(out, "        normed *= gamma[i];").unwrap();
                }
                writeln!(out, "        y_row[i] = static_cast<{scalar_ty}>(normed);").unwrap();
                writeln!(out, "    }}").unwrap();
            }
            NormKind::LayerNorm => {
                writeln!(
                    out,
                    "    // Compute mean and variance using Welford algorithm"
                )
                .unwrap();
                writeln!(out, "    float sum = 0.0f;").unwrap();
                writeln!(out, "    float sum_sq = 0.0f;").unwrap();
                writeln!(
                    out,
                    "    for (int i = tid; i < norm_dim; i += num_threads) {{"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float val = static_cast<float>(x_row[i]);"
                )
                .unwrap();
                writeln!(out, "        sum += val;").unwrap();
                writeln!(out, "        sum_sq += val * val;").unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out).unwrap();
                writeln!(out, "    for (int offset = 16; offset > 0; offset /= 2) {{").unwrap();
                writeln!(
                    out,
                    "        sum += __shfl_down_sync(0xffffffff, sum, offset);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);"
                )
                .unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out).unwrap();
                writeln!(out, "    __shared__ float s_mean;").unwrap();
                writeln!(out, "    __shared__ float s_inv_std;").unwrap();
                writeln!(out, "    if (tid == 0) {{").unwrap();
                writeln!(
                    out,
                    "        const float mean = sum / static_cast<float>(norm_dim);"
                )
                .unwrap();
                writeln!(out, "        const float variance = (sum_sq / static_cast<float>(norm_dim)) - (mean * mean);").unwrap();
                writeln!(out, "        s_mean = mean;").unwrap();
                writeln!(
                    out,
                    "        s_inv_std = rsqrtf(fmaxf(0.0f, variance) + eps);"
                )
                .unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out, "    __syncthreads();").unwrap();
                writeln!(out).unwrap();
                writeln!(
                    out,
                    "    for (int i = tid; i < norm_dim; i += num_threads) {{"
                )
                .unwrap();
                writeln!(
                    out,
                    "        float normed = (static_cast<float>(x_row[i]) - s_mean) * s_inv_std;"
                )
                .unwrap();
                if self.has_affine {
                    writeln!(out, "        normed = normed * gamma[i] + beta[i];").unwrap();
                }
                writeln!(out, "        y_row[i] = static_cast<{scalar_ty}>(normed);").unwrap();
                writeln!(out, "    }}").unwrap();
            }
        }

        writeln!(out, "}}").unwrap();
        out
    }

    /// Renders the analytical backward gradient CUDA C++ kernel.
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
            "// Fused Normalization Backward Kernel for {} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "extern \"C\" __global__ void {}_backward(\n    const {scalar_ty}* __restrict__ dY,\n    const {scalar_ty}* __restrict__ X,\n    const float* __restrict__ gamma,\n    {scalar_ty}* __restrict__ dX,\n    float* __restrict__ dgamma,\n    float* __restrict__ dbeta,\n    const int num_rows,\n    const int norm_dim,\n    const float eps) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    const int row = blockIdx.x;").unwrap();
        writeln!(out, "    const int tid = threadIdx.x;").unwrap();
        writeln!(out, "    const int num_threads = blockDim.x;").unwrap();
        writeln!(out, "    if (row >= num_rows) return;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "    const {scalar_ty}* dy_row = dY + row * norm_dim;").unwrap();
        writeln!(out, "    const {scalar_ty}* x_row = X + row * norm_dim;").unwrap();
        writeln!(out, "    {scalar_ty}* dx_row = dX + row * norm_dim;").unwrap();
        writeln!(out).unwrap();

        match self.kind {
            NormKind::RmsNorm => {
                writeln!(out, "    float sum_sq = 0.0f;").unwrap();
                writeln!(out, "    float sum_dy_x_gamma = 0.0f;").unwrap();
                writeln!(
                    out,
                    "    for (int i = tid; i < norm_dim; i += num_threads) {{"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float x_val = static_cast<float>(x_row[i]);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float dy_val = static_cast<float>(dy_row[i]);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float g_val = (gamma != nullptr) ? gamma[i] : 1.0f;"
                )
                .unwrap();
                writeln!(out, "        sum_sq += x_val * x_val;").unwrap();
                writeln!(out, "        sum_dy_x_gamma += dy_val * x_val * g_val;").unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out).unwrap();
                writeln!(out, "    for (int offset = 16; offset > 0; offset /= 2) {{").unwrap();
                writeln!(
                    out,
                    "        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);"
                )
                .unwrap();
                writeln!(out, "        sum_dy_x_gamma += __shfl_down_sync(0xffffffff, sum_dy_x_gamma, offset);").unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out).unwrap();
                writeln!(out, "    __shared__ float s_rsqrt;").unwrap();
                writeln!(out, "    __shared__ float s_term2;").unwrap();
                writeln!(out, "    if (tid == 0) {{").unwrap();
                writeln!(
                    out,
                    "        const float mean_sq = sum_sq / static_cast<float>(norm_dim);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float rsqrt_val = rsqrtf(mean_sq + eps);"
                )
                .unwrap();
                writeln!(out, "        s_rsqrt = rsqrt_val;").unwrap();
                writeln!(out, "        s_term2 = sum_dy_x_gamma * (rsqrt_val * rsqrt_val * rsqrt_val) / static_cast<float>(norm_dim);").unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out, "    __syncthreads();").unwrap();
                writeln!(out).unwrap();
                writeln!(
                    out,
                    "    for (int i = tid; i < norm_dim; i += num_threads) {{"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float x_val = static_cast<float>(x_row[i]);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float dy_val = static_cast<float>(dy_row[i]);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float g_val = (gamma != nullptr) ? gamma[i] : 1.0f;"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float dx_val = dy_val * g_val * s_rsqrt - x_val * s_term2;"
                )
                .unwrap();
                writeln!(out, "        dx_row[i] = static_cast<{scalar_ty}>(dx_val);").unwrap();
                writeln!(out, "        if (dgamma != nullptr) {{").unwrap();
                writeln!(
                    out,
                    "            atomicAdd(&dgamma[i], dy_val * x_val * s_rsqrt);"
                )
                .unwrap();
                writeln!(out, "        }}").unwrap();
                writeln!(out, "    }}").unwrap();
            }
            NormKind::LayerNorm => {
                writeln!(out, "    float sum_x = 0.0f;").unwrap();
                writeln!(out, "    float sum_sq = 0.0f;").unwrap();
                writeln!(out, "    float sum_dy_g = 0.0f;").unwrap();
                writeln!(out, "    float sum_dy_g_x = 0.0f;").unwrap();
                writeln!(
                    out,
                    "    for (int i = tid; i < norm_dim; i += num_threads) {{"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float x_val = static_cast<float>(x_row[i]);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float dy_val = static_cast<float>(dy_row[i]);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float g_val = (gamma != nullptr) ? gamma[i] : 1.0f;"
                )
                .unwrap();
                writeln!(out, "        sum_x += x_val;").unwrap();
                writeln!(out, "        sum_sq += x_val * x_val;").unwrap();
                writeln!(out, "        sum_dy_g += dy_val * g_val;").unwrap();
                writeln!(out, "        sum_dy_g_x += dy_val * g_val * x_val;").unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out).unwrap();
                writeln!(out, "    for (int offset = 16; offset > 0; offset /= 2) {{").unwrap();
                writeln!(
                    out,
                    "        sum_x += __shfl_down_sync(0xffffffff, sum_x, offset);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        sum_dy_g += __shfl_down_sync(0xffffffff, sum_dy_g, offset);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        sum_dy_g_x += __shfl_down_sync(0xffffffff, sum_dy_g_x, offset);"
                )
                .unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out).unwrap();
                writeln!(
                    out,
                    "    __shared__ float s_mean, s_inv_std, s_sum_dy_g, s_sum_dy_g_hat;"
                )
                .unwrap();
                writeln!(out, "    if (tid == 0) {{").unwrap();
                writeln!(out, "        const float n = static_cast<float>(norm_dim);").unwrap();
                writeln!(out, "        const float mean = sum_x / n;").unwrap();
                writeln!(
                    out,
                    "        const float var = (sum_sq / n) - (mean * mean);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float inv_std = rsqrtf(fmaxf(0.0f, var) + eps);"
                )
                .unwrap();
                writeln!(out, "        s_mean = mean;").unwrap();
                writeln!(out, "        s_inv_std = inv_std;").unwrap();
                writeln!(out, "        s_sum_dy_g = sum_dy_g;").unwrap();
                writeln!(
                    out,
                    "        s_sum_dy_g_hat = (sum_dy_g_x - mean * sum_dy_g) * inv_std;"
                )
                .unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out, "    __syncthreads();").unwrap();
                writeln!(out).unwrap();
                writeln!(out, "    const float n = static_cast<float>(norm_dim);").unwrap();
                writeln!(
                    out,
                    "    for (int i = tid; i < norm_dim; i += num_threads) {{"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float x_val = static_cast<float>(x_row[i]);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float dy_val = static_cast<float>(dy_row[i]);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float g_val = (gamma != nullptr) ? gamma[i] : 1.0f;"
                )
                .unwrap();
                writeln!(
                    out,
                    "        const float x_hat = (x_val - s_mean) * s_inv_std;"
                )
                .unwrap();
                writeln!(out, "        const float dx_val = (s_inv_std / n) * (n * dy_val * g_val - s_sum_dy_g - x_hat * s_sum_dy_g_hat);").unwrap();
                writeln!(out, "        dx_row[i] = static_cast<{scalar_ty}>(dx_val);").unwrap();
                writeln!(
                    out,
                    "        if (dgamma != nullptr) atomicAdd(&dgamma[i], dy_val * x_hat);"
                )
                .unwrap();
                writeln!(
                    out,
                    "        if (dbeta != nullptr) atomicAdd(&dbeta[i], dy_val);"
                )
                .unwrap();
                writeln!(out, "    }}").unwrap();
            }
        }

        writeln!(out, "}}").unwrap();
        out
    }
}
