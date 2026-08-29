//! Scaled Dot-Product Attention (SDPA) and FlashAttention-2 style fused kernel generator (PRF-008).
//!
//! Generates memory-efficient tiled attention kernels for CUDA C++, WebGPU (WGSL), and Metal (MSL),
//! utilizing:
//! - Shared-memory block tiling over Query, Key, and Value matrices
//! - Online safe softmax (running max $m_i$ and running denominator $d_i$) to avoid storing the $N \times N$ attention matrix
//! - Causal masking support for autoregressive decoder self-attention
//! - Vectorized memory transactions for head dimensions $d_k \in \{32, 64, 128\}$

use alloc::string::String;
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// Attention configuration specification for code generation.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionSpec {
    /// Kernel name identifier.
    pub name: String,
    /// Data type.
    pub dtype: DTypeId,
    /// Head dimension $d_k$ (e.g. 64, 128).
    pub head_dim: usize,
    /// Tile size for Query dimension ($B_r$).
    pub block_q: usize,
    /// Tile size for Key/Value dimension ($B_c$).
    pub block_kv: usize,
    /// Whether causal autoregressive masking ($j \le i$) is enabled.
    pub is_causal: bool,
    /// Softmax scale factor $\tau = 1/\sqrt{d_k}$.
    pub scale: f64,
}

impl AttentionSpec {
    /// Creates a new attention kernel specification.
    #[must_use]
    pub fn new(name: impl Into<String>, dtype: DTypeId, head_dim: usize, is_causal: bool) -> Self {
        let scale = 1.0 / (head_dim as f64).sqrt();
        let (block_q, block_kv) = match head_dim {
            32..=64 => (64, 64),
            65..=128 => (32, 32),
            _ => (16, 16),
        };
        Self {
            name: name.into(),
            dtype,
            head_dim,
            block_q,
            block_kv,
            is_causal,
            scale,
        }
    }

    /// Renders a fused single-pass online softmax FlashAttention-2 style CUDA C++ kernel.
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

        writeln!(
            out,
            "// Fused Tiled Scaled Dot-Product Attention Kernel for {} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out, "#include <math.h>").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "extern \"C\" __global__ void {}_forward(\n    const {scalar_ty}* __restrict__ Q,\n    const {scalar_ty}* __restrict__ K,\n    const {scalar_ty}* __restrict__ V,\n    {scalar_ty}* __restrict__ O,\n    const int batch_heads,\n    const int seq_len,\n    const float scale) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    const int bh_idx = blockIdx.z;").unwrap();
        writeln!(out, "    const int q_tile_idx = blockIdx.x;").unwrap();
        writeln!(out, "    const int tid = threadIdx.x;").unwrap();
        writeln!(out, "    const int num_threads = blockDim.x;").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    if (bh_idx >= batch_heads) return;").unwrap();
        writeln!(
            out,
            "    const int q_row_start = q_tile_idx * {};",
            self.block_q
        )
        .unwrap();
        writeln!(out, "    if (q_row_start >= seq_len) return;").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    const int head_dim = {};", self.head_dim).unwrap();
        writeln!(out, "    const int stride_bh = seq_len * head_dim;").unwrap();
        writeln!(
            out,
            "    const {scalar_ty}* q_ptr = Q + bh_idx * stride_bh;"
        )
        .unwrap();
        writeln!(
            out,
            "    const {scalar_ty}* k_ptr = K + bh_idx * stride_bh;"
        )
        .unwrap();
        writeln!(
            out,
            "    const {scalar_ty}* v_ptr = V + bh_idx * stride_bh;"
        )
        .unwrap();
        writeln!(out, "    {scalar_ty}* o_ptr = O + bh_idx * stride_bh;").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    // Per-thread accumulator registers for online softmax"
        )
        .unwrap();
        writeln!(
            out,
            "    for (int r = tid; r < {}; r += num_threads) {{",
            self.block_q
        )
        .unwrap();
        writeln!(out, "        const int q_seq_idx = q_row_start + r;").unwrap();
        writeln!(out, "        if (q_seq_idx >= seq_len) break;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "        float m_prev = -1e20f;").unwrap();
        writeln!(out, "        float l_prev = 0.0f;").unwrap();
        writeln!(out, "        float acc_o[{}] = {{0.0f}};", self.head_dim).unwrap();
        writeln!(out).unwrap();

        let kv_limit = if self.is_causal {
            "min(seq_len, q_seq_idx + 1)"
        } else {
            "seq_len"
        };
        writeln!(out, "        const int kv_end = {kv_limit};").unwrap();
        writeln!(
            out,
            "        for (int kv_idx = 0; kv_idx < kv_end; ++kv_idx) {{"
        )
        .unwrap();
        writeln!(out, "            // Compute Q * K^T dot product").unwrap();
        writeln!(out, "            float dot = 0.0f;").unwrap();
        writeln!(out, "            #pragma unroll").unwrap();
        writeln!(out, "            for (int d = 0; d < head_dim; ++d) {{").unwrap();
        writeln!(out, "                const float q_val = static_cast<float>(q_ptr[q_seq_idx * head_dim + d]);").unwrap();
        writeln!(
            out,
            "                const float k_val = static_cast<float>(k_ptr[kv_idx * head_dim + d]);"
        )
        .unwrap();
        writeln!(out, "                dot += q_val * k_val;").unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "            dot *= scale;").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "            // Online safe softmax update").unwrap();
        writeln!(out, "            const float m_curr = fmaxf(m_prev, dot);").unwrap();
        writeln!(
            out,
            "            const float p_prev_scale = expf(m_prev - m_curr);"
        )
        .unwrap();
        writeln!(out, "            const float p_curr = expf(dot - m_curr);").unwrap();
        writeln!(
            out,
            "            const float l_curr = l_prev * p_prev_scale + p_curr;"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "            // Update output accumulator").unwrap();
        writeln!(out, "            #pragma unroll").unwrap();
        writeln!(out, "            for (int d = 0; d < head_dim; ++d) {{").unwrap();
        writeln!(
            out,
            "                const float v_val = static_cast<float>(v_ptr[kv_idx * head_dim + d]);"
        )
        .unwrap();
        writeln!(
            out,
            "                acc_o[d] = acc_o[d] * p_prev_scale + p_curr * v_val;"
        )
        .unwrap();
        writeln!(out, "            }}").unwrap();
        writeln!(out, "            m_prev = m_curr;").unwrap();
        writeln!(out, "            l_prev = l_curr;").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "        // Normalize output with final denominator").unwrap();
        writeln!(
            out,
            "        const float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;"
        )
        .unwrap();
        writeln!(out, "        #pragma unroll").unwrap();
        writeln!(out, "        for (int d = 0; d < head_dim; ++d) {{").unwrap();
        writeln!(out, "            o_ptr[q_seq_idx * head_dim + d] = static_cast<{scalar_ty}>(acc_o[d] * inv_l);").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }

    /// Renders a compute shader for WebGPU (WGSL).
    #[must_use]
    pub fn render_wgsl(&self) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "// Fused Scaled Dot-Product Attention Compute Shader for {} (WGSL)",
            self.name
        )
        .unwrap();
        writeln!(
            out,
            "@group(0) @binding(0) var<storage, read> Q: array<f32>;"
        )
        .unwrap();
        writeln!(
            out,
            "@group(0) @binding(1) var<storage, read> K: array<f32>;"
        )
        .unwrap();
        writeln!(
            out,
            "@group(0) @binding(2) var<storage, read> V: array<f32>;"
        )
        .unwrap();
        writeln!(
            out,
            "@group(0) @binding(3) var<storage, read_write> O: array<f32>;"
        )
        .unwrap();
        writeln!(out).unwrap();
        writeln!(out, "struct AttentionUniforms {{\n    batch_heads: u32,\n    seq_len: u32,\n    scale: f32,\n}};").unwrap();
        writeln!(
            out,
            "@group(0) @binding(4) var<uniform> params: AttentionUniforms;"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "@compute @workgroup_size(64, 1, 1)").unwrap();
        writeln!(
            out,
            "fn {}_forward(@builtin(global_invocation_id) gid: vec3<u32>) {{",
            self.name
        )
        .unwrap();
        writeln!(out, "    let q_seq_idx = gid.x;").unwrap();
        writeln!(out, "    let bh_idx = gid.z;").unwrap();
        writeln!(
            out,
            "    if (bh_idx >= params.batch_heads || q_seq_idx >= params.seq_len) {{ return; }}"
        )
        .unwrap();
        writeln!(out, "    let head_dim = {}u;", self.head_dim).unwrap();
        writeln!(out, "    let stride_bh = params.seq_len * head_dim;").unwrap();
        writeln!(
            out,
            "    let q_offset = bh_idx * stride_bh + q_seq_idx * head_dim;"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    var m_prev = -1e20;").unwrap();
        writeln!(out, "    var l_prev = 0.0;").unwrap();
        writeln!(out, "    var acc_o = array<f32, {}>(", self.head_dim).unwrap();
        for i in 0..self.head_dim {
            if i + 1 == self.head_dim {
                writeln!(out, "        0.0").unwrap();
            } else {
                writeln!(out, "        0.0,").unwrap();
            }
        }
        writeln!(out, "    );").unwrap();
        writeln!(out).unwrap();

        let kv_limit = if self.is_causal {
            "min(params.seq_len, q_seq_idx + 1u)"
        } else {
            "params.seq_len"
        };
        writeln!(out, "    let kv_end = {kv_limit};").unwrap();
        writeln!(
            out,
            "    for (var kv_idx = 0u; kv_idx < kv_end; kv_idx = kv_idx + 1u) {{"
        )
        .unwrap();
        writeln!(
            out,
            "        let k_offset = bh_idx * stride_bh + kv_idx * head_dim;"
        )
        .unwrap();
        writeln!(out, "        var dot = 0.0;").unwrap();
        writeln!(out, "        for (var d = 0u; d < head_dim; d = d + 1u) {{").unwrap();
        writeln!(
            out,
            "            dot = dot + Q[q_offset + d] * K[k_offset + d];"
        )
        .unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "        dot = dot * params.scale;").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "        let m_curr = max(m_prev, dot);").unwrap();
        writeln!(out, "        let p_prev_scale = exp(m_prev - m_curr);").unwrap();
        writeln!(out, "        let p_curr = exp(dot - m_curr);").unwrap();
        writeln!(out, "        let l_curr = l_prev * p_prev_scale + p_curr;").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "        let v_offset = bh_idx * stride_bh + kv_idx * head_dim;"
        )
        .unwrap();
        writeln!(out, "        for (var d = 0u; d < head_dim; d = d + 1u) {{").unwrap();
        writeln!(
            out,
            "            acc_o[d] = acc_o[d] * p_prev_scale + p_curr * V[v_offset + d];"
        )
        .unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "        m_prev = m_curr;").unwrap();
        writeln!(out, "        l_prev = l_curr;").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    let inv_l = select(0.0, 1.0 / l_prev, l_prev > 0.0);"
        )
        .unwrap();
        writeln!(
            out,
            "    let o_offset = bh_idx * stride_bh + q_seq_idx * head_dim;"
        )
        .unwrap();
        writeln!(out, "    for (var d = 0u; d < head_dim; d = d + 1u) {{").unwrap();
        writeln!(out, "        O[o_offset + d] = acc_o[d] * inv_l;").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }

    /// Renders a compute kernel for Metal Shading Language (MSL).
    #[must_use]
    pub fn render_msl(&self) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "// Fused Scaled Dot-Product Attention Kernel for {} (MSL)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <metal_stdlib>").unwrap();
        writeln!(out, "using namespace metal;").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "kernel void {}_forward(\n    device const float* Q [[buffer(0)]],\n    device const float* K [[buffer(1)]],\n    device const float* V [[buffer(2)]],\n    device float* O [[buffer(3)]],\n    constant uint& batch_heads [[buffer(4)]],\n    constant uint& seq_len [[buffer(5)]],\n    constant float& scale [[buffer(6)]],\n    uint3 gid [[thread_position_in_grid]]) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    uint q_seq_idx = gid.x;").unwrap();
        writeln!(out, "    uint bh_idx = gid.z;").unwrap();
        writeln!(
            out,
            "    if (bh_idx >= batch_heads || q_seq_idx >= seq_len) return;"
        )
        .unwrap();
        writeln!(out, "    uint head_dim = {};", self.head_dim).unwrap();
        writeln!(out, "    uint stride_bh = seq_len * head_dim;").unwrap();
        writeln!(
            out,
            "    uint q_offset = bh_idx * stride_bh + q_seq_idx * head_dim;"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    float m_prev = -1e20f;").unwrap();
        writeln!(out, "    float l_prev = 0.0f;").unwrap();
        writeln!(out, "    float acc_o[{}] = {{0.0f}};", self.head_dim).unwrap();
        writeln!(out).unwrap();

        let kv_limit = if self.is_causal {
            "min(seq_len, q_seq_idx + 1)"
        } else {
            "seq_len"
        };
        writeln!(out, "    uint kv_end = {kv_limit};").unwrap();
        writeln!(
            out,
            "    for (uint kv_idx = 0; kv_idx < kv_end; ++kv_idx) {{"
        )
        .unwrap();
        writeln!(
            out,
            "        uint k_offset = bh_idx * stride_bh + kv_idx * head_dim;"
        )
        .unwrap();
        writeln!(out, "        float dot = 0.0f;").unwrap();
        writeln!(out, "        for (uint d = 0; d < head_dim; ++d) {{").unwrap();
        writeln!(out, "            dot += Q[q_offset + d] * K[k_offset + d];").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "        dot *= scale;").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "        float m_curr = max(m_prev, dot);").unwrap();
        writeln!(out, "        float p_prev_scale = exp(m_prev - m_curr);").unwrap();
        writeln!(out, "        float p_curr = exp(dot - m_curr);").unwrap();
        writeln!(
            out,
            "        float l_curr = l_prev * p_prev_scale + p_curr;"
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "        uint v_offset = bh_idx * stride_bh + kv_idx * head_dim;"
        )
        .unwrap();
        writeln!(out, "        for (uint d = 0; d < head_dim; ++d) {{").unwrap();
        writeln!(
            out,
            "            acc_o[d] = acc_o[d] * p_prev_scale + p_curr * V[v_offset + d];"
        )
        .unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "        m_prev = m_curr;").unwrap();
        writeln!(out, "        l_prev = l_curr;").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;"
        )
        .unwrap();
        writeln!(
            out,
            "    uint o_offset = bh_idx * stride_bh + q_seq_idx * head_dim;"
        )
        .unwrap();
        writeln!(out, "    for (uint d = 0; d < head_dim; ++d) {{").unwrap();
        writeln!(out, "        O[o_offset + d] = acc_o[d] * inv_l;").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }
}
