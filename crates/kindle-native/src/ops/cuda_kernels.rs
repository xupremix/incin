pub const MATMUL_SWIGLU_KERNEL: &str = r#"
extern "C" __global__ void fused_matmul_swiglu(
    const float* A, const float* B, float* C, 
    int M, int K, int N
) {
    // Basic tiled matrix multiplication with SwiGLU activation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        
        // SwiGLU activation: x * sigmoid(beta * x)
        // Note: typically SwiGLU takes two inputs, but for a simple fused
        // kernel we apply SiLU (Swish with beta=1) to the matmul output.
        // A true SwiGLU would have W_gate and W_up.
        float sig = 1.0f / (1.0f + expf(-sum));
        C[row * N + col] = sum * sig;
    }
}
"#;

pub const FLASH_ATTENTION_LITE_KERNEL: &str = r#"
extern "C" __global__ void flash_attention_lite(
    const float* Q, const float* K, const float* V, float* O,
    int seq_len, int head_dim
) {
    // Simplified FlashAttention for demonstration.
    // In SOTA this uses shared memory tiling for Q, K, V.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len) return;

    float max_score = -1e20f;
    float sum_exp = 0.0f;

    // First pass: compute max for numerical stability
    for (int j = 0; j < seq_len; ++j) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += Q[i * head_dim + d] * K[j * head_dim + d];
        }
        score /= sqrtf((float)head_dim);
        if (score > max_score) max_score = score;
    }

    // Second pass: compute sum of exps and output
    for (int j = 0; j < seq_len; ++j) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += Q[i * head_dim + d] * K[j * head_dim + d];
        }
        score /= sqrtf((float)head_dim);
        float exp_score = expf(score - max_score);
        sum_exp += exp_score;
        
        for (int d = 0; d < head_dim; ++d) {
            O[i * head_dim + d] += exp_score * V[j * head_dim + d];
        }
    }

    // Normalize
    for (int d = 0; d < head_dim; ++d) {
        O[i * head_dim + d] /= sum_exp;
    }
}
"#;

pub const FUSED_ADAMW_KERNEL: &str = r#"
extern "C" __global__ void fused_adamw_step(
    const float* __restrict__ p,
    float* __restrict__ p_out,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float wd,
    const int step,
    const int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Vectorized loads with float4
    if (idx < num_elements / 4) {
        float4 p_vec = reinterpret_cast<const float4*>(p)[idx];
        float4 g_vec = reinterpret_cast<const float4*>(g)[idx];
        float4 m_vec = reinterpret_cast<const float4*>(m)[idx];
        float4 v_vec = reinterpret_cast<const float4*>(v)[idx];
        float4 p_out_vec;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float p_val = ((float*)&p_vec)[i];
            float g_val = ((float*)&g_vec)[i];
            float m_val = ((float*)&m_vec)[i];
            float v_val = ((float*)&v_vec)[i];
            
            p_val = p_val - lr * wd * p_val; // weight decay
            m_val = beta1 * m_val + (1.0f - beta1) * g_val; // moment 1
            v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val; // moment 2
            float p_new = p_val - lr * m_val / (sqrtf(v_val) + eps); // update
            
            ((float*)&m_vec)[i] = m_val;
            ((float*)&v_vec)[i] = v_val;
            ((float*)&p_out_vec)[i] = p_new;
        }
        
        reinterpret_cast<float4*>(m)[idx] = m_vec;
        reinterpret_cast<float4*>(v)[idx] = v_vec;
        reinterpret_cast<float4*>(p_out)[idx] = p_out_vec;
    } 
    
    // Tail
    int tail_start = (num_elements / 4) * 4;
    int tail_idx = tail_start + idx;
    if (idx == 0 && tail_idx < num_elements) {
        for(int i = tail_start; i < num_elements; i++) {
            float p_val = p[i];
            float g_val = g[i];
            float m_val = m[i];
            float v_val = v[i];
            
            p_val = p_val - lr * wd * p_val;
            m_val = beta1 * m_val + (1.0f - beta1) * g_val;
            v_val = beta2 * v_val + (1.0f - beta2) * g_val * g_val;
            float p_new = p_val - lr * m_val / (sqrtf(v_val) + eps);
            
            m[i] = m_val;
            v[i] = v_val;
            p_out[i] = p_new;
        }
    }
}
"#;

pub const MATMUL_KERNEL: &str = r#"
extern "C" __global__ void matmul(
    const float* A, const float* B, float* C, 
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
"#;
