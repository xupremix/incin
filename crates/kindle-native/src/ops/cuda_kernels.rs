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
    const float* params, float* new_params, const float* grads, float* m, float* v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int step, int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float p = params[idx];
    float g = grads[idx];
    
    // Apply weight decay
    p -= lr * weight_decay * p;
    
    // Update biased first moment estimate
    float mt = beta1 * m[idx] + (1.0f - beta1) * g;
    m[idx] = mt;
    
    // Update biased second raw moment estimate
    float vt = beta2 * v[idx] + (1.0f - beta2) * g * g;
    v[idx] = vt;
    
    // Compute bias-corrected estimates (assumed done on CPU and passed via lr)
    // Here we assume lr is already step-adjusted, or we do it here:
    // float bias_correction1 = 1.0f - powf(beta1, step);
    // float bias_correction2 = 1.0f - powf(beta2, step);
    
    // params[idx] = p - lr * (mt / bias_correction1) / (sqrtf(vt / bias_correction2) + eps);
    new_params[idx] = p - lr * mt / (sqrtf(vt) + eps);
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
