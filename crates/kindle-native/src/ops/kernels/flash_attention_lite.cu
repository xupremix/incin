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
