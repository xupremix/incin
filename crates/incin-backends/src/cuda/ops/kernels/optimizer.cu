// ----------------------------------------------------------------------------
// Fused AdamW Optimizer Step Kernel
// Computes:
//   m = beta1 * m + (1 - beta1) * g
//   v = beta2 * v + (1 - beta2) * g * g
//   m_hat = m / (1 - beta1^t)
//   v_hat = v / (1 - beta2^t)
//   p = p - lr * weight_decay * p - lr * m_hat / (sqrt(v_hat) + eps)
// ----------------------------------------------------------------------------

extern "C" __global__ void adamw_step_f32(
    float* __restrict__ p_out,
    float* __restrict__ m_out,
    float* __restrict__ v_out,
    const float* __restrict__ p_in,
    const float* __restrict__ grad,
    const float* __restrict__ m_in,
    const float* __restrict__ v_in,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float bias_correction1,
    const float bias_correction2,
    const size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float p = p_in[idx];
    float g = grad[idx];
    float m = (m_in != nullptr) ? m_in[idx] : 0.0f;
    float v = (v_in != nullptr) ? v_in[idx] : 0.0f;

    // 1. Update biased 1st & 2nd moment estimates
    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;

    // 2. Compute bias-corrected moments
    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;

    // 3. Apply decoupled weight decay & update parameter
    p = p - lr * weight_decay * p - lr * (m_hat / (sqrtf(v_hat) + eps));

    // 4. Store outputs
    p_out[idx] = p;
    m_out[idx] = m;
    v_out[idx] = v;
}
