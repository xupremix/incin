// ----------------------------------------------------------------------------
// Fused Optimizer Kernels (Adam, AdamW, SGD)
// ----------------------------------------------------------------------------

typedef unsigned int uint32_t;

// ----------------------------------------------------------------------------
// AdamW Step
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

    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;

    p = p - lr * weight_decay * p - lr * (m_hat / (sqrtf(v_hat) + eps));

    p_out[idx] = p;
    m_out[idx] = m;
    v_out[idx] = v;
}

extern "C" __global__ void adamw_step_f64(
    double* __restrict__ p_out,
    double* __restrict__ m_out,
    double* __restrict__ v_out,
    const double* __restrict__ p_in,
    const double* __restrict__ grad,
    const double* __restrict__ m_in,
    const double* __restrict__ v_in,
    const double lr,
    const double beta1,
    const double beta2,
    const double eps,
    const double weight_decay,
    const double bias_correction1,
    const double bias_correction2,
    const size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    double p = p_in[idx];
    double g = grad[idx];
    double m = (m_in != nullptr) ? m_in[idx] : 0.0;
    double v = (v_in != nullptr) ? v_in[idx] : 0.0;

    m = beta1 * m + (1.0 - beta1) * g;
    v = beta2 * v + (1.0 - beta2) * g * g;

    double m_hat = m / bias_correction1;
    double v_hat = v / bias_correction2;

    p = p - lr * weight_decay * p - lr * (m_hat / (sqrt(v_hat) + eps));

    p_out[idx] = p;
    m_out[idx] = m;
    v_out[idx] = v;
}

// ----------------------------------------------------------------------------
// Standard Adam Step (without decoupled weight decay)
// ----------------------------------------------------------------------------

extern "C" __global__ void adam_step_f32(
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

    m = beta1 * m + (1.0f - beta1) * g;
    v = beta2 * v + (1.0f - beta2) * g * g;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;

    p = p - lr * (m_hat / (sqrtf(v_hat) + eps));

    p_out[idx] = p;
    m_out[idx] = m;
    v_out[idx] = v;
}

extern "C" __global__ void adam_step_f64(
    double* __restrict__ p_out,
    double* __restrict__ m_out,
    double* __restrict__ v_out,
    const double* __restrict__ p_in,
    const double* __restrict__ grad,
    const double* __restrict__ m_in,
    const double* __restrict__ v_in,
    const double lr,
    const double beta1,
    const double beta2,
    const double eps,
    const double bias_correction1,
    const double bias_correction2,
    const size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    double p = p_in[idx];
    double g = grad[idx];
    double m = (m_in != nullptr) ? m_in[idx] : 0.0;
    double v = (v_in != nullptr) ? v_in[idx] : 0.0;

    m = beta1 * m + (1.0 - beta1) * g;
    v = beta2 * v + (1.0 - beta2) * g * g;

    double m_hat = m / bias_correction1;
    double v_hat = v / bias_correction2;

    p = p - lr * (m_hat / (sqrt(v_hat) + eps));

    p_out[idx] = p;
    m_out[idx] = m;
    v_out[idx] = v;
}

// ----------------------------------------------------------------------------
// SGD Step
// ----------------------------------------------------------------------------

extern "C" __global__ void sgd_step_f32(
    float* __restrict__ p_out,
    const float* __restrict__ p_in,
    const float* __restrict__ grad,
    const float lr,
    const size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    p_out[idx] = p_in[idx] - lr * grad[idx];
}

extern "C" __global__ void sgd_step_f64(
    double* __restrict__ p_out,
    const double* __restrict__ p_in,
    const double* __restrict__ grad,
    const double lr,
    const size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    p_out[idx] = p_in[idx] - lr * grad[idx];
}
