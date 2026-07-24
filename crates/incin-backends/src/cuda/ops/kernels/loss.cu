// ----------------------------------------------------------------------------
// nll_loss
// ----------------------------------------------------------------------------
extern "C" __global__ void nll_loss(
    const float* __restrict__ log_sm,
    const int* __restrict__ target_buf,
    float* __restrict__ out,
    const int batch,
    const int n_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch) return;

    int t = target_buf[idx];
    if (t >= 0 && t < n_classes) {
        out[idx] = -log_sm[idx * n_classes + t];
    } else {
        out[idx] = 0.0f;
    }
}
