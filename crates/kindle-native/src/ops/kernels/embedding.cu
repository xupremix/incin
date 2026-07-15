extern "C" __global__ void embedding_forward(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    // Each block handles one index
    const size_t idx_idx = blockIdx.x;
    if (idx_idx >= num_indices) return;

    int64_t row_idx = indices[idx_idx];
    if (row_idx < 0 || row_idx >= vocab_size) {
        // Output zero if index is out of bounds (should ideally be checked on CPU, or return error)
        for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            output[idx_idx * hidden_size + h] = 0.0f;
        }
        return;
    }

    const float* __restrict__ weight_row = weight + row_idx * hidden_size;
    float* __restrict__ out_row = output + idx_idx * hidden_size;

    for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        out_row[h] = weight_row[h];
    }
}

extern "C" __global__ void embedding_backward(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_weight,
    const size_t num_indices,
    const size_t hidden_size)
{
    // Each block handles one index from the input sequence
    const size_t idx_idx = blockIdx.x;
    if (idx_idx >= num_indices) return;

    int64_t row_idx = indices[idx_idx];
    if (row_idx < 0) return;

    const float* __restrict__ grad_out_row = grad_output + idx_idx * hidden_size;
    float* __restrict__ grad_w_row = grad_weight + row_idx * hidden_size;

    for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        atomicAdd(&grad_w_row[h], grad_out_row[h]);
    }
}
