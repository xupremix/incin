typedef long long int64_t;
typedef unsigned int uint32_t;

extern "C" __global__ void embedding_forward(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    const size_t idx_idx = blockIdx.x;
    if (idx_idx >= num_indices) return;

    int64_t row_idx = indices[idx_idx];
    if (row_idx < 0 || (size_t)row_idx >= vocab_size) {
        atomicExch(error_flag, 1);
        return;
    }

    float* __restrict__ out_row = output + idx_idx * hidden_size;
    const float* __restrict__ weight_row = weight + row_idx * hidden_size;

    if (hidden_size % 4 == 0) {
        const float4* w4 = reinterpret_cast<const float4*>(weight_row);
        float4* o4 = reinterpret_cast<float4*>(out_row);
        size_t h4 = hidden_size / 4;
        for (size_t h = threadIdx.x; h < h4; h += blockDim.x) {
            o4[h] = w4[h];
        }
    } else {
        for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            out_row[h] = weight_row[h];
        }
    }
}

extern "C" __global__ void embedding_backward(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_weight,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    const size_t idx_idx = blockIdx.x;
    if (idx_idx >= num_indices) return;

    int64_t row_idx = indices[idx_idx];
    if (row_idx < 0 || (size_t)row_idx >= vocab_size) {
        atomicExch(error_flag, 1);
        return;
    }

    const float* __restrict__ grad_out_row = grad_output + idx_idx * hidden_size;
    float* __restrict__ grad_w_row = grad_weight + row_idx * hidden_size;

    if (hidden_size % 4 == 0) {
        const float4* g4 = reinterpret_cast<const float4*>(grad_out_row);
        size_t h4 = hidden_size / 4;
        for (size_t h = threadIdx.x; h < h4; h += blockDim.x) {
            float4 val = g4[h];
            atomicAdd(&grad_w_row[h * 4 + 0], val.x);
            atomicAdd(&grad_w_row[h * 4 + 1], val.y);
            atomicAdd(&grad_w_row[h * 4 + 2], val.z);
            atomicAdd(&grad_w_row[h * 4 + 3], val.w);
        }
    } else {
        for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            atomicAdd(&grad_w_row[h], grad_out_row[h]);
        }
    }
}
