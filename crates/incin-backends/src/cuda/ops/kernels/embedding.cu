#include <cuda_fp16.h>
#include <cuda_bf16.h>

typedef long long int64_t;
typedef unsigned int uint32_t;

// ----------------------------------------------------------------------------
// Template Forward Implementation
// ----------------------------------------------------------------------------
template <typename T>
__device__ void embedding_forward_impl(
    const int64_t* __restrict__ indices,
    const T* __restrict__ weight,
    T* __restrict__ output,
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

    T* __restrict__ out_row = output + idx_idx * hidden_size;
    const T* __restrict__ weight_row = weight + row_idx * hidden_size;

    for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        out_row[h] = weight_row[h];
    }
}

// ----------------------------------------------------------------------------
// Template Backward Implementation
// ----------------------------------------------------------------------------
template <typename T>
__device__ void embedding_backward_impl(
    const T* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    T* __restrict__ grad_weight,
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

    const T* __restrict__ grad_out_row = grad_output + idx_idx * hidden_size;
    T* __restrict__ grad_w_row = grad_weight + row_idx * hidden_size;

    for (size_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        atomicAdd(&grad_w_row[h], grad_out_row[h]);
    }
}

// ----------------------------------------------------------------------------
// Exported C Entry Points
// ----------------------------------------------------------------------------

extern "C" __global__ void embedding_forward(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    embedding_forward_impl<float>(indices, weight, output, error_flag, num_indices, vocab_size, hidden_size);
}

extern "C" __global__ void embedding_forward_f32(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    embedding_forward_impl<float>(indices, weight, output, error_flag, num_indices, vocab_size, hidden_size);
}

extern "C" __global__ void embedding_forward_f64(
    const int64_t* __restrict__ indices,
    const double* __restrict__ weight,
    double* __restrict__ output,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    embedding_forward_impl<double>(indices, weight, output, error_flag, num_indices, vocab_size, hidden_size);
}

extern "C" __global__ void embedding_forward_f16(
    const int64_t* __restrict__ indices,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    embedding_forward_impl<__half>(indices, weight, output, error_flag, num_indices, vocab_size, hidden_size);
}

extern "C" __global__ void embedding_forward_bf16(
    const int64_t* __restrict__ indices,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    embedding_forward_impl<__nv_bfloat16>(indices, weight, output, error_flag, num_indices, vocab_size, hidden_size);
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
    embedding_backward_impl<float>(grad_output, indices, grad_weight, error_flag, num_indices, vocab_size, hidden_size);
}

extern "C" __global__ void embedding_backward_f32(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_weight,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    embedding_backward_impl<float>(grad_output, indices, grad_weight, error_flag, num_indices, vocab_size, hidden_size);
}

extern "C" __global__ void embedding_backward_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_weight,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    embedding_backward_impl<double>(grad_output, indices, grad_weight, error_flag, num_indices, vocab_size, hidden_size);
}

extern "C" __global__ void embedding_backward_f16(
    const __half* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    __half* __restrict__ grad_weight,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    embedding_backward_impl<__half>(grad_output, indices, grad_weight, error_flag, num_indices, vocab_size, hidden_size);
}

extern "C" __global__ void embedding_backward_bf16(
    const __nv_bfloat16* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    __nv_bfloat16* __restrict__ grad_weight,
    uint32_t* __restrict__ error_flag,
    const size_t num_indices,
    const size_t vocab_size,
    const size_t hidden_size)
{
    embedding_backward_impl<__nv_bfloat16>(grad_output, indices, grad_weight, error_flag, num_indices, vocab_size, hidden_size);
}
