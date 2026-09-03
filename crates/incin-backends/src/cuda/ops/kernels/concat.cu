// Multi-width concatenation kernel for CUDA.

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned long long uint64_t;

template <typename T>
__device__ void concat_impl(
    const T* __restrict__ input,
    T* __restrict__ output,
    uint32_t outer_size,
    uint32_t in_dim_size,
    uint32_t out_dim_size,
    uint32_t inner_size,
    uint32_t offset
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = outer_size * in_dim_size * inner_size;
    if (idx >= total_elements) return;

    uint32_t inner_idx = idx % inner_size;
    uint32_t rem = idx / inner_size;
    uint32_t dim_idx = rem % in_dim_size;
    uint32_t outer_idx = rem / in_dim_size;

    uint32_t out_idx = outer_idx * (out_dim_size * inner_size) + 
                       (dim_idx + offset) * inner_size + 
                       inner_idx;
                       
    output[out_idx] = input[idx];
}

extern "C" __global__ void concat_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t outer_size,
    uint32_t in_dim_size,
    uint32_t out_dim_size,
    uint32_t inner_size,
    uint32_t offset
) {
    concat_impl<float>(input, output, outer_size, in_dim_size, out_dim_size, inner_size, offset);
}

extern "C" __global__ void concat_8bit(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint32_t outer_size,
    uint32_t in_dim_size,
    uint32_t out_dim_size,
    uint32_t inner_size,
    uint32_t offset
) {
    concat_impl<uint8_t>(input, output, outer_size, in_dim_size, out_dim_size, inner_size, offset);
}

extern "C" __global__ void concat_16bit(
    const uint16_t* __restrict__ input,
    uint16_t* __restrict__ output,
    uint32_t outer_size,
    uint32_t in_dim_size,
    uint32_t out_dim_size,
    uint32_t inner_size,
    uint32_t offset
) {
    concat_impl<uint16_t>(input, output, outer_size, in_dim_size, out_dim_size, inner_size, offset);
}

extern "C" __global__ void concat_32bit(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t outer_size,
    uint32_t in_dim_size,
    uint32_t out_dim_size,
    uint32_t inner_size,
    uint32_t offset
) {
    concat_impl<uint32_t>(input, output, outer_size, in_dim_size, out_dim_size, inner_size, offset);
}

extern "C" __global__ void concat_64bit(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    uint32_t outer_size,
    uint32_t in_dim_size,
    uint32_t out_dim_size,
    uint32_t inner_size,
    uint32_t offset
) {
    concat_impl<uint64_t>(input, output, outer_size, in_dim_size, out_dim_size, inner_size, offset);
}
