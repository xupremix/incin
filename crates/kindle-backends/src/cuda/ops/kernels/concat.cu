// NVRTC has no default header search path, so <stdint.h> isn't resolvable
// without an explicit --include-path; define the one typedef we need instead.
typedef unsigned int uint32_t;

extern "C" __global__ void concat_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
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
