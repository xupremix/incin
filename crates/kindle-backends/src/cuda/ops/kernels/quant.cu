#include <cuda_fp16.h>

// NVRTC has no default header search path for host libc headers (only for
// the CUDA toolkit's own headers, via --include-path), so <stdint.h> isn't
// resolvable; define the one typedef we need instead.
typedef signed char int8_t;

struct __align__(2) BlockQ8_0 {
    half d;
    int8_t qs[32];
};

extern "C" __global__ void quantize_q8_0(
    const float* __restrict__ inp,
    BlockQ8_0* __restrict__ out,
    const int num_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;

    const float* chunk = inp + (idx * 32);
    
    float max_abs = 0.0f;
    for (int i = 0; i < 32; i++) {
        float val = chunk[i];
        float abs_val = val >= 0.0f ? val : -val;
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    
    float d = max_abs / 127.0f;
    float inv_d = d == 0.0f ? 0.0f : 1.0f / d;
    
    out[idx].d = __float2half(d);
    
    for (int i = 0; i < 32; i++) {
        float q = roundf(chunk[i] * inv_d);
        out[idx].qs[i] = (int8_t)q;
    }
}

extern "C" __global__ void dequantize_q8_0(
    const BlockQ8_0* __restrict__ inp,
    float* __restrict__ out,
    const int num_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;

    BlockQ8_0 block = inp[idx];
    float d = __half2float(block.d);
    
    float* out_chunk = out + (idx * 32);
    for (int i = 0; i < 32; i++) {
        out_chunk[i] = (float)block.qs[i] * d;
    }
}
