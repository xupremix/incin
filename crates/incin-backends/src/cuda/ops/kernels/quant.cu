#include <cuda_fp16.h>
#include <cuda_bf16.h>

// NVRTC has no default header search path for host libc headers (only for
// the CUDA toolkit's own headers, via --include-path), so <stdint.h> isn't
// resolvable; define the one typedef we need instead.
typedef signed char int8_t;

struct __align__(2) BlockQ8_0 {
    half d;
    int8_t qs[32];
};

// Host-to-f32 load for the quantize input, one overload per dtype
// `quantize_entry_point` admits (issue #106(b)). Same algorithm CPU applies:
// max-abs scale / 127, round, store int8 + half scale.
__device__ __forceinline__ float quantize_to_f32(const float* p) { return *p; }
__device__ __forceinline__ float quantize_to_f32(const double* p) { return (float)*p; }
__device__ __forceinline__ float quantize_to_f32(const half* p) { return __half2float(*p); }
__device__ __forceinline__ float quantize_to_f32(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

// Shared quantize body: one block of 32 elements. `extern "C"` entries below
// each call this with their typed pointer - they are host-launched kernels,
// not nested launches (same pattern as `matmul_impl`).
template <typename T>
__device__ void quantize_q8_0_block(
    const T* __restrict__ chunk,
    BlockQ8_0* __restrict__ out_block
) {
    float max_abs = 0.0f;
    for (int i = 0; i < 32; i++) {
        float val = quantize_to_f32(&chunk[i]);
        float abs_val = val >= 0.0f ? val : -val;
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    float d = max_abs / 127.0f;
    float inv_d = d == 0.0f ? 0.0f : 1.0f / d;

    out_block->d = __float2half(d);

    for (int i = 0; i < 32; i++) {
        float q = roundf(quantize_to_f32(&chunk[i]) * inv_d);
        out_block->qs[i] = (int8_t)q;
    }
}

extern "C" __global__ void quantize_q8_0(
    const float* __restrict__ inp,
    BlockQ8_0* __restrict__ out,
    const int num_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    quantize_q8_0_block(inp + (idx * 32), out + idx);
}

extern "C" __global__ void quantize_q8_0_f64(
    const double* __restrict__ inp,
    BlockQ8_0* __restrict__ out,
    const int num_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    quantize_q8_0_block(inp + (idx * 32), out + idx);
}

extern "C" __global__ void quantize_q8_0_f16(
    const half* __restrict__ inp,
    BlockQ8_0* __restrict__ out,
    const int num_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    quantize_q8_0_block(inp + (idx * 32), out + idx);
}

extern "C" __global__ void quantize_q8_0_bf16(
    const __nv_bfloat16* __restrict__ inp,
    BlockQ8_0* __restrict__ out,
    const int num_blocks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    quantize_q8_0_block(inp + (idx * 32), out + idx);
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

extern "C" __global__ void quantized_matmul_q8_0(
    const float* __restrict__ A,
    const BlockQ8_0* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    int num_k_blocks = K / 32;

    __shared__ float s_a[32][32];
    __shared__ float s_b[32][32];

    float acc = 0.0f;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        int a_col = kb * 32 + threadIdx.x;
        if (row < M && a_col < K) {
            s_a[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N) {
            BlockQ8_0 b_block = B[kb * N + col];
            float d = __half2float(b_block.d);
            s_b[threadIdx.y][threadIdx.x] = (float)b_block.qs[threadIdx.y] * d;
        } else {
            s_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            acc += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
