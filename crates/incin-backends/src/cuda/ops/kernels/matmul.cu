#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// Element conversion between each storage type and the type its tiles
// accumulate in. The half types keep their operands as f16/bf16 but
// accumulate in float - the same storage/compute split `native_precision`
// reports for them - while f32 and f64 accumulate in their own type.
template <typename In, typename Acc>
struct MatmulCast;

template <>
struct MatmulCast<float, float> {
    static __device__ __forceinline__ float load(float v) { return v; }
    static __device__ __forceinline__ float store(float v) { return v; }
};

template <>
struct MatmulCast<double, double> {
    static __device__ __forceinline__ double load(double v) { return v; }
    static __device__ __forceinline__ double store(double v) { return v; }
};

template <>
struct MatmulCast<__half, float> {
    static __device__ __forceinline__ float load(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half store(float v) { return __float2half(v); }
};

template <>
struct MatmulCast<__nv_bfloat16, float> {
    static __device__ __forceinline__ float load(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __device__ __forceinline__ __nv_bfloat16 store(float v) { return __float2bfloat16(v); }
};

// The tiled shared-memory GEMM body (thread block: 16x16 = 256 threads;
// each thread computes an 8x8 tile of C). Shared tiles and the per-thread
// tile live in `Acc`: operands convert on load, the finished tile converts
// on store. Shared memory is passed in rather than declared here so each
// exported entry point owns its `__shared__` arrays at kernel scope.
template <typename In, typename Acc>
__device__ void matmul_impl(
    const In* A, const In* B, In* C,
    int M, int K, int N,
    Acc* sA, Acc* sB
) {
    typedef MatmulCast<In, Acc> Cast;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid = ty * blockDim.x + tx;

    // The thread's row and column within the shared memory load for A
    const int rowA = tid / BK;
    const int colA = tid % BK;

    // The thread's row and column within the shared memory load for B
    const int rowB = tid / BN;
    const int colB = tid % BN;

    // Registers for thread computation
    Acc rC[TM * TN];
    for (int i = 0; i < TM * TN; ++i) {
        rC[i] = (Acc)0;
    }

    // Loop over K dimension in chunks of BK
    for (int k = 0; k < K; k += BK) {
        // Load A into shared memory (each thread loads 4 elements)
        for (int i = 0; i < BM; i += (blockDim.x * blockDim.y) / BK) {
            int g_row = by * BM + rowA + i;
            int g_col = k + colA;
            if (g_row < M && g_col < K) {
                sA[(rowA + i) * BK + colA] = Cast::load(A[g_row * K + g_col]);
            } else {
                sA[(rowA + i) * BK + colA] = (Acc)0;
            }
        }

        // Load B into shared memory (each thread loads 4 elements)
        for (int i = 0; i < BK; i += (blockDim.x * blockDim.y) / BN) {
            int g_row = k + rowB + i;
            int g_col = bx * BN + colB;
            if (g_row < K && g_col < N) {
                sB[(rowB + i) * BN + colB] = Cast::load(B[g_row * N + g_col]);
            } else {
                sB[(rowB + i) * BN + colB] = (Acc)0;
            }
        }

        __syncthreads();

        // Compute thread tile
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load from shared memory to registers
            Acc regA[TM];
            Acc regB[TN];

            for (int i = 0; i < TM; ++i) {
                regA[i] = sA[(ty * TM + i) * BK + dotIdx];
            }
            for (int i = 0; i < TN; ++i) {
                regB[i] = sB[dotIdx * BN + (tx * TN + i)];
            }

            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    rC[i * TN + j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // Write back to C
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int g_row = by * BM + ty * TM + i;
            int g_col = bx * BN + tx * TN + j;
            if (g_row < M && g_col < N) {
                C[g_row * N + g_col] = Cast::store(rC[i * TN + j]);
            }
        }
    }
}

// Exported entry points, one per storage dtype the capability row admits.
// `matmul` stays the f32 name the launcher has always loaded; the
// suffixed forms follow `embedding.cu`'s convention. The launcher selects
// by operand dtype and refuses anything without an entry here.

extern "C" __global__ void matmul(
    const float* A, const float* B, float* C,
    int M, int K, int N
) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    matmul_impl<float, float>(A, B, C, M, K, N, sA, sB);
}

extern "C" __global__ void matmul_f64(
    const double* A, const double* B, double* C,
    int M, int K, int N
) {
    __shared__ double sA[BM * BK];
    __shared__ double sB[BK * BN];
    matmul_impl<double, double>(A, B, C, M, K, N, sA, sB);
}

extern "C" __global__ void matmul_f16(
    const __half* A, const __half* B, __half* C,
    int M, int K, int N
) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    matmul_impl<__half, float>(A, B, C, M, K, N, sA, sB);
}

extern "C" __global__ void matmul_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int K, int N
) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    matmul_impl<__nv_bfloat16, float>(A, B, C, M, K, N, sA, sB);
}

// Batched body (issue #85): one launch serves the whole batch. The slice
// index is `blockIdx.z` - so the grid's z extent caps a batch at 65535
// slices, and the launcher refuses larger batches rather than wrapping -
// and each operand's distance between consecutive slices is an explicit
// element stride. A stride of 0 reads the same matrix for every slice,
// which is exactly the batch-broadcast case, so a shared batch axis never
// needs a materialized copy. Every slice reuses `matmul_impl` unchanged,
// so a batched slice accumulates bit-for-bit like the unbatched kernel.
template <typename In, typename Acc>
__device__ void matmul_batched_impl(
    const In* A, const In* B, In* C,
    int M, int K, int N,
    long long strideA, long long strideB, long long strideC,
    Acc* sA, Acc* sB
) {
    const long long batch = (long long)blockIdx.z;
    matmul_impl<In, Acc>(
        A + batch * strideA,
        B + batch * strideB,
        C + batch * strideC,
        M, K, N, sA, sB
    );
}

// Batched entry points, one per storage dtype the unbatched kernel exports.
// The launcher selects by operand dtype through the same fail-closed mapping
// as the unbatched names: it launches exactly the name it selected.

extern "C" __global__ void matmul_batched(
    const float* A, const float* B, float* C,
    int M, int K, int N,
    long long strideA, long long strideB, long long strideC
) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    matmul_batched_impl<float, float>(
        A, B, C, M, K, N, strideA, strideB, strideC, sA, sB
    );
}

extern "C" __global__ void matmul_batched_f64(
    const double* A, const double* B, double* C,
    int M, int K, int N,
    long long strideA, long long strideB, long long strideC
) {
    __shared__ double sA[BM * BK];
    __shared__ double sB[BK * BN];
    matmul_batched_impl<double, double>(
        A, B, C, M, K, N, strideA, strideB, strideC, sA, sB
    );
}

extern "C" __global__ void matmul_batched_f16(
    const __half* A, const __half* B, __half* C,
    int M, int K, int N,
    long long strideA, long long strideB, long long strideC
) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    matmul_batched_impl<__half, float>(
        A, B, C, M, K, N, strideA, strideB, strideC, sA, sB
    );
}

extern "C" __global__ void matmul_batched_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int K, int N,
    long long strideA, long long strideB, long long strideC
) {
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];
    matmul_batched_impl<__nv_bfloat16, float>(
        A, B, C, M, K, N, strideA, strideB, strideC, sA, sB
    );
}
