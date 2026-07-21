#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

extern "C" __global__ void matmul(
    const float* A, const float* B, float* C, 
    int M, int K, int N
) {
    // Thread block: 16x16 (256 threads)
    // Each thread computes an 8x8 tile in C
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

    // Shared memory
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    // Registers for thread computation
    float rC[TM * TN] = {0.0f};

    // Loop over K dimension in chunks of BK
    for (int k = 0; k < K; k += BK) {
        // Load A into shared memory (each thread loads 4 elements)
        for (int i = 0; i < BM; i += (blockDim.x * blockDim.y) / BK) { // 128 / (256/8 = 32) -> 4 iterations
            int g_row = by * BM + rowA + i;
            int g_col = k + colA;
            if (g_row < M && g_col < K) {
                sA[(rowA + i) * BK + colA] = A[g_row * K + g_col];
            } else {
                sA[(rowA + i) * BK + colA] = 0.0f;
            }
        }

        // Load B into shared memory (each thread loads 4 elements)
        for (int i = 0; i < BK; i += (blockDim.x * blockDim.y) / BN) { // 8 / (256/128 = 2) -> 4 iterations
            int g_row = k + rowB + i;
            int g_col = bx * BN + colB;
            if (g_row < K && g_col < N) {
                sB[(rowB + i) * BN + colB] = B[g_row * N + g_col];
            } else {
                sB[(rowB + i) * BN + colB] = 0.0f;
            }
        }

        __syncthreads();

        // Compute thread tile
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // Load from shared memory to registers
            float regA[TM];
            float regB[TN];
            
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
                C[g_row * N + g_col] = rC[i * TN + j];
            }
        }
    }
}
