extern "C" __global__ void fused_matmul_swiglu(
    const float* A, const float* B, float* C, 
    int M, int K, int N
) {
    // Basic tiled matrix multiplication with SwiGLU activation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        
        // SwiGLU activation: x * sigmoid(beta * x)
        // Note: typically SwiGLU takes two inputs, but for a simple fused
        // kernel we apply SiLU (Swish with beta=1) to the matmul output.
        // A true SwiGLU would have W_gate and W_up.
        float sig = 1.0f / (1.0f + expf(-sum));
        C[row * N + col] = sum * sig;
    }
}
