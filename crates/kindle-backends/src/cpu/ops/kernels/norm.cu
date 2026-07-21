extern "C" {
    __device__ float rsqrtf(float);
}

// ----------------------------------------------------------------------------
// layer_norm
// ----------------------------------------------------------------------------
extern "C" __global__ void layer_norm(
    const float* __restrict__ inp,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    const float eps,
    const int norm_size,
    const int has_bias,
    const int batch_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int tid = threadIdx.x;
    const float* x = inp + batch_idx * norm_size;
    float* y = out + batch_idx * norm_size;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    // Step 1: Mean
    float local_sum = 0.0f;
    for (int i = tid; i < norm_size; i += blockDim.x) {
        local_sum += x[i];
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float mean = sdata[0] / norm_size;
    __syncthreads();
    
    // Step 2: Variance
    float local_var_sum = 0.0f;
    for (int i = tid; i < norm_size; i += blockDim.x) {
        float diff = x[i] - mean;
        local_var_sum += diff * diff;
    }
    
    sdata[tid] = local_var_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float var = sdata[0] / norm_size;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();
    
    // Step 3: Normalize and scale/shift
    for (int i = tid; i < norm_size; i += blockDim.x) {
        float norm_val = (x[i] - mean) * inv_std;
        float b = has_bias ? beta[i] : 0.0f;
        y[i] = norm_val * gamma[i] + b;
    }
}

// ----------------------------------------------------------------------------
// batch_norm (inference)
// ----------------------------------------------------------------------------
extern "C" __global__ void batch_norm(
    const float* __restrict__ inp,
    const float* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ rm,
    const float* __restrict__ rv,
    float* __restrict__ out,
    const float eps,
    const int num_channels,
    const int spatial_size,
    const int total_elements,
    const int has_w,
    const int has_b,
    const int has_rm,
    const int has_rv
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // input shape: [N, C, spatial_size] where spatial_size is e.g. H*W
    // So c = (idx / spatial_size) % num_channels
    int c = (idx / spatial_size) % num_channels;

    float mean = has_rm ? rm[c] : 0.0f;
    float var = has_rv ? rv[c] : 1.0f;
    float weight = has_w ? w[c] : 1.0f;
    float bias = has_b ? b[c] : 0.0f;

    float inv_std = rsqrtf(var + eps);
    float norm_val = (inp[idx] - mean) * inv_std;
    
    out[idx] = norm_val * weight + bias;
}
