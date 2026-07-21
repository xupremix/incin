extern "C" __global__ void sum_axis(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ in_shape,
    const int* __restrict__ in_strides,
    const int* __restrict__ out_shape,
    const int* __restrict__ out_strides,
    int in_offset,
    int out_offset,
    int reduce_axis,
    int reduce_dim_size,
    int ndim,
    int out_numel)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_numel) return;

    int temp = out_idx;
    int out_flat = out_offset;
    int base_in_flat = in_offset;

    for (int i = ndim - 1; i >= 0; i--) {
        int dim_idx = temp % out_shape[i];
        temp /= out_shape[i];
        out_flat += dim_idx * out_strides[i];
        if (i != reduce_axis) {
            base_in_flat += dim_idx * in_strides[i];
        }
    }

    float acc = 0.0f;
    for (int i = 0; i < reduce_dim_size; i++) {
        int in_flat = base_in_flat + i * in_strides[reduce_axis];
        acc += input[in_flat];
    }
    output[out_flat] = acc;
}

extern "C" __global__ void max_axis(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ in_shape,
    const int* __restrict__ in_strides,
    const int* __restrict__ out_shape,
    const int* __restrict__ out_strides,
    int in_offset,
    int out_offset,
    int reduce_axis,
    int reduce_dim_size,
    int ndim,
    int out_numel)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_numel) return;

    int temp = out_idx;
    int out_flat = out_offset;
    int base_in_flat = in_offset;

    for (int i = ndim - 1; i >= 0; i--) {
        int dim_idx = temp % out_shape[i];
        temp /= out_shape[i];
        out_flat += dim_idx * out_strides[i];
        if (i != reduce_axis) {
            base_in_flat += dim_idx * in_strides[i];
        }
    }

    float best = -1.0f / 0.0f;
    for (int i = 0; i < reduce_dim_size; i++) {
        int in_flat = base_in_flat + i * in_strides[reduce_axis];
        float val = input[in_flat];
        if (val > best) best = val;
    }
    output[out_flat] = best;
}

extern "C" __global__ void min_axis(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ in_shape,
    const int* __restrict__ in_strides,
    const int* __restrict__ out_shape,
    const int* __restrict__ out_strides,
    int in_offset,
    int out_offset,
    int reduce_axis,
    int reduce_dim_size,
    int ndim,
    int out_numel)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_numel) return;

    int temp = out_idx;
    int out_flat = out_offset;
    int base_in_flat = in_offset;

    for (int i = ndim - 1; i >= 0; i--) {
        int dim_idx = temp % out_shape[i];
        temp /= out_shape[i];
        out_flat += dim_idx * out_strides[i];
        if (i != reduce_axis) {
            base_in_flat += dim_idx * in_strides[i];
        }
    }

    float best = 1.0f / 0.0f;
    for (int i = 0; i < reduce_dim_size; i++) {
        int in_flat = base_in_flat + i * in_strides[reduce_axis];
        float val = input[in_flat];
        if (val < best) best = val;
    }
    output[out_flat] = best;
}

extern "C" __global__ void max_axis_with_indices(
    const float* __restrict__ input,
    float* __restrict__ out_vals,
    unsigned int* __restrict__ out_indices,
    const int* __restrict__ in_shape,
    const int* __restrict__ in_strides,
    const int* __restrict__ out_shape,
    const int* __restrict__ out_strides,
    int in_offset,
    int out_offset,
    int reduce_axis,
    int reduce_dim_size,
    int ndim,
    int out_numel)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_numel) return;

    int temp = out_idx;
    int out_flat = out_offset;
    int base_in_flat = in_offset;

    for (int i = ndim - 1; i >= 0; i--) {
        int dim_idx = temp % out_shape[i];
        temp /= out_shape[i];
        out_flat += dim_idx * out_strides[i];
        if (i != reduce_axis) {
            base_in_flat += dim_idx * in_strides[i];
        }
    }

    float best_val = -1.0f / 0.0f;
    unsigned int best_idx = 0;
    for (int i = 0; i < reduce_dim_size; i++) {
        int in_flat = base_in_flat + i * in_strides[reduce_axis];
        float val = input[in_flat];
        if (val > best_val) {
            best_val = val;
            best_idx = (unsigned int)in_flat;
        }
    }
    out_vals[out_flat] = best_val;
    out_indices[out_flat] = best_idx;
}

extern "C" __global__ void min_axis_with_indices(
    const float* __restrict__ input,
    float* __restrict__ out_vals,
    unsigned int* __restrict__ out_indices,
    const int* __restrict__ in_shape,
    const int* __restrict__ in_strides,
    const int* __restrict__ out_shape,
    const int* __restrict__ out_strides,
    int in_offset,
    int out_offset,
    int reduce_axis,
    int reduce_dim_size,
    int ndim,
    int out_numel)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_numel) return;

    int temp = out_idx;
    int out_flat = out_offset;
    int base_in_flat = in_offset;

    for (int i = ndim - 1; i >= 0; i--) {
        int dim_idx = temp % out_shape[i];
        temp /= out_shape[i];
        out_flat += dim_idx * out_strides[i];
        if (i != reduce_axis) {
            base_in_flat += dim_idx * in_strides[i];
        }
    }

    float best_val = 1.0f / 0.0f;
    unsigned int best_idx = 0;
    for (int i = 0; i < reduce_dim_size; i++) {
        int in_flat = base_in_flat + i * in_strides[reduce_axis];
        float val = input[in_flat];
        if (val < best_val) {
            best_val = val;
            best_idx = (unsigned int)in_flat;
        }
    }
    out_vals[out_flat] = best_val;
    out_indices[out_flat] = best_idx;
}
