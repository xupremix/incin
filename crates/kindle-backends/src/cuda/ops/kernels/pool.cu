// NVRTC has no default header search path, so fixed-width int typedefs
// aren't available without an explicit definition (unlike `size_t`, which
// is compiler builtin).
typedef unsigned int uint32_t;
typedef long long int64_t;

extern "C" __global__ void max_pool2d_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t* __restrict__ max_indices,
    const size_t batch_size,
    const size_t channels,
    const size_t h_in,
    const size_t w_in,
    const size_t h_out,
    const size_t w_out,
    const size_t k_h,
    const size_t k_w,
    const size_t stride_h,
    const size_t stride_w,
    const size_t pad_h,
    const size_t pad_w,
    const size_t dilation_h,
    const size_t dilation_w)
{
    const size_t total_elements = batch_size * channels * h_out * w_out;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    size_t ow = idx % w_out;
    size_t oh = (idx / w_out) % h_out;
    size_t c = (idx / (w_out * h_out)) % channels;
    size_t b = idx / (w_out * h_out * channels);

    size_t in_batch_offset = b * channels * h_in * w_in;
    size_t in_channel_offset = c * h_in * w_in;

    float max_val = -1e38f; // approx -inf
    size_t max_idx = 0;

    for (size_t kh = 0; kh < k_h; ++kh) {
        for (size_t kw = 0; kw < k_w; ++kw) {
            int64_t ih = (int64_t)(oh * stride_h + kh * dilation_h) - (int64_t)pad_h;
            int64_t iw = (int64_t)(ow * stride_w + kw * dilation_w) - (int64_t)pad_w;

            if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                size_t flat_idx = in_batch_offset + in_channel_offset + ih * w_in + iw;
                float val = input[flat_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = flat_idx;
                }
            }
        }
    }

    output[idx] = max_val;
    max_indices[idx] = (uint32_t)max_idx;
}

extern "C" __global__ void scatter_pool_grad_2d(
    const float* __restrict__ grad_out,
    const uint32_t* __restrict__ max_indices,
    float* __restrict__ grad_in,
    const size_t out_total)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= out_total) return;

    size_t in_idx = max_indices[idx];
    atomicAdd(&grad_in[in_idx], grad_out[idx]);
}

extern "C" __global__ void avg_pool2d_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    const size_t batch_size,
    const size_t channels,
    const size_t h_in,
    const size_t w_in,
    const size_t h_out,
    const size_t w_out,
    const size_t k_h,
    const size_t k_w,
    const size_t stride_h,
    const size_t stride_w,
    const size_t pad_h,
    const size_t pad_w)
{
    const size_t total_elements = batch_size * channels * h_out * w_out;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    size_t ow = idx % w_out;
    size_t oh = (idx / w_out) % h_out;
    size_t c = (idx / (w_out * h_out)) % channels;
    size_t b = idx / (w_out * h_out * channels);

    size_t in_batch_offset = b * channels * h_in * w_in;
    size_t in_channel_offset = c * h_in * w_in;

    float sum = 0.0f;
    float count = (float)(k_h * k_w);

    for (size_t kh = 0; kh < k_h; ++kh) {
        for (size_t kw = 0; kw < k_w; ++kw) {
            int64_t ih = (int64_t)(oh * stride_h + kh) - (int64_t)pad_h;
            int64_t iw = (int64_t)(ow * stride_w + kw) - (int64_t)pad_w;

            if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                size_t flat_idx = in_batch_offset + in_channel_offset + ih * w_in + iw;
                sum += input[flat_idx];
            }
        }
    }

    output[idx] = sum / count;
}

extern "C" __global__ void avg_pool2d_backward(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    const size_t batch_size,
    const size_t channels,
    const size_t h_in,
    const size_t w_in,
    const size_t h_out,
    const size_t w_out,
    const size_t k_h,
    const size_t k_w,
    const size_t stride_h,
    const size_t stride_w,
    const size_t pad_h,
    const size_t pad_w)
{
    const size_t total_elements = batch_size * channels * h_out * w_out;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    size_t ow = idx % w_out;
    size_t oh = (idx / w_out) % h_out;
    size_t c = (idx / (w_out * h_out)) % channels;
    size_t b = idx / (w_out * h_out * channels);

    size_t in_batch_offset = b * channels * h_in * w_in;
    size_t in_channel_offset = c * h_in * w_in;

    float g = grad_out[idx] / (float)(k_h * k_w);

    for (size_t kh = 0; kh < k_h; ++kh) {
        for (size_t kw = 0; kw < k_w; ++kw) {
            int64_t ih = (int64_t)(oh * stride_h + kh) - (int64_t)pad_h;
            int64_t iw = (int64_t)(ow * stride_w + kw) - (int64_t)pad_w;

            if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                size_t in_idx = in_batch_offset + in_channel_offset + ih * w_in + iw;
                atomicAdd(&grad_in[in_idx], g);
            }
        }
    }
}

extern "C" __device__ void adaptive_window_bounds(size_t input_size, size_t output_size, size_t i, size_t* start, size_t* end) {
    *start = (i * input_size) / output_size;
    *end = ((i + 1) * input_size + output_size - 1) / output_size; // div_ceil
}

extern "C" __global__ void adaptive_avg_pool2d_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    const size_t batch_size,
    const size_t channels,
    const size_t h_in,
    const size_t w_in,
    const size_t h_out,
    const size_t w_out)
{
    const size_t total_elements = batch_size * channels * h_out * w_out;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    size_t ow = idx % w_out;
    size_t oh = (idx / w_out) % h_out;
    size_t c = (idx / (w_out * h_out)) % channels;
    size_t b = idx / (w_out * h_out * channels);

    size_t in_batch_offset = b * channels * h_in * w_in;
    size_t in_channel_offset = c * h_in * w_in;

    size_t h_start, h_end, w_start, w_end;
    adaptive_window_bounds(h_in, h_out, oh, &h_start, &h_end);
    adaptive_window_bounds(w_in, w_out, ow, &w_start, &w_end);

    float sum = 0.0f;
    float count = (float)((h_end - h_start) * (w_end - w_start));

    for (size_t ih = h_start; ih < h_end; ++ih) {
        for (size_t iw = w_start; iw < w_end; ++iw) {
            size_t flat_idx = in_batch_offset + in_channel_offset + ih * w_in + iw;
            sum += input[flat_idx];
        }
    }

    output[idx] = sum / count;
}

extern "C" __global__ void adaptive_avg_pool2d_backward(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    const size_t batch_size,
    const size_t channels,
    const size_t h_in,
    const size_t w_in,
    const size_t h_out,
    const size_t w_out)
{
    const size_t total_elements = batch_size * channels * h_out * w_out;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    size_t ow = idx % w_out;
    size_t oh = (idx / w_out) % h_out;
    size_t c = (idx / (w_out * h_out)) % channels;
    size_t b = idx / (w_out * h_out * channels);

    size_t in_batch_offset = b * channels * h_in * w_in;
    size_t in_channel_offset = c * h_in * w_in;

    size_t h_start, h_end, w_start, w_end;
    adaptive_window_bounds(h_in, h_out, oh, &h_start, &h_end);
    adaptive_window_bounds(w_in, w_out, ow, &w_start, &w_end);

    float count = (float)((h_end - h_start) * (w_end - w_start));
    float g = grad_out[idx] / count;

    for (size_t ih = h_start; ih < h_end; ++ih) {
        for (size_t iw = w_start; iw < w_end; ++iw) {
            size_t in_idx = in_batch_offset + in_channel_offset + ih * w_in + iw;
            atomicAdd(&grad_in[in_idx], g);
        }
    }
}
