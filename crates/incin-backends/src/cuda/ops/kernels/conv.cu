// NVRTC has no default header search path, so `int64_t` isn't available
// without an explicit typedef (unlike `size_t`, which is compiler builtin).
typedef long long int64_t;

extern "C" __global__ void im2col_2d(
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
    const size_t pad_w,
    const size_t dilation_h,
    const size_t dilation_w)
{
    const size_t total_elements = batch_size * channels * h_out * w_out;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // Decode thread index back into (b, c, oh, ow)
    size_t ow = idx % w_out;
    size_t oh = (idx / w_out) % h_out;
    size_t c = (idx / (w_out * h_out)) % channels;
    size_t b = idx / (w_out * h_out * channels);

    size_t in_batch_offset = b * channels * h_in * w_in;
    size_t in_channel_offset = c * h_in * w_in;
    size_t out_batch_offset = b * (channels * k_h * k_w) * (h_out * w_out);
    size_t out_pixel = oh * w_out + ow;

    for (size_t kh = 0; kh < k_h; ++kh) {
        for (size_t kw = 0; kw < k_w; ++kw) {
            int64_t ih = (int64_t)(oh * stride_h + kh * dilation_h) - (int64_t)pad_h;
            int64_t iw = (int64_t)(ow * stride_w + kw * dilation_w) - (int64_t)pad_w;

            size_t out_channel_k = c * k_h * k_w + kh * k_w + kw;
            size_t out_idx = out_batch_offset + out_channel_k * (h_out * w_out) + out_pixel;

            if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                size_t in_idx = in_batch_offset + in_channel_offset + ih * w_in + iw;
                output[out_idx] = input[in_idx];
            } else {
                output[out_idx] = 0.0f;
            }
        }
    }
}

extern "C" __global__ void col2im_2d(
    const float* __restrict__ col,
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

    size_t out_batch_offset = b * channels * h_in * w_in;
    size_t out_channel_offset = c * h_in * w_in;
    size_t col_batch_offset = b * (channels * k_h * k_w) * (h_out * w_out);
    size_t col_pixel = oh * w_out + ow;

    for (size_t kh = 0; kh < k_h; ++kh) {
        for (size_t kw = 0; kw < k_w; ++kw) {
            int64_t ih = (int64_t)(oh * stride_h + kh * dilation_h) - (int64_t)pad_h;
            int64_t iw = (int64_t)(ow * stride_w + kw * dilation_w) - (int64_t)pad_w;

            if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                size_t col_channel_k = c * k_h * k_w + kh * k_w + kw;
                size_t col_idx = col_batch_offset + col_channel_k * (h_out * w_out) + col_pixel;
                size_t out_idx = out_batch_offset + out_channel_offset + ih * w_in + iw;
                
                atomicAdd(&output[out_idx], col[col_idx]);
            }
        }
    }
}

extern "C" __global__ void im2col_1d(
    const float* __restrict__ input,
    float* __restrict__ output,
    const size_t batch_size,
    const size_t channels,
    const size_t l_in,
    const size_t l_out,
    const size_t k,
    const size_t stride,
    const size_t pad,
    const size_t dilation)
{
    const size_t total_elements = batch_size * channels * l_out;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    size_t o_l = idx % l_out;
    size_t c = (idx / l_out) % channels;
    size_t b = idx / (l_out * channels);

    size_t in_batch_offset = b * channels * l_in;
    size_t in_channel_offset = c * l_in;
    size_t out_batch_offset = b * (channels * k) * l_out;

    for (size_t ki = 0; ki < k; ++ki) {
        int64_t il = (int64_t)(o_l * stride + ki * dilation) - (int64_t)pad;
        size_t out_channel_k = c * k + ki;
        size_t out_idx = out_batch_offset + out_channel_k * l_out + o_l;

        if (il >= 0 && il < l_in) {
            size_t in_idx = in_batch_offset + in_channel_offset + il;
            output[out_idx] = input[in_idx];
        } else {
            output[out_idx] = 0.0f;
        }
    }
}

extern "C" __global__ void col2im_1d(
    const float* __restrict__ col,
    float* __restrict__ output,
    const size_t batch_size,
    const size_t channels,
    const size_t l_in,
    const size_t l_out,
    const size_t k,
    const size_t stride,
    const size_t pad,
    const size_t dilation)
{
    const size_t total_elements = batch_size * channels * l_out;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    size_t o_l = idx % l_out;
    size_t c = (idx / l_out) % channels;
    size_t b = idx / (l_out * channels);

    size_t out_batch_offset = b * channels * l_in;
    size_t out_channel_offset = c * l_in;
    size_t col_batch_offset = b * (channels * k) * l_out;

    for (size_t ki = 0; ki < k; ++ki) {
        int64_t il = (int64_t)(o_l * stride + ki * dilation) - (int64_t)pad;

        if (il >= 0 && il < l_in) {
            size_t col_channel_k = c * k + ki;
            size_t col_idx = col_batch_offset + col_channel_k * l_out + o_l;
            size_t out_idx = out_batch_offset + out_channel_offset + il;
            
            atomicAdd(&output[out_idx], col[col_idx]);
        }
    }
}
