// Every item below is `#[cfg(any(feature = "cuda", test))]`; on a
// non-cuda, non-test build this file compiles empty, so this import is
// unused there.
#[allow(unused_imports)]
use super::*;

/// Renders the CUDA source for `layer_norm` or `batch_norm`.
///
/// `layer_norm` reduces over the last axis per row, computed with a
/// Welford accumulator so the pass is numerically stable without a second
/// read of the row. `batch_norm` reads precomputed running statistics per
/// channel rather than reducing at all, which is the inference form; it does
/// not compute batch statistics on the fly.
///
/// `softmax` and `rms_norm` have no case here: both are answered by
/// composing existing pointwise and reduction kernels in
/// `cuda::ops::norm` rather than by a dedicated kernel, and their
/// capability rows say `Composed` rather than `Native` because of it.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn render_cuda_normalization(op_name: &str, dtype: DTypeId) -> Result<RenderedKernel> {
    let scalar = CudaScalarSpec::for_float(dtype, "render_normalization")?;
    #[cfg(feature = "cuda")]
    let policy = {
        let req = PrecisionRequest::new(
            incin_core::shapes::error::OperationKind::Normalization,
            dtype.descriptor(),
            dtype.descriptor(),
            LayoutClass::Contiguous,
            1,
            false,
            MathMode::Fast,
        );
        crate::cuda::backend::native_precision(&req)?
    };
    #[cfg(not(feature = "cuda"))]
    let policy = {
        let compute = if matches!(dtype, DTypeId::F16 | DTypeId::BF16) {
            DTypeId::F32.descriptor()
        } else {
            dtype.descriptor()
        };
        incin_core::exec::ResolvedPrecision::new(
            dtype.descriptor(),
            compute,
            compute,
            dtype.descriptor(),
            incin_core::exec::LossScaling::None,
        )
    };
    debug_assert_eq!(policy.accumulator, policy.compute);
    let entry_point = format!("incin_normalization_{}_{}", scalar.suffix, op_name);
    let key = KernelKey::cuda(
        OperationKind::Normalization,
        KernelFamily::Normalization,
        op_name,
        dtype,
        if op_name == "layer_norm" || op_name == "rms_norm" || op_name == "softmax" {
            LayoutClass::RowWise
        } else {
            LayoutClass::ChannelWise
        },
        if op_name == "layer_norm" {
            KernelAccess::Welford
        } else if op_name == "rms_norm" || op_name == "softmax" {
            KernelAccess::WarpReduction
        } else {
            KernelAccess::Scalar { unroll_width: 1 }
        },
    )?;
    let inverse_std = if dtype == DTypeId::F64 {
        "1.0 / sqrt(variance + (double)eps)"
    } else {
        "rsqrtf(variance + eps)"
    };
    let source = match op_name {
        "layer_norm" => format!(
            r#"
{preamble}
struct IncinWelford {{
    {compute_type} mean;
    {compute_type} m2;
    int count;
}};

__device__ __forceinline__ IncinWelford incin_welford_combine(
    IncinWelford left, IncinWelford right)
{{
    if (right.count == 0) return left;
    if (left.count == 0) return right;
    int count = left.count + right.count;
    {compute_type} delta = right.mean - left.mean;
    {compute_type} right_ratio = ({compute_type})right.count / ({compute_type})count;
    IncinWelford combined;
    combined.mean = left.mean + delta * right_ratio;
    combined.m2 = left.m2 + right.m2 + delta * delta
        * (({compute_type})left.count * ({compute_type})right.count / ({compute_type})count);
    combined.count = count;
    return combined;
}}

extern "C" __global__ void {entry_point}(
    const {storage_type}* __restrict__ input,
    const {storage_type}* __restrict__ gamma,
    const {storage_type}* __restrict__ beta,
    {storage_type}* __restrict__ output,
    float eps,
    int norm_size,
    int has_bias,
    int batch_size,
    int input_offset,
    int gamma_offset,
    int beta_offset,
    {compute_type}* __restrict__ mean_out,
    {compute_type}* __restrict__ rstd_out,
    int save_stats)
{{
    int row = blockIdx.x;
    if (row >= batch_size) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    IncinWelford local = {{({compute_type})0.0, ({compute_type})0.0, 0}};
    int row_start = input_offset + row * norm_size;
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        local.count += 1;
        {compute_type} delta = value - local.mean;
        local.mean += delta / ({compute_type})local.count;
        {compute_type} delta2 = value - local.mean;
        local.m2 += delta * delta2;
    }}
    unsigned int active = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {{
        IncinWelford other;
        other.mean = __shfl_down_sync(active, local.mean, delta);
        other.m2 = __shfl_down_sync(active, local.m2, delta);
        other.count = __shfl_down_sync(active, local.count, delta);
        if (lane + delta < 32) local = incin_welford_combine(local, other);
    }}
    extern __shared__ unsigned char shared_raw[];
    int warp_count = (blockDim.x + 31) >> 5;
    {compute_type}* shared_mean = reinterpret_cast<{compute_type}*>(shared_raw);
    {compute_type}* shared_m2 = shared_mean + warp_count;
    int* shared_count = reinterpret_cast<int*>(shared_m2 + warp_count);
    if (lane == 0) {{
        shared_mean[warp] = local.mean;
        shared_m2[warp] = local.m2;
        shared_count[warp] = local.count;
    }}
    __syncthreads();
    if (warp == 0) {{
        local.mean = lane < warp_count ? shared_mean[lane] : ({compute_type})0.0;
        local.m2 = lane < warp_count ? shared_m2[lane] : ({compute_type})0.0;
        local.count = lane < warp_count ? shared_count[lane] : 0;
        active = __activemask();
        for (int delta = 16; delta > 0; delta >>= 1) {{
            IncinWelford other;
            other.mean = __shfl_down_sync(active, local.mean, delta);
            other.m2 = __shfl_down_sync(active, local.m2, delta);
            other.count = __shfl_down_sync(active, local.count, delta);
            if (lane + delta < 32) local = incin_welford_combine(local, other);
        }}
        if (lane == 0) {{
            shared_mean[0] = local.mean;
            shared_m2[0] = local.m2 / ({compute_type})local.count;
        }}
    }}
    __syncthreads();
    {compute_type} mean = shared_mean[0];
    {compute_type} variance = shared_m2[0];
    {compute_type} inverse_std = {inverse_std};
    // Saved for backward: the recipe must replay these exact statistics,
    // not recompute them, so `save_stats` distinguishes a training forward
    // from an inference one. The pointers are always valid -- the launcher
    // substitutes scratch when it has nowhere to keep the values -- and the
    // flag decides whether anything is written.
    if (save_stats && tid == 0) {{
        mean_out[row] = mean;
        rstd_out[row] = inverse_std;
    }}
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        {compute_type} scale = {load_prefix}gamma[gamma_offset + i]{load_suffix};
        {compute_type} shift = has_bias
            ? {load_prefix}beta[beta_offset + i]{load_suffix}
            : ({compute_type})0.0;
        {compute_type} normalized = (value - mean) * inverse_std;
        output[row * norm_size + i] = {store_prefix}(normalized * scale + shift){store_suffix};
    }}
}}

// Fused layer-norm backward: one row per block, like the forward above.
//
// Given the upstream gradient `grad_output` and the forward's saved
// per-row statistics (`mean`, `rstd`), this produces the input gradient
// plus the weight and bias gradients in a single launch. With
// `y = (x - mean) * rstd` and `gw = grad_output * gamma` written per
// element, the weight must enter *before* the means are taken:
//
//   bias_grad[j]   = sum over rows of grad_output
//   weight_grad[j] = sum over rows of grad_output * y
//   input_grad     = rstd * (gw - mean(gw) - y * mean(gw * y))
//
// Averaging `grad_output` first and multiplying by `gamma` after is the
// same only for uniform weight; everywhere else it is wrong, and wrong in
// a way that still passes a uniform-gradient smoke test, so the parity
// tests below pin the non-uniform case.
//
// The per-row sums need one block-wide reduction; the per-column weight and
// bias sums accumulate across rows with `atomicAdd`, which is how the
// reference implementations do it. That makes the column sums order-dependent
// at the last ulp across runs -- compared against the CPU reference with a
// tolerance, never bit-exact. The bias accumulation is skipped unless the
// forward ran with one; its buffer is always a valid allocation regardless,
// so no launch ever passes a null device pointer.
extern "C" __global__ void {entry_point}_backward(
    const {storage_type}* __restrict__ grad_output,
    const {storage_type}* __restrict__ input,
    const {storage_type}* __restrict__ gamma,
    const {compute_type}* __restrict__ mean,
    const {compute_type}* __restrict__ rstd,
    {storage_type}* __restrict__ grad_input,
    {compute_type}* __restrict__ grad_gamma,
    {compute_type}* __restrict__ grad_beta,
    int norm_size,
    int batch_size,
    int has_bias,
    int grad_output_offset,
    int input_offset,
    int gamma_offset)
{{
    int row = blockIdx.x;
    if (row >= batch_size) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int goff = grad_output_offset + row * norm_size;
    int row_start = input_offset + row * norm_size;
    {compute_type} m = mean[row];
    {compute_type} s = rstd[row];
    {compute_type} sum_gw = ({compute_type})0.0;
    {compute_type} sum_gwy = ({compute_type})0.0;
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} g = {load_prefix}grad_output[goff + i]{load_suffix};
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        {compute_type} y = (value - m) * s;
        {compute_type} w = {load_prefix}gamma[gamma_offset + i]{load_suffix};
        {compute_type} gw = g * w;
        sum_gw += gw;
        sum_gwy += gw * y;
    }}
    unsigned int active = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {{
        sum_gw += __shfl_down_sync(active, sum_gw, delta);
        sum_gwy += __shfl_down_sync(active, sum_gwy, delta);
    }}
    extern __shared__ unsigned char shared_raw[];
    int warp_count = (blockDim.x + 31) >> 5;
    {compute_type}* shared_sums = reinterpret_cast<{compute_type}*>(shared_raw);
    if (lane == 0) {{
        shared_sums[warp * 2] = sum_gw;
        shared_sums[warp * 2 + 1] = sum_gwy;
    }}
    __syncthreads();
    if (warp == 0) {{
        sum_gw = ({compute_type})0.0;
        sum_gwy = ({compute_type})0.0;
        for (int w = lane; w < warp_count; w += 32) {{
            sum_gw += shared_sums[w * 2];
            sum_gwy += shared_sums[w * 2 + 1];
        }}
        active = __activemask();
        for (int delta = 16; delta > 0; delta >>= 1) {{
            sum_gw += __shfl_down_sync(active, sum_gw, delta);
            sum_gwy += __shfl_down_sync(active, sum_gwy, delta);
        }}
        if (lane == 0) {{
            shared_sums[0] = sum_gw;
            shared_sums[1] = sum_gwy;
        }}
    }}
    __syncthreads();
    {compute_type} mean_gw = shared_sums[0] / ({compute_type})norm_size;
    {compute_type} mean_gwy = shared_sums[1] / ({compute_type})norm_size;
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} g = {load_prefix}grad_output[goff + i]{load_suffix};
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        {compute_type} y = (value - m) * s;
        {compute_type} w = {load_prefix}gamma[gamma_offset + i]{load_suffix};
        {compute_type} dx = s * (g * w - mean_gw - y * mean_gwy);
        grad_input[row * norm_size + i] = {store_prefix}dx{store_suffix};
        if (has_bias) {{
            atomicAdd(&grad_beta[i], g);
        }}
        atomicAdd(&grad_gamma[i], g * y);
    }}
}}
"#,
            preamble = scalar.preamble,
            compute_type = scalar.compute_type,
            storage_type = scalar.storage_type,
            load_prefix = scalar.load_prefix,
            load_suffix = scalar.load_suffix,
            store_prefix = scalar.store_prefix,
            store_suffix = scalar.store_suffix,
        ),
        "batch_norm" => format!(
            r#"
{preamble}
extern "C" __global__ void {entry_point}(
    const {storage_type}* __restrict__ input,
    const {storage_type}* __restrict__ weight,
    const {storage_type}* __restrict__ bias,
    const {storage_type}* __restrict__ running_mean,
    const {storage_type}* __restrict__ running_variance,
    {storage_type}* __restrict__ output,
    float eps,
    int num_channels,
    int spatial_size,
    int total_elements,
    int has_weight,
    int has_bias,
    int has_running_mean,
    int has_running_variance,
    int input_offset,
    int weight_offset,
    int bias_offset,
    int mean_offset,
    int variance_offset)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int channel = (idx / spatial_size) % num_channels;
    {compute_type} mean = has_running_mean
        ? {load_prefix}running_mean[mean_offset + channel]{load_suffix}
        : ({compute_type})0.0;
    {compute_type} variance = has_running_variance
        ? {load_prefix}running_variance[variance_offset + channel]{load_suffix}
        : ({compute_type})1.0;
    {compute_type} scale = has_weight
        ? {load_prefix}weight[weight_offset + channel]{load_suffix}
        : ({compute_type})1.0;
    {compute_type} shift = has_bias
        ? {load_prefix}bias[bias_offset + channel]{load_suffix}
        : ({compute_type})0.0;
    {compute_type} value = {load_prefix}input[input_offset + idx]{load_suffix};
    {compute_type} inverse_std = {inverse_std};
    {compute_type} normalized = (value - mean) * inverse_std;
    output[idx] = {store_prefix}(normalized * scale + shift){store_suffix};
}}
"#,
            preamble = scalar.preamble,
            compute_type = scalar.compute_type,
            storage_type = scalar.storage_type,
            load_prefix = scalar.load_prefix,
            load_suffix = scalar.load_suffix,
            store_prefix = scalar.store_prefix,
            store_suffix = scalar.store_suffix,
        ),
        "rms_norm" => format!(
            r#"
{preamble}
extern "C" __global__ void {entry_point}(
    const {storage_type}* __restrict__ input,
    const {storage_type}* __restrict__ weight,
    {storage_type}* __restrict__ output,
    float eps,
    int norm_size,
    int batch_size,
    int input_offset,
    int weight_offset,
    {compute_type}* __restrict__ inv_rms_out,
    int save_norm)
{{
    int row = blockIdx.x;
    if (row >= batch_size) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    
    {compute_type} local_sum_sq = ({compute_type})0.0;
    int row_start = input_offset + row * norm_size;
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        local_sum_sq += value * value;
    }}
    
    unsigned int active = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {{
        local_sum_sq += __shfl_down_sync(active, local_sum_sq, delta);
    }}
    
    extern __shared__ unsigned char shared_raw[];
    int warp_count = (blockDim.x + 31) >> 5;
    {compute_type}* shared_sum = reinterpret_cast<{compute_type}*>(shared_raw);
    if (lane == 0) {{
        shared_sum[warp] = local_sum_sq;
    }}
    __syncthreads();
    
    if (warp == 0) {{
        local_sum_sq = lane < warp_count ? shared_sum[lane] : ({compute_type})0.0;
        active = __activemask();
        for (int delta = 16; delta > 0; delta >>= 1) {{
            local_sum_sq += __shfl_down_sync(active, local_sum_sq, delta);
        }}
        if (lane == 0) {{
            {compute_type} mean_sq = local_sum_sq / ({compute_type})norm_size;
            shared_sum[0] = {inverse_rms};
        }}
    }}
    __syncthreads();
    
    {compute_type} inv_rms = shared_sum[0];
    // Saved for backward like layer_norm's statistics: the recipe replays
    // this exact factor rather than recomputing the mean of squares. The
    // pointers stay valid in every launch (scratch stands in when nothing
    // records) and the flag decides whether anything is written.
    if (save_norm && tid == 0) {{
        inv_rms_out[row] = inv_rms;
    }}
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        {compute_type} gamma = {load_prefix}weight[weight_offset + i]{load_suffix};
        output[row * norm_size + i] = {store_prefix}(value * inv_rms * gamma){store_suffix};
    }}
}}
"#,
            preamble = scalar.preamble,
            compute_type = scalar.compute_type,
            storage_type = scalar.storage_type,
            load_prefix = scalar.load_prefix,
            load_suffix = scalar.load_suffix,
            store_prefix = scalar.store_prefix,
            store_suffix = scalar.store_suffix,
            inverse_rms = if dtype == DTypeId::F64 {
                "1.0 / sqrt(mean_sq + (double)eps)"
            } else {
                "rsqrtf(mean_sq + eps)"
            },
        ),
        "softmax" => format!(
            r#"
{preamble}
extern "C" __global__ void {entry_point}(
    const {storage_type}* __restrict__ input,
    {storage_type}* __restrict__ output,
    int norm_size,
    int batch_size,
    int input_offset)
{{
    int row = blockIdx.x;
    if (row >= batch_size) return;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row_start = input_offset + row * norm_size;
    
    // 1. Local max reduction
    {compute_type} local_max = ({compute_type})-1e38;
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        if (value > local_max) local_max = value;
    }}
    unsigned int active = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {{
        {compute_type} other = __shfl_down_sync(active, local_max, delta);
        if (other > local_max) local_max = other;
    }}
    
    extern __shared__ unsigned char shared_raw[];
    int warp_count = (blockDim.x + 31) >> 5;
    {compute_type}* shared_scratch = reinterpret_cast<{compute_type}*>(shared_raw);
    if (lane == 0) {{
        shared_scratch[warp] = local_max;
    }}
    __syncthreads();
    
    if (warp == 0) {{
        local_max = lane < warp_count ? shared_scratch[lane] : ({compute_type})-1e38;
        active = __activemask();
        for (int delta = 16; delta > 0; delta >>= 1) {{
            {compute_type} other = __shfl_down_sync(active, local_max, delta);
            if (other > local_max) local_max = other;
        }}
        if (lane == 0) {{
            shared_scratch[0] = local_max;
        }}
    }}
    __syncthreads();
    {compute_type} row_max = shared_scratch[0];
    
    // 2. Sum of exponentials
    {compute_type} local_sum_exp = ({compute_type})0.0;
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        local_sum_exp += {exp_func}(value - row_max);
    }}
    active = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {{
        local_sum_exp += __shfl_down_sync(active, local_sum_exp, delta);
    }}
    if (lane == 0) {{
        shared_scratch[warp] = local_sum_exp;
    }}
    __syncthreads();
    
    if (warp == 0) {{
        local_sum_exp = lane < warp_count ? shared_scratch[lane] : ({compute_type})0.0;
        active = __activemask();
        for (int delta = 16; delta > 0; delta >>= 1) {{
            local_sum_exp += __shfl_down_sync(active, local_sum_exp, delta);
        }}
        if (lane == 0) {{
            shared_scratch[0] = local_sum_exp;
        }}
    }}
    __syncthreads();
    {compute_type} row_sum_exp = shared_scratch[0];
    {compute_type} inv_sum = row_sum_exp > ({compute_type})0.0 ? (({compute_type})1.0 / row_sum_exp) : ({compute_type})0.0;
    
    // 3. Write normalized probabilities
    for (int i = tid; i < norm_size; i += blockDim.x) {{
        {compute_type} value = {load_prefix}input[row_start + i]{load_suffix};
        {compute_type} prob = {exp_func}(value - row_max) * inv_sum;
        output[row * norm_size + i] = {store_prefix}prob{store_suffix};
    }}
}}
"#,
            preamble = scalar.preamble,
            compute_type = scalar.compute_type,
            storage_type = scalar.storage_type,
            load_prefix = scalar.load_prefix,
            load_suffix = scalar.load_suffix,
            store_prefix = scalar.store_prefix,
            store_suffix = scalar.store_suffix,
            exp_func = if dtype == DTypeId::F64 { "exp" } else { "expf" },
        ),
        _ => {
            return Err(Error::Msg(format!(
                "unsupported CUDA normalization operation {op_name:?}"
            )));
        }
    };
    Ok(RenderedKernel {
        entry_point,
        cache_key: source_scoped_cache_id(&key, &source),
        source,
        dtype,
        element_size: scalar.element_size,
        unroll_width: 1,
        vector_width: 1,
        key,
    })
}
