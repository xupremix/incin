use crate::cuda::storage::{CudaBuffer, CudaStorage};
use crate::cuda::{checked_i32, checked_i32_vec};
use crate::iteration::OperandIteration;
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::exec::PrecisionRequest;
use incin_core::shapes::OperationKind;
use incin_core::tensor::dtype::DTypeId;

fn validate_reduction<'a>(
    storage: &'a CudaStorage,
    axis: usize,
    op_name: &'static str,
) -> Result<(&'a CudaBuffer, usize)> {
    let buffer = &*storage.buffer;
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, "reduction")?;
    if axis >= storage.shape.len() {
        return Err(Error::Msg(format!(
            "CUDA reduction axis {axis} is out of bounds for rank {}",
            storage.shape.len()
        )));
    }
    let reduce_dim_size = storage.shape[axis];
    if reduce_dim_size == 0 && op_name != "sum" {
        return Err(Error::Msg(format!(
            "CUDA {op_name} reduction is undefined for an empty axis"
        )));
    }
    let operand = OperandIteration {
        strides: storage.strides.clone(),
        offset: storage.offset_elements,
    };
    if let Some(max_index) = operand.max_physical_index(&storage.shape)?
        && max_index >= buffer.len
    {
        return Err(Error::Msg(format!(
            "CUDA reduction accesses storage index {max_index}, but buffer length is {}",
            buffer.len
        )));
    }
    Ok((buffer, reduce_dim_size))
}

fn reduction_shapes(shape: &[usize], axis: usize, keepdim: bool) -> (Vec<usize>, Vec<usize>) {
    let mut keepdim_shape = shape.to_vec();
    keepdim_shape[axis] = 1;
    let final_shape = if keepdim {
        keepdim_shape.clone()
    } else {
        let mut squeezed = keepdim_shape.clone();
        squeezed.remove(axis);
        squeezed
    };
    (keepdim_shape, final_shape)
}

fn is_contiguous_last_axis(storage: &CudaStorage, axis: usize) -> bool {
    axis + 1 == storage.shape.len()
        && storage.strides == crate::layout::contiguous_strides(&storage.shape)
}

struct ReductionLaunchSelection {
    candidate: crate::tuning::LaunchCandidate,
    #[cfg(feature = "autotune")]
    tuning_permit: Option<crate::tuning::TuningPermit>,
}

fn reduction_launch_selection(
    context: &cudarc::driver::CudaContext,
    kernel: &crate::kernel::RenderedKernel,
    rows: usize,
    reduction_size: usize,
    contiguous_last_axis: bool,
) -> Result<ReductionLaunchSelection> {
    let candidates = crate::tuning::reduction_candidates(contiguous_last_axis);
    let fallback = crate::tuning::default_reduction_candidate(&candidates)?;
    #[cfg(feature = "autotune")]
    {
        let key = crate::tuning::TuningKey::new(
            crate::tuning::identity::TuningEnvironmentFingerprint::<
                incin_core::tensor::device::Cuda,
            >::from_cuda_context(context)?
            .erase(),
            &kernel.key,
            crate::tuning::WorkloadBucket::reduction(rows, reduction_size),
        );
        match crate::tuning::claim_tuning(key, &candidates)? {
            crate::tuning::TuningDecision::Cached(tuned) => Ok(ReductionLaunchSelection {
                candidate: tuned.candidate,
                tuning_permit: None,
            }),
            crate::tuning::TuningDecision::Measure(permit) => Ok(ReductionLaunchSelection {
                candidate: fallback,
                tuning_permit: Some(permit),
            }),
        }
    }
    #[cfg(not(feature = "autotune"))]
    {
        let _ = (context, kernel, rows, reduction_size);
        Ok(ReductionLaunchSelection {
            candidate: fallback,
        })
    }
}

#[cfg(feature = "cuda")]
fn empirically_select_reduction_candidate<F>(
    stream: &cudarc::driver::CudaStream,
    selection: ReductionLaunchSelection,
    contiguous_last_axis: bool,
    mut launch: F,
) -> Result<crate::tuning::LaunchCandidate>
where
    F: FnMut(crate::tuning::LaunchCandidate) -> Result<()>,
{
    #[cfg(feature = "autotune")]
    if let Some(permit) = selection.tuning_permit {
        let candidates = crate::tuning::reduction_candidates(contiguous_last_axis);
        let mut measurements = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            measurements.push(crate::tuning::measure_cuda_candidate(
                stream,
                candidate,
                || launch(candidate),
            )?);
        }
        return Ok(permit.record(&measurements)?.candidate);
    }
    #[cfg(not(feature = "autotune"))]
    let _ = (stream, contiguous_last_axis, &mut launch);
    Ok(selection.candidate)
}

#[cfg(feature = "cuda")]
fn ensure_reduction_loaded(device_id: usize, kernel: &crate::kernel::RenderedKernel) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, &kernel.cache_key).is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel(
            &kernel.entry_point,
            &kernel.source,
            &kernel.cache_key,
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_reduce_op(
    op_name: &'static str,
    storage: &CudaStorage,
    axis: usize,
    keepdim: bool,
) -> Result<CudaStorage> {
    let (buffer, reduce_dim_size) = validate_reduction(storage, axis, op_name)?;
    let (keepdim_shape, final_shape) = reduction_shapes(&storage.shape, axis, keepdim);
    let out_numel = keepdim_shape.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| Error::Msg("CUDA reduction output element count overflow".into()))
    })?;
    let fast = is_contiguous_last_axis(storage, axis);
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, op_name)?;
    let kernel = crate::kernel::render_cuda_reduction(op_name, builtin_id, false, fast)?;
    if Some(kernel.dtype) != buffer.dtype.builtin_id()
        || kernel.element_size
            != buffer
                .dtype
                .encoding()
                .scalar_bytes()
                .ok_or_else(|| Error::Msg("Invalid scalar bytes".into()))?
    {
        return Err(Error::Msg(
            "CUDA reduction kernel/storage ABI mismatch".into(),
        ));
    }
    let launch_selection =
        reduction_launch_selection(&buffer.device, &kernel, out_numel, reduce_dim_size, fast)?;

    let stream = buffer.device.default_stream();
    let byte_len = crate::bytes::byte_len(kernel.dtype, out_numel, OperationKind::Reduction)?;
    let mut out_buffer = CudaBuffer {
        len: out_numel,
        dtype: buffer.dtype,
        data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|error| {
            Error::Msg(format!(
                "CUDA reduction output allocation failed: {error:?}"
            ))
        })?),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    if out_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), final_shape));
    }

    ensure_reduction_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;
    let in_offset = checked_i32(storage.offset_elements, "input offset")?;
    let reduce_dim = checked_i32(reduce_dim_size, "axis length")?;
    let out_numel_i32 = checked_i32(out_numel, "output element count")?;

    // SAFETY: checked axis, offsets, element counts, and launch dimensions
    // bound the reduction views; out_buffer is freshly and uniquely owned.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data).ok_or_else(|| {
            Error::Msg("fresh CUDA reduction output buffer was unexpectedly shared".into())
        })?;
        use cudarc::driver::PushKernelArg;
        if fast {
            let req = PrecisionRequest::new(
                OperationKind::Reduction,
                buffer.dtype,
                buffer.dtype,
                incin_core::exec::LayoutClass::Contiguous,
                1,
                false,
                incin_core::exec::MathMode::Fast,
            );
            let policy = crate::cuda::backend::native_precision(&req)?;
            let grid = u32::try_from(out_numel).map_err(|_| {
                Error::Msg(format!(
                    "CUDA reduction block count {out_numel} exceeds u32 grid ABI"
                ))
            })?;
            let scalar_bytes = policy
                .accumulator
                .encoding()
                .scalar_bytes()
                .ok_or_else(|| Error::Msg("CUDA reduction invalid accumulator encoding".into()))?;
            let mut launch = |candidate: crate::tuning::LaunchCandidate| -> Result<()> {
                let block_size = u32::from(candidate.block_size);
                let shared_mem_bytes = (usize::from(candidate.block_size) / 32)
                    .checked_mul(scalar_bytes)
                    .and_then(|bytes| u32::try_from(bytes).ok())
                    .ok_or_else(|| {
                        Error::Msg("CUDA reduction shared-memory size overflow".into())
                    })?;
                let config = cudarc::driver::LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes,
                };
                stream
                    .launch_builder(&function)
                    .arg(&*buffer.data)
                    .arg(&mut *out_u8)
                    .arg(&in_offset)
                    .arg(&reduce_dim)
                    .arg(&out_numel_i32)
                    .launch(config)
                    .map(|_| ())
                    .map_err(|error| {
                        Error::Msg(format!("CUDA {op_name} reduction launch failed: {error:?}"))
                    })
            };
            let launch_candidate = empirically_select_reduction_candidate(
                &stream,
                launch_selection,
                true,
                &mut launch,
            )?;
            launch(launch_candidate)
        } else {
            let in_strides = checked_i32_vec(&storage.strides, "input stride")?;
            let out_shape = checked_i32_vec(&keepdim_shape, "output shape")?;
            let out_strides_host = crate::layout::contiguous_strides(&keepdim_shape);
            let out_strides = checked_i32_vec(&out_strides_host, "output stride")?;
            let in_strides_dev = stream.clone_htod(&in_strides).map_err(|error| {
                Error::Msg(format!("CUDA reduction stride upload failed: {error:?}"))
            })?;
            let out_shape_dev = stream.clone_htod(&out_shape).map_err(|error| {
                Error::Msg(format!("CUDA reduction shape upload failed: {error:?}"))
            })?;
            let out_strides_dev = stream.clone_htod(&out_strides).map_err(|error| {
                Error::Msg(format!(
                    "CUDA reduction output-stride upload failed: {error:?}"
                ))
            })?;
            let ndim = checked_i32(storage.shape.len(), "rank")?;
            let work_items = u32::try_from(out_numel).map_err(|_| {
                Error::Msg(format!(
                    "CUDA reduction work-item count {out_numel} exceeds u32 grid ABI"
                ))
            })?;
            let axis_i32 = checked_i32(axis, "axis")?;
            let mut launch = |candidate: crate::tuning::LaunchCandidate| -> Result<()> {
                let block_size = u32::from(candidate.block_size);
                let config = cudarc::driver::LaunchConfig {
                    grid_dim: (work_items.div_ceil(block_size), 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                stream
                    .launch_builder(&function)
                    .arg(&*buffer.data)
                    .arg(&mut *out_u8)
                    .arg(&in_strides_dev)
                    .arg(&out_shape_dev)
                    .arg(&out_strides_dev)
                    .arg(&in_offset)
                    .arg(&0i32)
                    .arg(&axis_i32)
                    .arg(&reduce_dim)
                    .arg(&ndim)
                    .arg(&out_numel_i32)
                    .launch(config)
                    .map(|_| ())
                    .map_err(|error| {
                        Error::Msg(format!("CUDA {op_name} reduction launch failed: {error:?}"))
                    })
            };
            let launch_candidate = empirically_select_reduction_candidate(
                &stream,
                launch_selection,
                false,
                &mut launch,
            )?;
            launch(launch_candidate)
        }?;
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), final_shape))
}

const REDUCE_OPS_SRC: &str = r#"
typedef long long int64_t;

extern "C" __global__ void incin_cuda_argmax_argmin(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int in_offset,
    int reduce_dim_size,
    int out_numel,
    int is_argmin)
{
    int out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;
    int tid = threadIdx.x;
    int row_start = in_offset + out_idx * reduce_dim_size;
    
    float best_val = is_argmin ? 1e38f : -1e38f;
    int64_t best_idx = -1;
    
    for (int i = tid; i < reduce_dim_size; i += blockDim.x) {
        float val = input[row_start + i];
        int64_t idx = (int64_t)i;
        if (best_idx < 0) {
            best_val = val;
            best_idx = idx;
        } else if (is_argmin) {
            if (val < best_val || (val == best_val && idx < best_idx)) {
                best_val = val;
                best_idx = idx;
            }
        } else {
            if (val > best_val || (val == best_val && idx < best_idx)) {
                best_val = val;
                best_idx = idx;
            }
        }
    }
    
    unsigned int active = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {
        float other_val = __shfl_down_sync(active, best_val, delta);
        int64_t other_idx = __shfl_down_sync(active, best_idx, delta);
        if (other_idx >= 0) {
            if (best_idx < 0) {
                best_val = other_val;
                best_idx = other_idx;
            } else if (is_argmin) {
                if (other_val < best_val || (other_val == best_val && other_idx < best_idx)) {
                    best_val = other_val;
                    best_idx = other_idx;
                }
            } else {
                if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                    best_val = other_val;
                    best_idx = other_idx;
                }
            }
        }
    }
    
    extern __shared__ unsigned char shared_raw[];
    float* s_val = reinterpret_cast<float*>(shared_raw);
    int64_t* s_idx = reinterpret_cast<int64_t*>(s_val + 32);
    
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        s_val[warp] = best_val;
        s_idx[warp] = best_idx;
    }
    __syncthreads();
    
    if (warp == 0) {
        int warp_count = (blockDim.x + 31) >> 5;
        if (lane < warp_count) {
            best_val = s_val[lane];
            best_idx = s_idx[lane];
        } else {
            best_val = is_argmin ? 1e38f : -1e38f;
            best_idx = -1;
        }
        active = __activemask();
        for (int delta = 16; delta > 0; delta >>= 1) {
            float other_val = __shfl_down_sync(active, best_val, delta);
            int64_t other_idx = __shfl_down_sync(active, best_idx, delta);
            if (other_idx >= 0) {
                if (best_idx < 0) {
                    best_val = other_val;
                    best_idx = other_idx;
                } else if (is_argmin) {
                    if (other_val < best_val || (other_val == best_val && other_idx < best_idx)) {
                        best_val = other_val;
                        best_idx = other_idx;
                    }
                } else {
                    if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                        best_val = other_val;
                        best_idx = other_idx;
                    }
                }
            }
        }
        if (lane == 0) {
            output[out_idx] = best_idx;
        }
    }
}

struct WelfordTuple {
    int count;
    float mean;
    float m2;
};

__device__ inline WelfordTuple merge_welford(WelfordTuple a, WelfordTuple b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    int count = a.count + b.count;
    float delta = b.mean - a.mean;
    float mean = a.mean + delta * ((float)b.count / (float)count);
    float m2 = a.m2 + b.m2 + delta * delta * ((float)a.count * (float)b.count / (float)count);
    WelfordTuple res;
    res.count = count;
    res.mean = mean;
    res.m2 = m2;
    return res;
}

extern "C" __global__ void incin_cuda_welford(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_offset,
    int reduce_dim_size,
    int out_numel,
    int unbiased,
    int is_std)
{
    int out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;
    int tid = threadIdx.x;
    int row_start = in_offset + out_idx * reduce_dim_size;
    
    WelfordTuple acc;
    acc.count = 0; acc.mean = 0.0f; acc.m2 = 0.0f;
    for (int i = tid; i < reduce_dim_size; i += blockDim.x) {
        float x = input[row_start + i];
        WelfordTuple curr;
        curr.count = 1; curr.mean = x; curr.m2 = 0.0f;
        acc = merge_welford(acc, curr);
    }
    
    unsigned int active = __activemask();
    for (int delta = 16; delta > 0; delta >>= 1) {
        int o_count = __shfl_down_sync(active, acc.count, delta);
        float o_mean = __shfl_down_sync(active, acc.mean, delta);
        float o_m2 = __shfl_down_sync(active, acc.m2, delta);
        WelfordTuple other;
        other.count = o_count; other.mean = o_mean; other.m2 = o_m2;
        acc = merge_welford(acc, other);
    }
    
    extern __shared__ unsigned char shared_raw[];
    WelfordTuple* s_welford = reinterpret_cast<WelfordTuple*>(shared_raw);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        s_welford[warp] = acc;
    }
    __syncthreads();
    
    if (warp == 0) {
        int warp_count = (blockDim.x + 31) >> 5;
        if (lane < warp_count) {
            acc = s_welford[lane];
        } else {
            acc.count = 0; acc.mean = 0.0f; acc.m2 = 0.0f;
        }
        active = __activemask();
        for (int delta = 16; delta > 0; delta >>= 1) {
            int o_count = __shfl_down_sync(active, acc.count, delta);
            float o_mean = __shfl_down_sync(active, acc.mean, delta);
            float o_m2 = __shfl_down_sync(active, acc.m2, delta);
            WelfordTuple other;
            other.count = o_count; other.mean = o_mean; other.m2 = o_m2;
            acc = merge_welford(acc, other);
        }
        if (lane == 0) {
            float divisor = (float)(acc.count - (unbiased ? 1 : 0));
            float var = divisor > 0.0f ? (acc.m2 / divisor) : 0.0f;
            output[out_idx] = is_std ? sqrtf(var) : var;
        }
    }
}

extern "C" __global__ void incin_cuda_cumsum(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_offset,
    int dim_len,
    int num_slices,
    int slice_stride,
    int is_reverse)
{
    int slice_idx = blockIdx.x;
    if (slice_idx >= num_slices) return;
    
    int base_offset = in_offset + (slice_idx / slice_stride) * (dim_len * slice_stride) + (slice_idx % slice_stride);
    
    float acc = 0.0f;
    if (is_reverse) {
        for (int i = dim_len - 1; i >= 0; i--) {
            int pos = base_offset + i * slice_stride;
            acc += input[pos];
            output[pos] = acc;
        }
    } else {
        for (int i = 0; i < dim_len; i++) {
            int pos = base_offset + i * slice_stride;
            acc += input[pos];
            output[pos] = acc;
        }
    }
}

extern "C" __global__ void incin_cuda_topk(
    const float* __restrict__ input,
    float* __restrict__ out_vals,
    int64_t* __restrict__ out_indices,
    int in_offset,
    int dim_len,
    int num_slices,
    int slice_stride,
    int k,
    int largest)
{
    int slice_idx = blockIdx.x;
    if (slice_idx >= num_slices) return;
    
    int in_base = in_offset + (slice_idx / slice_stride) * (dim_len * slice_stride) + (slice_idx % slice_stride);
    int out_base = (slice_idx / slice_stride) * (k * slice_stride) + (slice_idx % slice_stride);
    
    for (int step = 0; step < k; step++) {
        float best_v = largest ? -1e38f : 1e38f;
        int best_pos = -1;
        
        for (int j = 0; j < dim_len; j++) {
            bool already_picked = false;
            for (int s = 0; s < step; s++) {
                if (out_indices[out_base + s * slice_stride] == (int64_t)j) {
                    already_picked = true;
                    break;
                }
            }
            if (already_picked) continue;
            
            float v = input[in_base + j * slice_stride];
            if (best_pos < 0) {
                best_v = v;
                best_pos = j;
            } else if (largest) {
                if (v > best_v || (v == best_v && j < best_pos)) {
                    best_v = v;
                    best_pos = j;
                }
            } else {
                if (v < best_v || (v == best_v && j < best_pos)) {
                    best_v = v;
                    best_pos = j;
                }
            }
        }
        
        out_vals[out_base + step * slice_stride] = best_v;
        out_indices[out_base + step * slice_stride] = (int64_t)best_pos;
    }
}
"#;

#[cfg(feature = "cuda")]
fn ensure_reduce_ops_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "reduce_ops").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel("reduce_ops", REDUCE_OPS_SRC, "reduce_ops")?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_argmax_argmin_op(
    op_name: &'static str,
    storage: &CudaStorage,
    axis: Option<usize>,
    index_dtype: DTypeId,
) -> Result<CudaStorage> {
    let buffer = &*storage.buffer;
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, op_name)?;
    let device_id = buffer.device_id;
    ensure_reduce_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("reduce_ops", "incin_cuda_argmax_argmin")?;
    let stream = buffer.device.default_stream();

    let (_reduce_axis, _keepdim_shape, final_shape, reduce_dim_size) = match axis {
        Some(dim) => {
            if dim >= storage.shape.len() {
                return Err(Error::Msg(format!(
                    "CUDA {op_name} axis {dim} is out of bounds for rank {}",
                    storage.shape.len()
                )));
            }
            let (k_shape, f_shape) = reduction_shapes(&storage.shape, dim, false);
            (dim, k_shape, f_shape, storage.shape[dim])
        }
        None => {
            let total = storage.shape.iter().product::<usize>();
            (0, vec![1], vec![], total)
        }
    };

    let out_numel = final_shape.iter().product::<usize>().max(1);
    let is_argmin = if op_name == "argmin" { 1i32 } else { 0i32 };

    let byte_len = crate::bytes::byte_len(index_dtype, out_numel, OperationKind::Reduction)?;
    let mut out_buffer =
        CudaBuffer {
            len: out_numel,
            dtype: index_dtype.descriptor(),
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                Error::Msg(format!("CUDA {op_name} output allocation failed: {e:?}"))
            })?),
            device: buffer.device.clone(),
            device_id,
        };

    if out_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), final_shape));
    }

    let in_offset = checked_i32(storage.offset_elements, "input offset")?;
    let reduce_dim = checked_i32(reduce_dim_size, "reduce dimension")?;
    let out_numel_i32 = checked_i32(out_numel, "output element count")?;
    let block_size = 256u32;
    // One block per output element, not one thread. The kernel reads its output
    // position from `blockIdx.x` and uses the whole block to stride over the
    // reduction axis, cooperating through a warp shuffle -- so a grid sized as
    // `out_numel / block_size` launches one block for any output up to 256
    // elements, computes row zero, and leaves every other row at its
    // zero-initialised value. That is silently wrong rather than an error:
    // `argmax` over a 2 x N returns a correct first row and a zero second one.
    let grid_size = crate::cuda::checked_u32(out_numel.max(1), "grid size")?;
    // SAFETY: Launches reduction kernel with bounds-checked arguments and exclusive output buffer access.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 32 * 4 + 32 * 8,
        };
        stream
            .launch_builder(&function)
            .arg(&*buffer.data)
            .arg(&mut *out_u8)
            .arg(&in_offset)
            .arg(&reduce_dim)
            .arg(&out_numel_i32)
            .arg(&is_argmin)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA {op_name} launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), final_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_welford_var_std(
    storage: &CudaStorage,
    axis: Option<usize>,
    keepdim: bool,
    unbiased: bool,
    is_std: bool,
) -> Result<CudaStorage> {
    let buffer = &*storage.buffer;
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, "welford")?;
    let device_id = buffer.device_id;
    ensure_reduce_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("reduce_ops", "incin_cuda_welford")?;
    let stream = buffer.device.default_stream();

    let (final_shape, reduce_dim_size) = match axis {
        Some(dim) => {
            if dim >= storage.shape.len() {
                return Err(Error::Msg(format!(
                    "CUDA welford axis {dim} is out of bounds for rank {}",
                    storage.shape.len()
                )));
            }
            let (k_shape, f_shape) = reduction_shapes(&storage.shape, dim, keepdim);
            (if keepdim { k_shape } else { f_shape }, storage.shape[dim])
        }
        None => {
            let total = storage.shape.iter().product::<usize>();
            (
                if keepdim {
                    vec![1; storage.shape.len()]
                } else {
                    vec![]
                },
                total,
            )
        }
    };

    let out_numel = final_shape.iter().product::<usize>().max(1);
    let byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::Reduction)?;
    let mut out_buffer =
        CudaBuffer {
            len: out_numel,
            dtype: DTypeId::F32.descriptor(),
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                Error::Msg(format!("CUDA welford output allocation failed: {e:?}"))
            })?),
            device: buffer.device.clone(),
            device_id,
        };

    if out_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), final_shape));
    }

    let in_offset = checked_i32(storage.offset_elements, "input offset")?;
    let reduce_dim = checked_i32(reduce_dim_size, "reduce dimension")?;
    let out_numel_i32 = checked_i32(out_numel, "output element count")?;
    let unbiased_i32 = if unbiased { 1i32 } else { 0i32 };
    let std_i32 = if is_std { 1i32 } else { 0i32 };
    let block_size = 256u32;
    let grid_size = u32::try_from(out_numel).map_err(|_| Error::Msg("Grid overflow".into()))?;

    // SAFETY: Launches Welford reduction kernel with bounds-checked arguments and exclusive output buffer access.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 32 * 12,
        };
        stream
            .launch_builder(&function)
            .arg(&*buffer.data)
            .arg(&mut *out_u8)
            .arg(&in_offset)
            .arg(&reduce_dim)
            .arg(&out_numel_i32)
            .arg(&unbiased_i32)
            .arg(&std_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA welford launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), final_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_cumsum_op(storage: &CudaStorage, axis: usize) -> Result<CudaStorage> {
    launch_scan_op(storage, axis, false)
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_reverse_cumsum_op(storage: &CudaStorage, axis: usize) -> Result<CudaStorage> {
    launch_scan_op(storage, axis, true)
}

#[cfg(feature = "cuda")]
fn launch_scan_op(storage: &CudaStorage, axis: usize, is_reverse: bool) -> Result<CudaStorage> {
    let buffer = &*storage.buffer;
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, "cumsum")?;
    if axis >= storage.shape.len() {
        return Err(Error::Msg(format!(
            "CUDA scan axis {axis} is out of bounds for rank {}",
            storage.shape.len()
        )));
    }
    let device_id = buffer.device_id;
    ensure_reduce_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("reduce_ops", "incin_cuda_cumsum")?;
    let stream = buffer.device.default_stream();

    let total_numel = storage.shape.iter().product::<usize>();
    let dim_len = storage.shape[axis];
    let slice_stride = storage.shape[axis + 1..].iter().product::<usize>().max(1);
    let num_slices = total_numel / dim_len.max(1);

    let byte_len = crate::bytes::byte_len(DTypeId::F32, total_numel, OperationKind::Reduction)?;
    let mut out_buffer = CudaBuffer {
        len: total_numel,
        dtype: DTypeId::F32.descriptor(),
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(byte_len)
                .map_err(|e| Error::Msg(format!("CUDA scan output allocation failed: {e:?}")))?,
        ),
        device: buffer.device.clone(),
        device_id,
    };

    if total_numel == 0 {
        return Ok(CudaStorage::new(
            Arc::new(out_buffer),
            storage.shape.to_vec(),
        ));
    }

    let in_offset = checked_i32(storage.offset_elements, "input offset")?;
    let dim_len_i32 = checked_i32(dim_len, "dim length")?;
    let num_slices_i32 = checked_i32(num_slices, "num slices")?;
    let slice_stride_i32 = checked_i32(slice_stride, "slice stride")?;
    let reverse_i32 = if is_reverse { 1i32 } else { 0i32 };

    // SAFETY: Launches scan kernel with validated dimensions and exclusive output buffer access.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (u32::try_from(num_slices).unwrap_or(1), 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        stream
            .launch_builder(&function)
            .arg(&*buffer.data)
            .arg(&mut *out_u8)
            .arg(&in_offset)
            .arg(&dim_len_i32)
            .arg(&num_slices_i32)
            .arg(&slice_stride_i32)
            .arg(&reverse_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA cumsum launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(
        Arc::new(out_buffer),
        storage.shape.to_vec(),
    ))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_topk_op(
    storage: &CudaStorage,
    k: usize,
    axis: usize,
    largest: bool,
    index_dtype: DTypeId,
) -> Result<(CudaStorage, CudaStorage)> {
    let buffer = &*storage.buffer;
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, "topk")?;
    if axis >= storage.shape.len() {
        return Err(Error::Msg(format!(
            "CUDA topk axis {axis} is out of bounds for rank {}",
            storage.shape.len()
        )));
    }
    let device_id = buffer.device_id;
    ensure_reduce_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("reduce_ops", "incin_cuda_topk")?;
    let stream = buffer.device.default_stream();

    let mut out_shape = storage.shape.to_vec();
    let k_clamped = k.min(storage.shape[axis]);
    out_shape[axis] = k_clamped;
    let out_numel = out_shape.iter().product::<usize>();

    let total_numel = storage.shape.iter().product::<usize>();
    let dim_len = storage.shape[axis];
    let slice_stride = storage.shape[axis + 1..].iter().product::<usize>().max(1);
    let num_slices = total_numel / dim_len.max(1);

    let val_byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::Reduction)?;
    let idx_byte_len = crate::bytes::byte_len(index_dtype, out_numel, OperationKind::Reduction)?;

    let mut val_buffer = CudaBuffer {
        len: out_numel,
        dtype: DTypeId::F32.descriptor(),
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(val_byte_len)
                .map_err(|e| Error::Msg(format!("CUDA topk values allocation failed: {e:?}")))?,
        ),
        device: buffer.device.clone(),
        device_id,
    };
    let mut idx_buffer = CudaBuffer {
        len: out_numel,
        dtype: index_dtype.descriptor(),
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(idx_byte_len)
                .map_err(|e| Error::Msg(format!("CUDA topk indices allocation failed: {e:?}")))?,
        ),
        device: buffer.device.clone(),
        device_id,
    };

    if out_numel == 0 {
        return Ok((
            CudaStorage::new(Arc::new(val_buffer), out_shape.clone()),
            CudaStorage::new(Arc::new(idx_buffer), out_shape),
        ));
    }

    let in_offset = checked_i32(storage.offset_elements, "input offset")?;
    let dim_len_i32 = checked_i32(dim_len, "dim length")?;
    let num_slices_i32 = checked_i32(num_slices, "num slices")?;
    let slice_stride_i32 = checked_i32(slice_stride, "slice stride")?;
    let k_i32 = checked_i32(k_clamped, "k")?;
    let largest_i32 = if largest { 1i32 } else { 0i32 };

    // SAFETY: Launches topk kernel with validated dimensions and exclusive output buffer access.
    unsafe {
        let val_u8 = Arc::get_mut(&mut val_buffer.data)
            .ok_or_else(|| Error::Msg("Values buffer unexpectedly shared".into()))?;
        let idx_u8 = Arc::get_mut(&mut idx_buffer.data)
            .ok_or_else(|| Error::Msg("Indices buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (u32::try_from(num_slices).unwrap_or(1), 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        stream
            .launch_builder(&function)
            .arg(&*buffer.data)
            .arg(&mut *val_u8)
            .arg(&mut *idx_u8)
            .arg(&in_offset)
            .arg(&dim_len_i32)
            .arg(&num_slices_i32)
            .arg(&slice_stride_i32)
            .arg(&k_i32)
            .arg(&largest_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA topk launch failed: {e:?}")))?;
    }

    Ok((
        CudaStorage::new(Arc::new(val_buffer), out_shape.clone()),
        CudaStorage::new(Arc::new(idx_buffer), out_shape),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use incin_core::tensor::dtype::DTypeId;

    #[test]
    fn checked_reduction_metadata_rejects_narrowing_and_overflow() {
        assert_eq!(checked_i32(i32::MAX as usize, "test").unwrap(), i32::MAX);
        assert!(checked_i32(i32::MAX as usize + 1, "test").is_err());
        assert!(
            crate::bytes::byte_len(DTypeId::F16, usize::MAX, OperationKind::Reduction).is_err()
        );
    }

    #[test]
    fn reduction_shapes_preserve_or_remove_exactly_one_axis() {
        assert_eq!(
            reduction_shapes(&[2, 3, 4], 1, true),
            (vec![2, 1, 4], vec![2, 1, 4])
        );
        assert_eq!(
            reduction_shapes(&[2, 3, 4], 1, false),
            (vec![2, 1, 4], vec![2, 4])
        );
    }
}
