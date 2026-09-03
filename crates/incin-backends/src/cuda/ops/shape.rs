use super::alloc_zeroed_bytes;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use alloc::vec::Vec;
use incin_core::error::{BackendError, Error, Result};
use incin_core::shapes::{OperationKind, ShapeBuf, ShapeError};
use incin_core::tensor::dtype::DTypeId;

/// Packs the `[u32; 21]` params buffer `kernels/shape.cu`'s `shape_op` kernel
/// expects: `[op_mode, rank, n_elements, out_shape(6), inp_shape(6), aux(6)]`,
/// shapes right-aligned/padded with leading `1`s to a fixed rank-6 layout.
/// Direct port of `wgpu/dispatch.rs::prepare_shape_params` - same op_mode
/// values (0=narrow, 2=transpose, 3=broadcast), same `aux` semantics (narrow
/// start offsets, or transpose's per-output-dim source-dim map, offset by
/// the output's padding amount so it indexes correctly into `multi_idx`).
/// `pub(crate)` for `cuda::ops::select::launch_broadcast_bool_mask`, which
/// reuses this to broadcast a `bool` mask through the exact same index
/// arithmetic `shape_op`'s `op_mode == 3` uses, without going through
/// `launch_broadcast`/`shape_op` itself: that kernel's data pointers are a
/// hardcoded `float*`/`float*`, which a 1-byte `bool` buffer cannot answer
/// (see `select.rs`'s own doc for the byte-width trap this avoids).
#[cfg(feature = "cuda")]
pub(crate) fn prepare_shape_params(
    op_mode: u32,
    n_elements: u32,
    out_shape: &[usize],
    inp_shape: &[usize],
    aux: &[usize],
) -> Result<[u32; 21]> {
    if out_shape.len() > 6 || inp_shape.len() > 6 || aux.len() > 6 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Storage,
            parameter: "CUDA shape-kernel rank",
            value: core::cmp::max(out_shape.len(), core::cmp::max(inp_shape.len(), aux.len())),
        }
        .into());
    }
    let mut params = [0u32; 21];
    params[0] = op_mode;
    params[1] = crate::cuda::checked_u32(
        core::cmp::max(out_shape.len(), inp_shape.len()),
        "CUDA shape-kernel rank",
    )?;
    params[2] = n_elements;

    let pad_out = 6 - out_shape.len();
    for (i, &s) in out_shape.iter().enumerate() {
        params[3 + pad_out + i] = crate::cuda::checked_u32(s, "CUDA output dimension")?;
    }
    for i in 0..pad_out {
        params[3 + i] = 1;
    }

    let pad_inp = 6 - inp_shape.len();
    for (i, &s) in inp_shape.iter().enumerate() {
        params[9 + pad_inp + i] = crate::cuda::checked_u32(s, "CUDA input dimension")?;
    }
    for i in 0..pad_inp {
        params[9 + i] = 1;
    }

    let pad_aux = 6 - aux.len();
    for (i, &s) in aux.iter().enumerate() {
        let mut val = crate::cuda::checked_u32(s, "CUDA shape auxiliary value")?;
        if op_mode == 2 {
            val = val
                .checked_add(crate::cuda::checked_u32(pad_out, "CUDA transpose padding")?)
                .ok_or(ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Transpose,
                    expression: "CUDA transpose axis plus padding",
                })?;
        }
        params[15 + pad_aux + i] = val;
    }
    for i in 0..pad_aux {
        params[15 + i] = 0;
    }

    Ok(params)
}

#[cfg(feature = "cuda")]
fn ensure_shape_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "shape").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel(
            "shape",
            crate::cuda::ops::kernels::SHAPE_KERNEL,
            "shape",
        )?;
    }
    Ok(())
}

/// Shared launcher for `narrow`/`paste`/`transpose`/`broadcast_as` - all are
/// the same per-thread strided gather-or-scatter, differing only in how a
/// thread's index maps to an input/output flat offset (see `shape.cu`).
/// `launch_n` is the thread count: the output's element count for
/// narrow/transpose/broadcast, but the (smaller) *input*'s element count
/// for paste, which scatters into a larger, pre-zeroed output - see
/// `scatter_into_zeros`.
#[cfg(feature = "cuda")]
fn launch_shape_op(
    op_mode: u32,
    t: &CudaStorage,
    out_shape: Vec<usize>,
    aux: &[usize],
    launch_n: usize,
) -> Result<CudaStorage> {
    let t_buf = &*t.buffer;
    let device_id = t_buf.device_id;
    ensure_shape_loaded(device_id)?;

    let item_bytes = crate::bytes::byte_len(t_buf.dtype, 1, OperationKind::Reshape)?;
    let kernel_name = match item_bytes {
        1 => "shape_op_8bit",
        2 => "shape_op_16bit",
        8 => "shape_op_64bit",
        _ => "shape_op_32bit",
    };

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("shape", kernel_name)?;
    let stream = t_buf.device.default_stream();

    let n_elements: usize = incin_core::shapes::ShapeBuf::from_slice(&(out_shape))
        .checked_numel(incin_core::shapes::error::OperationKind::Storage)?;
    let launch_n_u32 = crate::cuda::checked_u32(launch_n, "CUDA shape-op grid dimension")?;
    let params = prepare_shape_params(op_mode, launch_n_u32, &out_shape, &t.shape, aux)?;
    let params_u8: &[u8] = bytemuck::cast_slice(&params);
    let params_dev = stream
        .clone_htod(params_u8)
        .map_err(|e| incin_core::error::Error::Msg(format!("shape params upload failed: {e:?}")))?;

    let mut out_b = CudaBuffer {
        len: n_elements,
        dtype: t_buf.dtype,
        data: Arc::new(alloc_zeroed_bytes(
            &stream,
            t_buf.dtype,
            n_elements,
            OperationKind::Reshape,
        )?),
        device: t_buf.device.clone(),
        device_id,
    };

    let block_size: u32 = 256;
    let grid_size = launch_n_u32.div_ceil(block_size);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: checked reshape metadata fixes the source and 21-u32 parameter
    // views, output length, and launch dimensions; out_b is uniquely owned.
    unsafe {
        let in_f32 = t_buf.data.transmute::<f32>(t_buf.len).unwrap();
        let params_f32 = params_dev.transmute::<u32>(21).unwrap();
        // out_b.data was allocated once immediately above and never cloned,
        // so it stays uniquely owned (refcount 1) here - Arc::get_mut
        // succeeds without cloning first.
        let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .expect("out_b.data is freshly allocated and uniquely owned here");
        let mut out_f32 = out_u8.transmute_mut::<f32>(n_elements).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&in_f32)
            .arg(&mut out_f32)
            .arg(&params_f32)
            .launch(cfg)
            .map_err(|e| incin_core::error::Error::Msg(format!("shape_op launch failed: {e:?}")))?;
    }

    let strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0)
}

/// Narrows dimension `dim` of `t` to the half-open range `[start, start+len)`.
/// Materializes a fresh contiguous buffer (unlike CPU's metadata-only
/// `narrow`) - CUDA's elementwise/matmul/reduce kernels read flat contiguous
/// memory, so a non-contiguous, stride-sharing view would silently corrupt
/// any op run on it afterward. Mirrors `wgpu/backend.rs::narrow`'s same
/// materializing choice, made for the same reason.
#[cfg(feature = "cuda")]
pub(crate) fn launch_narrow(
    t: &CudaStorage,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<CudaStorage> {
    let mut out_shape = t.shape.to_vec();
    out_shape[dim] = len;
    let mut aux = alloc::vec![0usize; t.shape.len()];
    aux[dim] = start;
    let launch_n = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::Storage)?;
    launch_shape_op(0, t, out_shape, &aux, launch_n)
}

/// Swaps dims `dim1`/`dim2`. Materializes (see `launch_narrow`'s doc for why).
#[cfg(feature = "cuda")]
pub(crate) fn launch_transpose(t: &CudaStorage, dim1: usize, dim2: usize) -> Result<CudaStorage> {
    let mut out_shape = t.shape.to_vec();
    out_shape.swap(dim1, dim2);
    // aux[output_dim] = source_dim it reads from, i.e. the same permutation
    // that produced out_shape from t.shape.
    let mut aux: Vec<usize> = (0..t.shape.len()).collect();
    aux.swap(dim1, dim2);
    let launch_n = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::Storage)?;
    launch_shape_op(2, t, out_shape, &aux, launch_n)
}

/// Transposes by permuting metadata, without touching the buffer.
///
/// The counterpart to `launch_transpose`, which runs a permutation kernel into
/// a fresh contiguous allocation. This one shares the original buffer and
/// permutes the shape and strides, so it costs an `Arc` clone and no device
/// work at all.
///
/// Neither is universally better, which is why both exist. Measured on a
/// GTX 1650 for a transpose followed by pointwise consumption of the result,
/// the view beats the copy by roughly 11% when the result is read once, and
/// loses by roughly 15% when it is read eight times; the crossover sits between
/// two and four reads. That is a property of the *consumer*, which the
/// transpose cannot know, so the caller chooses. See
/// `cuda::ops::view_cost_bench` and issue #113.
///
/// The result is genuinely non-contiguous, so it takes the strided pointwise
/// kernels rather than the dense ones.
#[cfg(feature = "cuda")]
pub(crate) fn launch_transpose_view(
    t: &CudaStorage,
    dim1: usize,
    dim2: usize,
) -> Result<CudaStorage> {
    let mut out_shape = t.shape.to_vec();
    let mut out_strides = t.strides.to_vec();
    if dim1 >= out_shape.len() || dim2 >= out_shape.len() {
        return Err(incin_core::error::Error::Msg(alloc::format!(
            "transpose_view dims ({dim1}, {dim2}) out of range for shape {:?}",
            t.shape
        )));
    }
    out_shape.swap(dim1, dim2);
    out_strides.swap(dim1, dim2);
    CudaStorage::try_from_parts(t.buffer.clone(), out_shape, out_strides, t.offset_elements)
}

/// Broadcasts `t` to `target_shape`. Materializes (see `launch_narrow`'s doc
/// for why). Caller must validate shape compatibility first - this function
/// assumes `target_shape` is already a legal broadcast target of `t.shape`.
#[cfg(feature = "cuda")]
pub(crate) fn launch_broadcast(t: &CudaStorage, target_shape: &[usize]) -> Result<CudaStorage> {
    let launch_n = ShapeBuf::from_slice(target_shape).checked_numel(OperationKind::Broadcast)?;
    launch_shape_op(3, t, target_shape.to_vec(), &[], launch_n)
}

/// Scatters `values` into a fresh, zero-initialized buffer of shape
/// `original_shape` at `region_start` - `narrow`'s backward (the gradient
/// w.r.t. the un-narrowed input is zero everywhere except the narrowed
/// region, which gets `values` verbatim). Iterates over `values` (always
/// the smaller side) rather than the zeroed output, so `launch_n` is
/// `values`'s element count, not `original_shape`'s.
#[cfg(feature = "cuda")]
pub(crate) fn scatter_into_zeros(
    original_shape: &[usize],
    region_start: &[usize],
    values: &CudaStorage,
) -> Result<CudaStorage> {
    let launch_n = ShapeBuf::from_slice(&values.shape).checked_numel(OperationKind::Storage)?;
    launch_shape_op(1, values, original_shape.to_vec(), region_start, launch_n)
}

#[cfg(feature = "cuda")]
const CONCAT_SRC: &str = include_str!("kernels/concat.cu");

#[cfg(feature = "cuda")]
fn ensure_concat_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "concat").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel("concat", CONCAT_SRC, "concat")?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_concat(tensors: &[&CudaStorage], dim: usize) -> Result<CudaStorage> {
    if tensors.is_empty() {
        return Err(incin_core::error::Error::Msg(
            "concat: empty tensor list".into(),
        ));
    }

    let first_buf = &*tensors[0].buffer;
    let device_id = first_buf.device_id;
    for tensor in tensors {
        if tensor.buffer.dtype != first_buf.dtype {
            return Err(Error::DTypeStorageMismatch {
                expected: first_buf.dtype,
                got: tensor.buffer.dtype,
            });
        }
    }
    ensure_concat_loaded(device_id)?;

    let item_bytes = crate::bytes::byte_len(first_buf.dtype, 1, OperationKind::Concat)?;
    let kernel_name = match item_bytes {
        1 => "concat_8bit",
        2 => "concat_16bit",
        8 => "concat_64bit",
        _ => "concat_32bit",
    };

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("concat", kernel_name)?;
    let stream = first_buf.device.default_stream();

    let mut out_shape = tensors[0].shape.to_vec();
    let out_dim_total = tensors.iter().try_fold(0usize, |total, tensor| {
        total
            .checked_add(tensor.shape[dim])
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Concat,
                expression: "CUDA concat output dimension",
            })
    })?;
    out_shape[dim] = out_dim_total;

    let total = ShapeBuf::from_slice(&out_shape).checked_numel(OperationKind::Storage)?;
    let mut out_b = CudaBuffer {
        len: total,
        dtype: first_buf.dtype,
        data: Arc::new(alloc_zeroed_bytes(
            &stream,
            first_buf.dtype,
            total,
            OperationKind::Concat,
        )?),
        device: first_buf.device.clone(),
        device_id,
    };

    let outer_size: usize = incin_core::shapes::ShapeBuf::from_slice(&(out_shape[0..dim]))
        .checked_numel(incin_core::shapes::OperationKind::Storage)?;
    let inner_size: usize = if dim + 1 < out_shape.len() {
        ShapeBuf::from_slice(&out_shape[dim + 1..]).checked_numel(OperationKind::Concat)?
    } else {
        1
    };

    let outer_size_u32 = crate::cuda::checked_u32(outer_size, "CUDA concat outer size")?;
    let out_dim_total_u32 =
        crate::cuda::checked_u32(out_dim_total, "CUDA concat output dimension")?;
    let inner_size_u32 = crate::cuda::checked_u32(inner_size, "CUDA concat inner size")?;
    let mut current_offset = 0usize;
    for t in tensors {
        let t_buf = &*t.buffer;

        let in_dim_size = t.shape[dim];
        let elements = ShapeBuf::from_slice(&[outer_size, in_dim_size, inner_size])
            .checked_numel(OperationKind::Concat)?;
        if elements == 0 {
            current_offset =
                current_offset
                    .checked_add(in_dim_size)
                    .ok_or(ShapeError::ArithmeticOverflow {
                        operation: OperationKind::Concat,
                        expression: "CUDA concat cumulative offset",
                    })?;
            continue;
        }

        let block_size: u32 = 256;
        let grid_size =
            crate::cuda::checked_u32(elements, "CUDA concat grid dimension")?.div_ceil(block_size);
        let in_dim_size_u32 = crate::cuda::checked_u32(in_dim_size, "CUDA concat input dimension")?;
        let current_offset_u32 =
            crate::cuda::checked_u32(current_offset, "CUDA concat input offset")?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: each checked slice range stays within the validated source;
        // out_b remains uniquely owned across launches and lengths match f32.
        unsafe {
            let in_f32 = t_buf.data.transmute::<f32>(t_buf.len).unwrap();
            // out_b.data was allocated once before this loop and never cloned, so it
            // stays uniquely owned (refcount 1) across every iteration and
            // Arc::get_mut succeeds without cloning first.
            let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
                .expect("out_b.data is uniquely owned for the lifetime of this loop");
            let mut out_f32 = out_u8.transmute_mut::<f32>(total).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&in_f32)
                .arg(&mut out_f32)
                .arg(&outer_size_u32)
                .arg(&in_dim_size_u32)
                .arg(&out_dim_total_u32)
                .arg(&inner_size_u32)
                .arg(&current_offset_u32)
                .launch(cfg)
                .map_err(|e| {
                    incin_core::error::Error::Msg(format!("concat launch failed: {e:?}"))
                })?;
        }

        current_offset =
            current_offset
                .checked_add(in_dim_size)
                .ok_or(ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Concat,
                    expression: "CUDA concat cumulative offset",
                })?;
    }

    let strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0)
}

#[cfg(feature = "cuda")]
const EMBEDDING_SRC: &str = include_str!("kernels/embedding.cu");

#[cfg(feature = "cuda")]
const INDEX_OPS_SRC: &str = r#"
typedef long long int64_t;
typedef unsigned int uint32_t;

extern "C" __global__ void incin_cuda_gather(
    const float* __restrict__ input,
    const int64_t* __restrict__ index,
    float* __restrict__ output,
    uint32_t* __restrict__ error_flag,
    int numel,
    int rank,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_shape,
    const int* __restrict__ out_strides,
    const int* __restrict__ in_strides,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int rem = idx;
    int in_flat = 0;
    for (int d = 0; d < rank; d++) {
        int coord = rem / out_strides[d];
        rem = rem % out_strides[d];
        if (d == dim) {
            int64_t target_i = index[idx];
            if (target_i < 0 || target_i >= in_shape[dim]) {
                atomicExch(error_flag, 1);
                return;
            }
            coord = (int)target_i;
        }
        in_flat += coord * in_strides[d];
    }
    output[idx] = input[in_flat];
}

extern "C" __global__ void incin_cuda_scatter(
    const float* __restrict__ input,
    const int64_t* __restrict__ index,
    const float* __restrict__ src,
    float* __restrict__ output,
    uint32_t* __restrict__ error_flag,
    int numel_src,
    int numel_out,
    int rank,
    const int* __restrict__ idx_shape,
    const int* __restrict__ out_shape,
    const int* __restrict__ idx_strides,
    const int* __restrict__ out_strides,
    const int* __restrict__ in_strides,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel_out) {
        output[idx] = input[idx];
    }
    __syncthreads();

    if (idx < numel_src) {
        int rem = idx;
        int out_flat = 0;
        for (int d = 0; d < rank; d++) {
            int coord = rem / idx_strides[d];
            rem = rem % idx_strides[d];
            if (d == dim) {
                int64_t target_i = index[idx];
                if (target_i < 0 || target_i >= out_shape[dim]) {
                    atomicExch(error_flag, 1);
                    return;
                }
                coord = (int)target_i;
            }
            out_flat += coord * out_strides[d];
        }
        output[out_flat] = src[idx];
    }
}

extern "C" __global__ void incin_cuda_triangular(
    const float* __restrict__ input,
    float* __restrict__ output,
    int numel,
    int rows,
    int cols,
    int diagonal,
    int is_upper)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int matrix_size = rows * cols;
    int matrix_idx = idx % matrix_size;
    int r = matrix_idx / cols;
    int c = matrix_idx % cols;

    bool keep = is_upper ? (c >= r + diagonal) : (c <= r + diagonal);
    output[idx] = keep ? input[idx] : 0.0f;
}

extern "C" __global__ void incin_cuda_pad(
    const float* __restrict__ input,
    float* __restrict__ output,
    int numel_out,
    int rank,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_shape,
    const int* __restrict__ out_strides,
    const int* __restrict__ in_strides,
    const int* __restrict__ pad_before,
    float pad_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_out) return;

    int rem = idx;
    int in_flat = 0;
    bool is_padding = false;
    for (int d = 0; d < rank; d++) {
        int coord = rem / out_strides[d];
        rem = rem % out_strides[d];
        int in_coord = coord - pad_before[d];
        if (in_coord < 0 || in_coord >= in_shape[d]) {
            is_padding = true;
            break;
        }
        in_flat += in_coord * in_strides[d];
    }
    output[idx] = is_padding ? pad_val : input[in_flat];
}

extern "C" __global__ void incin_cuda_repeat(
    const float* __restrict__ input,
    float* __restrict__ output,
    int numel_out,
    int rank,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_shape,
    const int* __restrict__ out_strides,
    const int* __restrict__ in_strides)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_out) return;

    int rem = idx;
    int in_flat = 0;
    for (int d = 0; d < rank; d++) {
        int coord = rem / out_strides[d];
        rem = rem % out_strides[d];
        int in_coord = coord % in_shape[d];
        in_flat += in_coord * in_strides[d];
    }
    output[idx] = input[in_flat];
}

extern "C" __global__ void incin_cuda_diag_1d_to_2d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    int out_dim,
    int diagonal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel_out = out_dim * out_dim;
    if (idx < numel_out) {
        output[idx] = 0.0f;
    }
    __syncthreads();

    if (idx < n) {
        int r = diagonal >= 0 ? idx : idx - diagonal;
        int c = diagonal >= 0 ? idx + diagonal : idx;
        if (r < out_dim && c < out_dim) {
            output[r * out_dim + c] = input[idx];
        }
    }
}

extern "C" __global__ void incin_cuda_diag_2d_to_1d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    int out_len,
    int diagonal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_len) return;

    int r = diagonal >= 0 ? idx : idx - diagonal;
    int c = diagonal >= 0 ? idx + diagonal : idx;
    if (r < rows && c < cols) {
        output[idx] = input[r * cols + c];
    }
}
"#;

#[cfg(feature = "cuda")]
fn ensure_index_ops_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "index_ops").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel("index_ops", INDEX_OPS_SRC, "index_ops")?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn ensure_embedding_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "embedding").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel("embedding", EMBEDDING_SRC, "embedding")?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_embedding(weight: &CudaStorage, indices: &CudaStorage) -> Result<CudaStorage> {
    if weight.shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            op: "embedding",
            expected: vec![
                weight.shape.first().copied().unwrap_or(0),
                weight.shape.get(1).copied().unwrap_or(0),
            ],
            got: weight.shape.to_vec(),
            msg: "embedding weight must be 2D".into(),
        });
    }
    let vocab_size = weight.shape[0];
    let hidden_size = weight.shape[1];
    let num_indices = indices.shape.iter().product::<usize>();
    let mut out_shape = indices.shape.to_vec();
    out_shape.push(hidden_size);

    let device_id = weight.buffer.device_id;
    ensure_embedding_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let entry_point = match weight.buffer.dtype.builtin_id() {
        Some(DTypeId::F64) => "embedding_forward_f64",
        Some(DTypeId::F16) => "embedding_forward_f16",
        Some(DTypeId::BF16) => "embedding_forward_bf16",
        _ => "embedding_forward_f32",
    };
    let function = dispatcher.get_function("embedding", entry_point)?;
    let stream = weight.buffer.device.default_stream();

    let out_numel = out_shape.iter().product::<usize>();
    let byte_len = crate::bytes::byte_len(weight.buffer.dtype, out_numel, OperationKind::Storage)?;
    let mut out_buffer =
        CudaBuffer {
            len: out_numel,
            dtype: weight.buffer.dtype,
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                Error::Msg(format!("CUDA embedding output allocation failed: {e:?}"))
            })?),
            device: weight.buffer.device.clone(),
            device_id,
        };

    let error_flag_dev = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| Error::Msg(format!("CUDA error flag allocation failed: {e:?}")))?;

    if num_indices > 0 {
        let block_size = 256u32.min(hidden_size as u32).max(1);
        let grid_size = num_indices as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Launches embedding kernel with validated buffer sizes and device error flag.
        unsafe {
            let out_u8 = Arc::get_mut(&mut out_buffer.data)
                .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&function)
                .arg(&*indices.buffer.data)
                .arg(&*weight.buffer.data)
                .arg(&mut *out_u8)
                .arg(&error_flag_dev)
                .arg(&num_indices)
                .arg(&vocab_size)
                .arg(&hidden_size)
                .launch(config)
                .map_err(|e| Error::Msg(format!("CUDA embedding launch failed: {e:?}")))?;
        }

        let mut host_err = [0u32; 1];
        stream
            .memcpy_dtoh(&error_flag_dev, &mut host_err)
            .map_err(|e| Error::Msg(format!("CUDA error flag readback failed: {e:?}")))?;
        if host_err[0] != 0 {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::Embedding,
                reason: "embedding index out of bounds",
            }));
        }
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_embedding_backward(
    grad_output: &CudaStorage,
    indices: &CudaStorage,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<CudaStorage> {
    let num_indices = indices.shape.iter().product::<usize>();
    let out_shape = vec![vocab_size, hidden_size];
    let device_id = grad_output.buffer.device_id;
    ensure_embedding_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let entry_point = match grad_output.buffer.dtype.builtin_id() {
        Some(DTypeId::F64) => "embedding_backward_f64",
        Some(DTypeId::F16) => "embedding_backward_f16",
        Some(DTypeId::BF16) => "embedding_backward_bf16",
        _ => "embedding_backward_f32",
    };
    let function = dispatcher.get_function("embedding", entry_point)?;
    let stream = grad_output.buffer.device.default_stream();

    let out_numel = vocab_size * hidden_size;
    let byte_len =
        crate::bytes::byte_len(grad_output.buffer.dtype, out_numel, OperationKind::Storage)?;
    let mut out_buffer =
        CudaBuffer {
            len: out_numel,
            dtype: grad_output.buffer.dtype,
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                Error::Msg(format!("CUDA embedding grad allocation failed: {e:?}"))
            })?),
            device: grad_output.buffer.device.clone(),
            device_id,
        };

    let error_flag_dev = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| Error::Msg(format!("CUDA error flag allocation failed: {e:?}")))?;

    if num_indices > 0 {
        let block_size = 256u32.min(hidden_size as u32).max(1);
        let grid_size = num_indices as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Launches embedding backward kernel with validated buffer sizes and device error flag.
        unsafe {
            let out_u8 = Arc::get_mut(&mut out_buffer.data)
                .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&function)
                .arg(&*grad_output.buffer.data)
                .arg(&*indices.buffer.data)
                .arg(&mut *out_u8)
                .arg(&error_flag_dev)
                .arg(&num_indices)
                .arg(&vocab_size)
                .arg(&hidden_size)
                .launch(config)
                .map_err(|e| Error::Msg(format!("CUDA embedding backward launch failed: {e:?}")))?;
        }

        let mut host_err = [0u32; 1];
        stream
            .memcpy_dtoh(&error_flag_dev, &mut host_err)
            .map_err(|e| Error::Msg(format!("CUDA error flag readback failed: {e:?}")))?;
        if host_err[0] != 0 {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::Embedding,
                reason: "embedding backward index out of bounds",
            }));
        }
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_gather(
    input: &CudaStorage,
    dim: usize,
    index: &CudaStorage,
) -> Result<CudaStorage> {
    if dim >= input.shape.len() {
        return Err(Error::Msg(format!(
            "CUDA gather dimension {dim} is out of bounds for input shape {:?}",
            input.shape
        )));
    }
    let rank = input.shape.len();
    if index.shape.len() != rank {
        return Err(Error::Msg(format!(
            "CUDA gather index rank {} must match input rank {}",
            index.shape.len(),
            rank
        )));
    }

    let device_id = input.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_gather")?;
    let stream = input.buffer.device.default_stream();

    let out_shape = index.shape.to_vec();
    let out_numel = out_shape.iter().product::<usize>();
    let byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::Storage)?;

    let mut out_buffer = CudaBuffer {
        len: out_numel,
        dtype: DTypeId::F32.descriptor(),
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(byte_len)
                .map_err(|e| Error::Msg(format!("CUDA gather output allocation failed: {e:?}")))?,
        ),
        device: input.buffer.device.clone(),
        device_id,
    };

    if out_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), out_shape));
    }

    let error_flag_dev = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| Error::Msg(format!("CUDA error flag allocation failed: {e:?}")))?;

    let out_strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .iter()
        .map(|&s| s as i32)
        .collect::<Vec<_>>();
    let in_strides = crate::layout::contiguous_strides(&input.shape)
        .strides()
        .iter()
        .map(|&s| s as i32)
        .collect::<Vec<_>>();
    let out_shape_i32 = out_shape.iter().map(|&s| s as i32).collect::<Vec<_>>();
    let in_shape_i32 = input.shape.iter().map(|&s| s as i32).collect::<Vec<_>>();

    let out_shape_dev = stream
        .clone_htod(&out_shape_i32)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let in_shape_dev = stream
        .clone_htod(&in_shape_i32)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let out_strides_dev = stream
        .clone_htod(&out_strides)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let in_strides_dev = stream
        .clone_htod(&in_strides)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;

    let block_size = 256u32;
    let grid_size = (out_numel as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_numel_i32 = out_numel as i32;
    let rank_i32 = rank as i32;
    let dim_i32 = dim as i32;

    // SAFETY: Launches gather kernel with bounds-checked parameters and device error flag.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*input.buffer.data)
            .arg(&*index.buffer.data)
            .arg(&mut *out_u8)
            .arg(&error_flag_dev)
            .arg(&out_numel_i32)
            .arg(&rank_i32)
            .arg(&out_shape_dev)
            .arg(&in_shape_dev)
            .arg(&out_strides_dev)
            .arg(&in_strides_dev)
            .arg(&dim_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA gather launch failed: {e:?}")))?;
    }

    let mut host_err = [0u32; 1];
    stream
        .memcpy_dtoh(&error_flag_dev, &mut host_err)
        .map_err(|e| Error::Msg(format!("CUDA error flag readback failed: {e:?}")))?;
    if host_err[0] != 0 {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::Gather,
            reason: "index out of bounds",
        }));
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_scatter(
    input: &CudaStorage,
    dim: usize,
    index: &CudaStorage,
    src: &CudaStorage,
) -> Result<CudaStorage> {
    if dim >= input.shape.len() {
        return Err(Error::Msg(format!(
            "CUDA scatter dimension {dim} is out of bounds for input shape {:?}",
            input.shape
        )));
    }
    let rank = input.shape.len();
    if index.shape.len() != rank || src.shape.len() != rank {
        return Err(Error::Msg(format!(
            "CUDA scatter index and src rank must match input rank {rank}"
        )));
    }

    let device_id = input.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_scatter")?;
    let stream = input.buffer.device.default_stream();

    let out_shape = input.shape.to_vec();
    let out_numel = out_shape.iter().product::<usize>();
    let src_numel = src.shape.iter().product::<usize>();
    let byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::Storage)?;

    let mut out_buffer =
        CudaBuffer {
            len: out_numel,
            dtype: DTypeId::F32.descriptor(),
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                Error::Msg(format!("CUDA scatter output allocation failed: {e:?}"))
            })?),
            device: input.buffer.device.clone(),
            device_id,
        };

    if out_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), out_shape));
    }

    let error_flag_dev = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| Error::Msg(format!("CUDA error flag allocation failed: {e:?}")))?;

    let out_strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .iter()
        .map(|&s| s as i32)
        .collect::<Vec<_>>();
    let in_strides = crate::layout::contiguous_strides(&input.shape)
        .strides()
        .iter()
        .map(|&s| s as i32)
        .collect::<Vec<_>>();
    let idx_strides = crate::layout::contiguous_strides(&index.shape)
        .strides()
        .iter()
        .map(|&s| s as i32)
        .collect::<Vec<_>>();
    let out_shape_i32 = out_shape.iter().map(|&s| s as i32).collect::<Vec<_>>();
    let idx_shape_i32 = index.shape.iter().map(|&s| s as i32).collect::<Vec<_>>();

    let out_shape_dev = stream
        .clone_htod(&out_shape_i32)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let idx_shape_dev = stream
        .clone_htod(&idx_shape_i32)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let out_strides_dev = stream
        .clone_htod(&out_strides)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let in_strides_dev = stream
        .clone_htod(&in_strides)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let idx_strides_dev = stream
        .clone_htod(&idx_strides)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;

    let block_size = 256u32;
    let grid_size = (out_numel.max(src_numel) as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let src_numel_i32 = src_numel as i32;
    let out_numel_i32 = out_numel as i32;
    let rank_i32 = rank as i32;
    let dim_i32 = dim as i32;

    // SAFETY: Launches scatter kernel with bounds-checked parameters and device error flag.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*input.buffer.data)
            .arg(&*index.buffer.data)
            .arg(&*src.buffer.data)
            .arg(&mut *out_u8)
            .arg(&error_flag_dev)
            .arg(&src_numel_i32)
            .arg(&out_numel_i32)
            .arg(&rank_i32)
            .arg(&idx_shape_dev)
            .arg(&out_shape_dev)
            .arg(&idx_strides_dev)
            .arg(&out_strides_dev)
            .arg(&in_strides_dev)
            .arg(&dim_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA scatter launch failed: {e:?}")))?;
    }

    let mut host_err = [0u32; 1];
    stream
        .memcpy_dtoh(&error_flag_dev, &mut host_err)
        .map_err(|e| Error::Msg(format!("CUDA error flag readback failed: {e:?}")))?;
    if host_err[0] != 0 {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::Scatter,
            reason: "index out of bounds",
        }));
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_tril(input: &CudaStorage, diagonal: i32) -> Result<CudaStorage> {
    launch_triangular(input, diagonal, false)
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_triu(input: &CudaStorage, diagonal: i32) -> Result<CudaStorage> {
    launch_triangular(input, diagonal, true)
}

#[cfg(feature = "cuda")]
fn launch_triangular(input: &CudaStorage, diagonal: i32, is_upper: bool) -> Result<CudaStorage> {
    let rank = input.shape.len();
    if rank < 2 {
        return Err(Error::ShapeMismatch {
            op: if is_upper { "triu" } else { "tril" },
            expected: vec![2],
            got: vec![rank],
            msg: "triangular operations require rank >= 2".into(),
        });
    }
    let rows = input.shape[rank - 2];
    let cols = input.shape[rank - 1];

    let device_id = input.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_triangular")?;
    let stream = input.buffer.device.default_stream();

    let out_shape = input.shape.to_vec();
    let out_numel = out_shape.iter().product::<usize>();
    let byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::Storage)?;

    let mut out_buffer =
        CudaBuffer {
            len: out_numel,
            dtype: DTypeId::F32.descriptor(),
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                Error::Msg(format!("CUDA triangular output allocation failed: {e:?}"))
            })?),
            device: input.buffer.device.clone(),
            device_id,
        };

    if out_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), out_shape));
    }

    let block_size = 256u32;
    let grid_size = (out_numel as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let numel_i32 = out_numel as i32;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;
    let is_upper_i32 = if is_upper { 1i32 } else { 0i32 };

    // SAFETY: Launches triangular mask kernel with verified output shape and bounds.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*input.buffer.data)
            .arg(&mut *out_u8)
            .arg(&numel_i32)
            .arg(&rows_i32)
            .arg(&cols_i32)
            .arg(&diagonal)
            .arg(&is_upper_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA triangular launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_pad(
    input: &CudaStorage,
    padding: &[(usize, usize)],
    value: f64,
) -> Result<CudaStorage> {
    let rank = input.shape.len();
    if padding.len() != rank {
        return Err(Error::Msg(format!(
            "CUDA pad expects {} padding pairs, got {}",
            rank,
            padding.len()
        )));
    }

    let mut out_shape = Vec::with_capacity(rank);
    let mut pad_before = Vec::with_capacity(rank);
    for (d, &(before, after)) in padding.iter().enumerate() {
        let extent = input.shape[d] + before + after;
        out_shape.push(extent);
        pad_before.push(before as i32);
    }

    let device_id = input.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_pad")?;
    let stream = input.buffer.device.default_stream();

    let out_numel = out_shape.iter().product::<usize>();
    let byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::Storage)?;

    let mut out_buffer = CudaBuffer {
        len: out_numel,
        dtype: DTypeId::F32.descriptor(),
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(byte_len)
                .map_err(|e| Error::Msg(format!("CUDA pad output allocation failed: {e:?}")))?,
        ),
        device: input.buffer.device.clone(),
        device_id,
    };

    if out_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), out_shape));
    }

    let out_strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .iter()
        .map(|&s| s as i32)
        .collect::<Vec<_>>();
    let in_strides = crate::layout::contiguous_strides(&input.shape)
        .strides()
        .iter()
        .map(|&s| s as i32)
        .collect::<Vec<_>>();
    let out_shape_i32 = out_shape.iter().map(|&s| s as i32).collect::<Vec<_>>();
    let in_shape_i32 = input.shape.iter().map(|&s| s as i32).collect::<Vec<_>>();

    let out_shape_dev = stream
        .clone_htod(&out_shape_i32)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let in_shape_dev = stream
        .clone_htod(&in_shape_i32)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let out_strides_dev = stream
        .clone_htod(&out_strides)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let in_strides_dev = stream
        .clone_htod(&in_strides)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let pad_before_dev = stream
        .clone_htod(&pad_before)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;

    let block_size = 256u32;
    let grid_size = (out_numel as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_numel_i32 = out_numel as i32;
    let rank_i32 = rank as i32;
    let val_f32 = value as f32;

    // SAFETY: Launches padding kernel with validated strides and bounds.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*input.buffer.data)
            .arg(&mut *out_u8)
            .arg(&out_numel_i32)
            .arg(&rank_i32)
            .arg(&out_shape_dev)
            .arg(&in_shape_dev)
            .arg(&out_strides_dev)
            .arg(&in_strides_dev)
            .arg(&pad_before_dev)
            .arg(&val_f32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA pad launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_repeat(input: &CudaStorage, repeats: &[usize]) -> Result<CudaStorage> {
    let rank = input.shape.len();
    if repeats.len() != rank {
        return Err(Error::Msg(format!(
            "CUDA repeat expects {} repeats, got {}",
            rank,
            repeats.len()
        )));
    }

    let mut out_shape = Vec::with_capacity(rank);
    for (d, &r) in repeats.iter().enumerate() {
        out_shape.push(input.shape[d] * r);
    }

    let device_id = input.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_repeat")?;
    let stream = input.buffer.device.default_stream();

    let out_numel = out_shape.iter().product::<usize>();
    let byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::Storage)?;

    let mut out_buffer = CudaBuffer {
        len: out_numel,
        dtype: DTypeId::F32.descriptor(),
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(byte_len)
                .map_err(|e| Error::Msg(format!("CUDA repeat output allocation failed: {e:?}")))?,
        ),
        device: input.buffer.device.clone(),
        device_id,
    };

    if out_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), out_shape));
    }

    let out_strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .iter()
        .map(|&s| s as i32)
        .collect::<Vec<_>>();
    let in_strides = crate::layout::contiguous_strides(&input.shape)
        .strides()
        .iter()
        .map(|&s| s as i32)
        .collect::<Vec<_>>();
    let out_shape_i32 = out_shape.iter().map(|&s| s as i32).collect::<Vec<_>>();
    let in_shape_i32 = input.shape.iter().map(|&s| s as i32).collect::<Vec<_>>();

    let out_shape_dev = stream
        .clone_htod(&out_shape_i32)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let in_shape_dev = stream
        .clone_htod(&in_shape_i32)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let out_strides_dev = stream
        .clone_htod(&out_strides)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;
    let in_strides_dev = stream
        .clone_htod(&in_strides)
        .map_err(|e| Error::Msg(format!("{e:?}")))?;

    let block_size = 256u32;
    let grid_size = (out_numel as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_numel_i32 = out_numel as i32;
    let rank_i32 = rank as i32;

    // SAFETY: Launches repeat kernel with validated strides and bounds.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*input.buffer.data)
            .arg(&mut *out_u8)
            .arg(&out_numel_i32)
            .arg(&rank_i32)
            .arg(&out_shape_dev)
            .arg(&in_shape_dev)
            .arg(&out_strides_dev)
            .arg(&in_strides_dev)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA repeat launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_diag(input: &CudaStorage, diagonal: i32) -> Result<CudaStorage> {
    let rank = input.shape.len();
    let device_id = input.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let stream = input.buffer.device.default_stream();

    if rank == 1 {
        let n = input.shape[0];
        let diag_abs = diagonal.unsigned_abs() as usize;
        let out_dim = n + diag_abs;
        let out_shape = vec![out_dim, out_dim];
        let out_numel = out_dim * out_dim;
        let byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::Storage)?;

        let mut out_buffer =
            CudaBuffer {
                len: out_numel,
                dtype: DTypeId::F32.descriptor(),
                data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                    Error::Msg(format!("CUDA diag output allocation failed: {e:?}"))
                })?),
                device: input.buffer.device.clone(),
                device_id,
            };

        if n > 0 {
            let function = dispatcher.get_function("index_ops", "incin_cuda_diag_1d_to_2d")?;
            let block_size = 256u32;
            let grid_size = (out_numel as u32).div_ceil(block_size);
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            let out_dim_i32 = out_dim as i32;

            // SAFETY: Launches diag 1d to 2d kernel with validated dimensions.
            unsafe {
                let out_u8 = Arc::get_mut(&mut out_buffer.data)
                    .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
                use cudarc::driver::PushKernelArg;
                stream
                    .launch_builder(&function)
                    .arg(&*input.buffer.data)
                    .arg(&mut *out_u8)
                    .arg(&n_i32)
                    .arg(&out_dim_i32)
                    .arg(&diagonal)
                    .launch(config)
                    .map_err(|e| Error::Msg(format!("CUDA diag launch failed: {e:?}")))?;
            }
        }

        Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
    } else if rank == 2 {
        let (rows, cols) = (input.shape[0], input.shape[1]);
        let out_len = if diagonal >= 0 {
            let d = diagonal as usize;
            if d < cols { (cols - d).min(rows) } else { 0 }
        } else {
            let d = (-diagonal) as usize;
            if d < rows { (rows - d).min(cols) } else { 0 }
        };
        let out_shape = vec![out_len];
        let byte_len = crate::bytes::byte_len(DTypeId::F32, out_len, OperationKind::Storage)?;

        let mut out_buffer =
            CudaBuffer {
                len: out_len,
                dtype: DTypeId::F32.descriptor(),
                data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                    Error::Msg(format!("CUDA diag output allocation failed: {e:?}"))
                })?),
                device: input.buffer.device.clone(),
                device_id,
            };

        if out_len > 0 {
            let function = dispatcher.get_function("index_ops", "incin_cuda_diag_2d_to_1d")?;
            let block_size = 256u32;
            let grid_size = (out_len as u32).div_ceil(block_size);
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let rows_i32 = rows as i32;
            let cols_i32 = cols as i32;
            let out_len_i32 = out_len as i32;

            // SAFETY: Launches diag 2d to 1d kernel with validated dimensions.
            unsafe {
                let out_u8 = Arc::get_mut(&mut out_buffer.data)
                    .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
                use cudarc::driver::PushKernelArg;
                stream
                    .launch_builder(&function)
                    .arg(&*input.buffer.data)
                    .arg(&mut *out_u8)
                    .arg(&rows_i32)
                    .arg(&cols_i32)
                    .arg(&out_len_i32)
                    .arg(&diagonal)
                    .launch(config)
                    .map_err(|e| Error::Msg(format!("CUDA diag launch failed: {e:?}")))?;
            }
        }

        Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
    } else {
        Err(Error::ShapeMismatch {
            op: "diag",
            expected: vec![1, 2],
            got: vec![rank],
            msg: "diag requires 1D or 2D tensor".into(),
        })
    }
}
