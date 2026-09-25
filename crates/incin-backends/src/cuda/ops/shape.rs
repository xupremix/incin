use super::alloc_zeroed_bytes;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use crate::cuda::{checked_i32, checked_i32_vec, checked_u32};
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
///
/// Private to this module since #122: the only out-of-module consumer was
/// `select::launch_broadcast_bool_mask`, deleted once `shape_op`'s
/// width-parametric entry points (`shape_op_8bit` and friends) let
/// `launch_broadcast` below carry `bool` masks on the same path as every
/// other dtype.
#[cfg(feature = "cuda")]
fn prepare_shape_params(
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
    //
    // The views are `u8`, not `f32`: the kernel entry point above was already
    // selected by element width (`shape_op_8bit`/`16bit`/`32bit`/`64bit`), so
    // the device pointer is reinterpreted by the kernel itself. Reinterpreting
    // here as `f32` with an element count would ask for `len * 4` bytes, which
    // panics for 1- and 2-byte dtypes whose allocation is smaller, and only
    // passed for 8-byte dtypes because `transmute` checks `<=` rather than
    // `==`. Passing the byte buffers through unchanged keeps the launch
    // width-correct for every dtype the kernel selection admits.
    unsafe {
        let in_bytes = crate::bytes::byte_len(t_buf.dtype, t_buf.len, OperationKind::Reshape)?;
        let in_u8 = t_buf
            .data
            .transmute::<u8>(in_bytes)
            .expect("CUDA shape-op input allocation covers its element count by construction");
        let params_f32 = params_dev.transmute::<u32>(21).unwrap();
        // out_b.data was allocated once immediately above and never cloned,
        // so it stays uniquely owned (refcount 1) here - Arc::get_mut
        // succeeds without cloning first.
        let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .expect("out_b.data is freshly allocated and uniquely owned here");
        let out_bytes = crate::bytes::byte_len(t_buf.dtype, n_elements, OperationKind::Reshape)?;
        let mut out_view = out_u8
            .transmute_mut::<u8>(out_bytes)
            .expect("CUDA shape-op output allocation covers its element count by construction");

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&in_u8)
            .arg(&mut out_view)
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
///
/// TRACKED DEVIATION from the views-everywhere contract (issue #113,
/// orchestrator decision 2026-09-25): the operation contract names a strided
/// view sharing the input's buffer (`LayoutRule::ViewWhenPossible`, aliasing
/// documented on `Tensor::transpose`), and this launcher instead runs a
/// permutation kernel into a fresh contiguous allocation. It stays a copy
/// because this backend's matmul/reduce consumers refuse strided operands and
/// there is no device here to verify strided support on -- not because the
/// contract allows two behaviours. Pending strided-consumer support verified
/// on hardware; do not "fix" by relabelling the contract.
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
/// the view beats the copy by roughly 45% when the result is read once, and
/// loses by roughly 23% when it is read eight times; the crossover sits at
/// about four reads. That is a property of the *consumer*, which the transpose
/// cannot know, so the caller chooses. See `cuda::ops::view_cost_bench` and
/// issue #113.
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
///
/// Every dtype the backend can store rides this one path, `bool` masks
/// included: `launch_shape_op` picks the kernel entry point by element width
/// (`shape_op_8bit` for one-byte elements), so there is no separate
/// bool-broadcast launcher anymore (#122 deleted the last one,
/// `select::launch_broadcast_bool_mask`, after `shape_op` grew those
/// width-parametric entry points).
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
        // out_b remains uniquely owned across launches. The views are `u8`
        // for the same width reason as `launch_shape_op` above: the kernel
        // entry point was already selected by element width, so reinterpreting
        // as `f32` with an element count panics for 1- and 2-byte dtypes.
        unsafe {
            let in_bytes = crate::bytes::byte_len(t_buf.dtype, t_buf.len, OperationKind::Concat)?;
            let in_u8 = t_buf
                .data
                .transmute::<u8>(in_bytes)
                .expect("CUDA concat input allocation covers its element count by construction");
            // out_b.data was allocated once before this loop and never cloned, so it
            // stays uniquely owned (refcount 1) across every iteration and
            // Arc::get_mut succeeds without cloning first.
            let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
                .expect("out_b.data is uniquely owned for the lifetime of this loop");
            let out_bytes = crate::bytes::byte_len(first_buf.dtype, total, OperationKind::Concat)?;
            let mut out_view = out_u8
                .transmute_mut::<u8>(out_bytes)
                .expect("CUDA concat output allocation covers its element count by construction");

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&in_u8)
                .arg(&mut out_view)
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

extern "C" __global__ void incin_cuda_scatter_add(
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
        atomicAdd(&output[out_flat], src[idx]);
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

// `repeat_interleave`: each output coordinate copies its source coordinate
// with `axis` divided by `repeats` (adjacent copies, CPU's combine.rs rule).
// `out_strides` decodes the output's row-major flat back into coordinates
// (contiguous - the launcher supplies it); `in_strides` are the input's
// physical strides, so a strided input reads through them, offset by
// `input_offset`.
extern "C" __global__ void incin_cuda_repeat_interleave(
    const float* __restrict__ input,
    float* __restrict__ output,
    int numel_out,
    int rank,
    int repeats,
    int axis,
    int input_offset,
    const int* __restrict__ out_strides,
    const int* __restrict__ in_strides)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_out) return;
    int rem = idx;
    int in_flat = input_offset;
    for (int d = 0; d < rank; d++) {
        int coord = rem / out_strides[d];
        rem = rem % out_strides[d];
        if (d == axis) coord /= repeats;
        in_flat += coord * in_strides[d];
    }
    output[idx] = input[in_flat];
}

// Backward of `repeat_interleave`: each source element's cotangent is the
// sum over its `repeats` output copies. Threads own source elements; the
// inner loop walks the copies in `r = 0, 1, ...` order - the same order CPU's
// row-major accumulation visits them (positions differing only at `axis` are
// row-major ordered by that coordinate) - accumulating in `double` so the
// single f32 rounding at the end matches CPU's f64 accumulation followed by
// one `from_f64_values` conversion. `grad_out_strides` are the incoming
// gradient's physical strides (it may arrive as a view); the source decodes
// through `in_contig` because `idx` is the source's row-major flat.
extern "C" __global__ void incin_cuda_repeat_interleave_backward(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int numel_in,
    int rank,
    int repeats,
    int axis,
    int grad_offset,
    const int* __restrict__ in_contig,
    const int* __restrict__ grad_out_strides)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_in) return;
    double acc = 0.0;
    for (int r = 0; r < repeats; r++) {
        int rem = idx;
        int out_flat = grad_offset;
        for (int d = 0; d < rank; d++) {
            int coord = rem / in_contig[d];
            rem = rem % in_contig[d];
            if (d == axis) coord = coord * repeats + r;
            out_flat += coord * grad_out_strides[d];
        }
        acc += (double)grad_out[out_flat];
    }
    grad_in[idx] = (float)acc;
}

// `one_hot`: slot the target at `idx * depth + v` when `v` is inside
// `[0, depth)`; out-of-range targets leave the row all-false (ONNX rule -
// no error), matching CPU's `value >= 0 && value < depth` test. `out` is
// Bool (1 byte). Decode of the target operand runs through `in_contig`
// (logical row-major) with reads through `in_strides` (physical).
extern "C" __global__ void incin_cuda_one_hot(
    const int64_t* __restrict__ targets,
    unsigned char* __restrict__ out,
    int numel,
    int rank,
    int depth,
    int input_offset,
    const int* __restrict__ in_contig,
    const int* __restrict__ in_strides)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int rem = idx;
    int phys = input_offset;
    for (int d = 0; d < rank; d++) {
        int coord = rem / in_contig[d];
        rem = rem % in_contig[d];
        phys += coord * in_strides[d];
    }
    long long v = targets[phys];
    if (v >= 0 && v < (long long)depth) {
        out[(long long)idx * depth + v] = 1;
    }
}

// `bincount`: one increment per element. The index operand is physically
// i64 (u8/u32 storages cannot exist on CUDA), so every value is already a
// finite whole number and CPU's fract/is_finite checks reduce to the range
// test; an out-of-range value raises `error_flag` instead of silently
// lowering a count (CPU's message is reproduced host-side). Increments use
// unsigned-long-long atomics: non-negative integer addition is associative,
// so the per-bin total is exact and order-independent - deterministic by
// construction, unlike a float reduction.
extern "C" __global__ void incin_cuda_bincount(
    const int64_t* __restrict__ targets,
    long long* __restrict__ counts,
    uint32_t* __restrict__ error_flag,
    int numel,
    int rank,
    int bins,
    int input_offset,
    const int* __restrict__ in_contig,
    const int* __restrict__ in_strides)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    int rem = idx;
    int phys = input_offset;
    for (int d = 0; d < rank; d++) {
        int coord = rem / in_contig[d];
        rem = rem % in_contig[d];
        phys += coord * in_strides[d];
    }
    long long v = targets[phys];
    if (v < 0 || v >= (long long)bins) {
        atomicExch(error_flag, 1);
        return;
    }
    atomicAdd((unsigned long long*)counts + v, 1ULL);
}

// Pass 1 of `scatter_add`: for each source index position `j`, compute the
// destination flat index and the source's physical read offset, writing both
// into `scratch = [dest(flat), src(phys)]`. Mirrors CPU exactly: the
// destination coordinate at `axis` takes the index value with negatives
// clamped to zero (Rust's `as usize` saturating cast), the flat is formed
// from the OUTPUT's contiguous strides, and acceptance is a single
// `flat < numel_out` test - CPU performs no per-coordinate bounds check, so
// an out-of-range coordinate that still lands inside the buffer aliases to
// that position, as this does. Rejected positions store `-1`. Two decode
// passes avoid a coordinate buffer: one forms the index/source physical
// addresses, the second forms the destination flat.
extern "C" __global__ void incin_cuda_scatter_add_map(
    const int64_t* __restrict__ index,
    long long* __restrict__ scratch,
    int numel_src,
    int numel_out,
    int rank,
    int axis,
    int index_offset,
    int src_offset,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_contig,
    const int* __restrict__ index_strides,
    const int* __restrict__ src_strides,
    const int* __restrict__ out_contig)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= numel_src) return;

    int rem = j;
    long long index_phys = index_offset;
    long long src_phys = src_offset;
    for (int d = 0; d < rank; d++) {
        int coord = rem / in_contig[d];
        rem = rem % in_contig[d];
        index_phys += (long long)coord * index_strides[d];
        src_phys += (long long)coord * src_strides[d];
    }
    long long v = index[index_phys];
    if (v < 0) v = 0;

    rem = j;
    long long dest = 0;
    for (int d = 0; d < rank; d++) {
        int coord = rem / in_contig[d];
        rem = rem % in_contig[d];
        if (d == axis) {
            dest += v * (long long)out_contig[d];
        } else {
            dest += (long long)coord * out_contig[d];
        }
    }
    if (dest >= (long long)numel_out) dest = -1;

    scratch[j] = dest;
    scratch[numel_src + j] = src_phys;
}

// Pass 2 of `scatter_add`: each output thread copies its input value, then
// walks `j` in ascending order - CPU's row-major index iteration - adding
// every source whose mapped destination is this output position, in `double`
// and rounding once. That reproduces CPU's accumulate-in-f64-then-
// `from_f64_values` rounding exactly for both f32 and i64 value operands
// (integral doubles round-trip; the C cast truncates toward zero, which on
// an integral value is exact). `input` is the untouched starting copy;
// `dtype_code` selects f32 (0) or i64 (1). Note the tape's fixed summation
// order differs from the pre-existing atomic `incin_cuda_scatter_add` (used
// only by gather's backward), which is nondeterministic - this path is the
// executor's forward and must match CPU bitwise.
extern "C" __global__ void incin_cuda_scatter_add_ordered(
    const void* __restrict__ input,
    const void* __restrict__ src,
    void* __restrict__ output,
    const long long* __restrict__ scratch,
    int numel_out,
    int numel_src,
    int rank,
    int dtype_code,
    int input_offset,
    const int* __restrict__ out_contig,
    const int* __restrict__ in_strides)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= numel_out) return;

    int rem = o;
    int in_flat = input_offset;
    for (int d = 0; d < rank; d++) {
        int coord = rem / out_contig[d];
        rem = rem % out_contig[d];
        in_flat += coord * in_strides[d];
    }

    double acc;
    if (dtype_code == 0) {
        acc = (double)((const float*)input)[in_flat];
    } else {
        acc = (double)((const long long*)input)[in_flat];
    }
    for (int j = 0; j < numel_src; j++) {
        if (scratch[j] == (long long)o) {
            if (dtype_code == 0) {
                acc += (double)((const float*)src)[scratch[numel_src + j]];
            } else {
                acc += (double)((const long long*)src)[scratch[numel_src + j]];
            }
        }
    }
    if (dtype_code == 0) {
        ((float*)output)[o] = (float)acc;
    } else {
        ((long long*)output)[o] = (long long)acc;
    }
}

// `scatter_add`'s source cotangent: thread per index position `j`. CPU
// accumulates `grad_source[flat_src] += grad_out[flat_dest]` over the writes
// it recorded, but each recorded write's `flat_src` comes from `j`'s own
// coordinates and `j` visits distinct coordinates, so the slots are distinct
// and the `+=` degenerates to assignment: this is a gather of `grad_out` at
// the recorded destination (or zero when the write was dropped) written at
// `j`. `dtype_code` selects f32 (0) or i64 (1), matching the gradient's
// dtype the same way CPU's `from_f64_values` on `grad_out.buffer` does.
extern "C" __global__ void incin_cuda_scatter_add_backward(
    const void* __restrict__ grad_out,
    void* __restrict__ grad_src,
    const long long* __restrict__ scratch,
    int numel_src,
    int rank,
    int dtype_code,
    int grad_offset,
    const int* __restrict__ out_contig,
    const int* __restrict__ grad_strides)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= numel_src) return;

    long long dest = scratch[j];
    if (dest < 0) {
        if (dtype_code == 0) {
            ((float*)grad_src)[j] = 0.0f;
        } else {
            ((long long*)grad_src)[j] = 0;
        }
        return;
    }

    int rem = (int)dest;
    int phys = grad_offset;
    for (int d = 0; d < rank; d++) {
        int coord = rem / out_contig[d];
        rem = rem % out_contig[d];
        phys += coord * grad_strides[d];
    }
    if (dtype_code == 0) {
        ((float*)grad_src)[j] = ((const float*)grad_out)[phys];
    } else {
        ((long long*)grad_src)[j] = ((const long long*)grad_out)[phys];
    }
}

// `repeat` backward: the transpose of the forward's modulo map. Each output
// coordinate copies `out_coord[d] % in_shape[d]`, so each source element is
// copied to every coordinate `s[d] + t[d] * in_shape[d]` for
// `t[d] in [0, repeats[d])`, and its cotangent is the sum over those tiles.
//
// Threads own source elements. The inner loop walks the tiles in mixed-radix
// with the LAST axis varying fastest, which is row-major order over the
// output coordinates (`out_coord[d]` increases with `t[d]`, and row-major
// visits the last axis fastest) - the same order CPU's odometer accumulates
// them in, so the f64 sums are bitwise identical. Accumulation is in
// `double` with one f32 store at the end, matching CPU's f64 `grads` plus a
// single `from_f64_values` rounding. `grad_strides` are the incoming
// cotangent's physical strides (it may arrive as a view) and `grad_offset`
// its base; the source decodes through `in_shape` because `idx` is the
// source's row-major flat.
extern "C" __global__ void incin_cuda_repeat_backward(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int numel_in,
    int rank,
    int num_tiles,
    int grad_offset,
    const int* __restrict__ in_shape,
    const int* __restrict__ repeats,
    const int* __restrict__ grad_strides)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_in) return;
    double acc = 0.0;
    for (int tile = 0; tile < num_tiles; tile++) {
        int rem_t = tile;
        int rem_s = idx;
        int out_flat = grad_offset;
        for (int d = rank - 1; d >= 0; d--) {
            int t_d = rem_t % repeats[d];
            rem_t /= repeats[d];
            int s_d = rem_s % in_shape[d];
            rem_s /= in_shape[d];
            out_flat += (s_d + t_d * in_shape[d]) * grad_strides[d];
        }
        acc += (double)grad_out[out_flat];
    }
    grad_in[idx] = (float)acc;
}

// `diag` backward for the EXTRACT form (rank-two input -> rank-one output):
// the input gradient is zero everywhere except the extracted diagonal, where
// it carries the outgoing cotangent. The forward read position `i` from
// `(r, c) = (i, i + diagonal)` when `diagonal >= 0` and from
// `(i - diagonal, i)` otherwise, so the inverse maps back to `i = r` in the
// first case and `i = c` in the second; `out_len` guards the read and leaves
// the cell at zero when the cotangent is shorter than the diagonal.
// `grad_strides[0]` and `grad_offset` let the cotangent arrive as a view.
extern "C" __global__ void incin_cuda_diag_backward(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int numel_in,
    int cols,
    int out_len,
    int diagonal,
    int grad_offset,
    const int* __restrict__ grad_strides)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel_in) return;
    int r = idx / cols;
    int c = idx % cols;
    if ((long long)c != (long long)r + (long long)diagonal) return;
    int i = diagonal >= 0 ? r : c;
    if (i < 0 || i >= out_len) return;
    grad_in[idx] = grad_out[grad_offset + i * grad_strides[0]];
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
        // `hidden_size` and `num_indices` are `usize`; the launch grid is
        // `u32`. A bare `as` would wrap silently past `u32::MAX` and launch an
        // undersized kernel while the error-flag path assumes full coverage,
        // so both go through the same fallible conversion the norm kernels
        // use instead of truncating.
        let hidden = checked_u32(hidden_size, "embedding width")?;
        let block_size = 256u32.min(hidden).max(1);
        let grid_size = checked_u32(num_indices, "embedding index count")?;
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

    let out_numel = vocab_size
        .checked_mul(hidden_size)
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::EmbeddingExact,
            expression: "CUDA embedding output element count",
        })?;
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
        // Same fallible grid conversion as the forward kernel above: a bare
        // `as` would wrap silently past `u32::MAX` and launch an undersized
        // kernel while the error-flag path assumes full coverage.
        let hidden = checked_u32(hidden_size, "embedding width")?;
        let block_size = 256u32.min(hidden).max(1);
        let grid_size = checked_u32(num_indices, "embedding index count")?;
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

    let out_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&out_shape).strides(),
        "stride",
    )?;
    let in_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&input.shape).strides(),
        "stride",
    )?;
    let out_shape_i32 = checked_i32_vec(&out_shape, "shape")?;
    let in_shape_i32 = checked_i32_vec(&input.shape, "shape")?;

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
    // The grid is `u32` while the count is `usize`: converting with `as`
    // would wrap past `u32::MAX` and launch an undersized kernel over a
    // buffer the allocation already proved that large.
    let grid_size = checked_u32(out_numel, "element count")?.div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_numel_i32 = checked_i32(out_numel, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
    let dim_i32 = checked_i32(dim, "axis")?;

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

    let out_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&out_shape).strides(),
        "stride",
    )?;
    let in_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&input.shape).strides(),
        "stride",
    )?;
    let idx_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&index.shape).strides(),
        "stride",
    )?;
    let out_shape_i32 = checked_i32_vec(&out_shape, "shape")?;
    let idx_shape_i32 = checked_i32_vec(&index.shape, "shape")?;

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
    let grid_size = checked_u32(out_numel.max(src_numel), "element count")?.div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let src_numel_i32 = checked_i32(src_numel, "element count")?;
    let out_numel_i32 = checked_i32(out_numel, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
    let dim_i32 = checked_i32(dim, "axis")?;

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

/// Scatter-add: like [`launch_scatter`], except colliding writes sum via
/// `atomicAdd` rather than the last one winning.
///
/// This is the backward of [`launch_gather`]: every output position routes
/// its cotangent back to the source position its index named, accumulating
/// where an index selects the same row twice (matching CPU's
/// `gather_storage` backward, which does `grad_t_data[flat_dst] += ...`).
/// The overwrite kernel kept only one contribution there, so duplicate
/// indices diverged as `[1,1,0]` where the CPU reference is `[2,1,0]`.
/// f32-only, like the scatter/gather pair it mirrors.
#[cfg(feature = "cuda")]
pub(crate) fn launch_scatter_add(
    input: &CudaStorage,
    dim: usize,
    index: &CudaStorage,
    src: &CudaStorage,
) -> Result<CudaStorage> {
    if dim >= input.shape.len() {
        return Err(Error::Msg(format!(
            "CUDA scatter_add dimension {dim} is out of bounds for input shape {:?}",
            input.shape
        )));
    }
    let rank = input.shape.len();
    if index.shape.len() != rank || src.shape.len() != rank {
        return Err(Error::Msg(format!(
            "CUDA scatter_add index and src rank must match input rank {rank}"
        )));
    }

    let device_id = input.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_scatter_add")?;
    let stream = input.buffer.device.default_stream();

    let out_shape = input.shape.to_vec();
    let out_numel = out_shape.iter().product::<usize>();
    let src_numel = src.shape.iter().product::<usize>();
    let byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::Storage)?;

    let mut out_buffer = CudaBuffer {
        len: out_numel,
        dtype: DTypeId::F32.descriptor(),
        data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
            Error::Msg(format!("CUDA scatter_add output allocation failed: {e:?}"))
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

    let out_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&out_shape).strides(),
        "stride",
    )?;
    let in_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&input.shape).strides(),
        "stride",
    )?;
    let idx_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&index.shape).strides(),
        "stride",
    )?;
    let out_shape_i32 = checked_i32_vec(&out_shape, "shape")?;
    let idx_shape_i32 = checked_i32_vec(&index.shape, "shape")?;

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
    let grid_size = checked_u32(out_numel.max(src_numel), "element count")?.div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let src_numel_i32 = checked_i32(src_numel, "element count")?;
    let out_numel_i32 = checked_i32(out_numel, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
    let dim_i32 = checked_i32(dim, "axis")?;

    // SAFETY: Launches scatter_add kernel with bounds-checked parameters and device error flag.
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
            .map_err(|e| Error::Msg(format!("CUDA scatter_add launch failed: {e:?}")))?;
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

/// Scatter source-gradient with last-write-wins masking.
///
/// `launch_gather(grad_out, dim, index)` routes the output cotangent to
/// every index position, but the scatter forward's last-write-wins rule
/// means only the LAST write to each destination contributed: earlier
/// writes to the same position earned no cotangent. With unique indices
/// the gather is exact; with duplicates it over-counts (e.g. `[1,1]`
/// where the CPU reference is `[0,1]`).
///
/// This gathers on-device, then zeroes non-surviving positions on the host:
/// the index is read back, the last writer per destination is identified in
/// row-major order (matching CPU's `scatter_storage` bookkeeping), and the
/// gathered gradient is masked before re-upload. The host round-trips are
/// the honest cost of deterministic duplicate handling without a bespoke
/// on-device last-writer kernel; the forward race itself (which value wins
/// with duplicates) remains nondeterministic -- see the shape_ops caller.
#[cfg(feature = "cuda")]
pub(crate) fn launch_scatter_src_grad(
    grad_out: &CudaStorage,
    dim: usize,
    index: &CudaStorage,
) -> Result<CudaStorage> {
    let gathered = launch_gather(grad_out, dim, index)?;
    let src_numel = index.shape.iter().product::<usize>();
    if src_numel == 0 {
        return Ok(gathered);
    }
    // Read back the index as i64.
    let index_bytes = index
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*index.buffer.data)
        .map_err(|e| {
            Error::Msg(format!(
                "CUDA scatter src-grad index readback failed: {e:?}"
            ))
        })?;
    let index_vals: Vec<i64> = bytemuck::cast_slice::<u8, i64>(&index_bytes).to_vec();

    let idx_strides = crate::layout::contiguous_strides(&index.shape)
        .strides()
        .to_vec();
    let out_strides = crate::layout::contiguous_strides(&grad_out.shape)
        .strides()
        .to_vec();
    let out_numel: usize = grad_out.shape.iter().product::<usize>().max(1);
    let dim_extent = grad_out.shape.get(dim).copied().unwrap_or(0);

    // Last writer per destination flat index, in row-major src order.
    let mut last_writer: alloc::collections::BTreeMap<usize, usize> =
        alloc::collections::BTreeMap::new();
    for src_flat in 0..src_numel {
        // Decode row-major multi-index of src_flat within index.shape.
        let mut rem = src_flat;
        let mut dest_flat = 0usize;
        let mut oob = false;
        for (d, (&stride, &extent)) in idx_strides.iter().zip(index.shape.iter()).enumerate() {
            let coord = rem / stride;
            rem %= stride;
            let _ = extent;
            if d == dim {
                let target = index_vals.get(src_flat).copied().unwrap_or(-1);
                if target < 0 || (target as usize) >= dim_extent {
                    oob = true;
                    break;
                }
                dest_flat += (target as usize) * out_strides[d];
            } else {
                dest_flat += coord * out_strides[d];
            }
        }
        if oob || dest_flat >= out_numel {
            continue;
        }
        last_writer.insert(dest_flat, src_flat);
    }
    let surviving: alloc::collections::BTreeSet<usize> = last_writer.into_values().collect();

    let grad_bytes = gathered
        .buffer
        .device
        .default_stream()
        .clone_dtoh(&*gathered.buffer.data)
        .map_err(|e| Error::Msg(format!("CUDA scatter src-grad readback failed: {e:?}")))?;
    let mut grad_vals: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&grad_bytes).to_vec();
    for (i, v) in grad_vals.iter_mut().enumerate() {
        if !surviving.contains(&i) {
            *v = 0.0;
        }
    }
    let stream = gathered.buffer.device.default_stream();
    let data = stream
        .clone_htod(bytemuck::cast_slice(&grad_vals))
        .map_err(|e| Error::Msg(format!("CUDA scatter src-grad upload failed: {e:?}")))?;
    let buffer = CudaBuffer {
        len: gathered.buffer.len,
        dtype: gathered.buffer.dtype,
        data: Arc::new(data),
        device: gathered.buffer.device.clone(),
        device_id: gathered.buffer.device_id,
    };
    Ok(CudaStorage::new(Arc::new(buffer), gathered.shape.to_vec()))
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
    // Rank 1 is admitted (descriptor min_rank) and CPU keeps it: a single
    // index is column `idx` of row 0, so the kernel runs unchanged with
    // `rows = 1`. Only the empty shape has no row and no column.
    if rank == 0 {
        return Err(Error::ShapeMismatch {
            op: if is_upper { "triu" } else { "tril" },
            expected: vec![1],
            got: vec![0],
            msg: "triangular operations require at least one dimension".into(),
        });
    }
    let (rows, cols) = if rank == 1 {
        (1, input.shape[0])
    } else {
        (input.shape[rank - 2], input.shape[rank - 1])
    };

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
    // The grid is `u32` while the count is `usize`: converting with `as`
    // would wrap past `u32::MAX` and launch an undersized kernel over a
    // buffer the allocation already proved that large.
    let grid_size = checked_u32(out_numel, "element count")?.div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let numel_i32 = checked_i32(out_numel, "element count")?;
    let rows_i32 = checked_i32(rows, "extent")?;
    let cols_i32 = checked_i32(cols, "extent")?;
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

/// Uploads shape/stride vectors for kernel args, guaranteeing a non-empty
/// allocation: a rank-0 operand leaves the vector empty, and an NVRTC pointer
/// argument must still carry a dereferenceable address even though a rank-0
/// launch never enters the loop that would read it.
#[cfg(feature = "cuda")]
fn dev_i32_arg(
    stream: &Arc<cudarc::driver::CudaStream>,
    values: &[usize],
    field: &'static str,
) -> Result<cudarc::driver::CudaSlice<i32>> {
    let mut i32s = checked_i32_vec(values, field)?;
    if i32s.is_empty() {
        i32s.push(0);
    }
    stream
        .clone_htod(&i32s)
        .map_err(|e| Error::Msg(format!("{e:?}")))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_repeat_interleave(
    input: &CudaStorage,
    repeats: usize,
    axis: usize,
) -> Result<CudaStorage> {
    if input.buffer.dtype.builtin_id() != Some(DTypeId::F32) {
        return Err(Error::UnsupportedDType {
            dtype: input.buffer.dtype,
            backend: "Cuda",
            op: "repeat_interleave",
        });
    }
    if axis >= input.shape.len() {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::RepeatInterleave,
            reason: "repeat_interleave axis is outside the operand's rank",
        }));
    }
    if repeats == 0 {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::RepeatInterleave,
            reason: "repeat_interleave needs at least one repeat per element",
        }));
    }
    let mut out_shape = input.shape.to_vec();
    out_shape[axis] = out_shape[axis].checked_mul(repeats).ok_or_else(|| {
        Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::RepeatInterleave,
            reason: "interleaved output dimension overflows usize",
        })
    })?;

    let rank = input.shape.len();
    let out_numel = crate::bytes::checked_numel(&out_shape)?;
    let device_id = input.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_repeat_interleave")?;
    let stream = input.buffer.device.default_stream();

    let out_strides = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    let out_strides_dev = dev_i32_arg(&stream, &out_strides, "stride")?;
    let in_strides_dev = dev_i32_arg(&stream, input.strides.strides(), "stride")?;

    let byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::Storage)?;
    let out_buffer = CudaBuffer {
        len: out_numel,
        dtype: DTypeId::F32.descriptor(),
        data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
            Error::Msg(format!(
                "CUDA repeat_interleave output allocation failed: {e:?}"
            ))
        })?),
        device: input.buffer.device.clone(),
        device_id,
    };
    if out_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), out_shape));
    }

    let numel_i32 = checked_i32(out_numel, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
    let repeats_i32 = checked_i32(repeats, "repeats")?;
    let axis_i32 = checked_i32(axis, "axis")?;
    let offset_i32 = checked_i32(input.offset_elements, "input offset")?;
    let block_size = 256u32;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut out_buffer = out_buffer;
    // SAFETY: Launches repeat_interleave with bounds-checked parameters over a
    // fresh output allocation.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*input.buffer.data)
            .arg(&mut *out_u8)
            .arg(&numel_i32)
            .arg(&rank_i32)
            .arg(&repeats_i32)
            .arg(&axis_i32)
            .arg(&offset_i32)
            .arg(&out_strides_dev)
            .arg(&in_strides_dev)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA repeat_interleave launch failed: {e:?}")))?;
    }
    Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_repeat_interleave_backward(
    grad_out: &CudaStorage,
    repeats: usize,
    axis: usize,
    source_shape: &[usize],
) -> Result<CudaStorage> {
    let rank = source_shape.len();
    if axis >= rank {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::RepeatInterleave,
            reason: "repeat_interleave axis is outside the operand's rank",
        }));
    }
    let numel_in = crate::bytes::checked_numel(source_shape)?;
    let device_id = grad_out.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_repeat_interleave_backward")?;
    let stream = grad_out.buffer.device.default_stream();

    let in_contig = crate::layout::contiguous_strides(source_shape)
        .strides()
        .to_vec();
    let in_contig_dev = dev_i32_arg(&stream, &in_contig, "stride")?;
    let grad_strides_dev = dev_i32_arg(&stream, grad_out.strides.strides(), "stride")?;

    let byte_len = crate::bytes::byte_len(DTypeId::F32, numel_in, OperationKind::Storage)?;
    let mut out_buffer = CudaBuffer {
        len: numel_in,
        dtype: DTypeId::F32.descriptor(),
        data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
            Error::Msg(format!(
                "CUDA repeat_interleave backward allocation failed: {e:?}"
            ))
        })?),
        device: grad_out.buffer.device.clone(),
        device_id,
    };
    if numel_in == 0 {
        return Ok(CudaStorage::new(
            Arc::new(out_buffer),
            source_shape.to_vec(),
        ));
    }

    let numel_i32 = checked_i32(numel_in, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
    let repeats_i32 = checked_i32(repeats, "repeats")?;
    let axis_i32 = checked_i32(axis, "axis")?;
    let offset_i32 = checked_i32(grad_out.offset_elements, "grad offset")?;
    let block_size = 256u32;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: Launches the group-sum backward over a fresh allocation with
    // validated strides and offsets.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*grad_out.buffer.data)
            .arg(&mut *out_u8)
            .arg(&numel_i32)
            .arg(&rank_i32)
            .arg(&repeats_i32)
            .arg(&axis_i32)
            .arg(&offset_i32)
            .arg(&in_contig_dev)
            .arg(&grad_strides_dev)
            .launch(config)
            .map_err(|e| {
                Error::Msg(format!(
                    "CUDA repeat_interleave backward launch failed: {e:?}"
                ))
            })?;
    }
    Ok(CudaStorage::new(
        Arc::new(out_buffer),
        source_shape.to_vec(),
    ))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_one_hot(t: &CudaStorage, depth: usize) -> Result<CudaStorage> {
    if t.buffer.dtype.builtin_id() != Some(DTypeId::I64) {
        return Err(Error::UnsupportedDType {
            dtype: t.buffer.dtype,
            backend: "Cuda",
            op: "one_hot",
        });
    }
    let rank = t.shape.len();
    let numel = crate::bytes::checked_numel(&t.shape)?;
    let mut out_shape = t.shape.to_vec();
    out_shape.push(depth);
    let out_total = crate::bytes::checked_numel(&out_shape)?;

    let device_id = t.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_one_hot")?;
    let stream = t.buffer.device.default_stream();

    let in_contig = crate::layout::contiguous_strides(&t.shape)
        .strides()
        .to_vec();
    let in_contig_dev = dev_i32_arg(&stream, &in_contig, "stride")?;
    let in_strides_dev = dev_i32_arg(&stream, t.strides.strides(), "stride")?;

    let bool_dtype = DTypeId::Bool.descriptor();
    let mut out_buffer = CudaBuffer {
        len: out_total,
        dtype: bool_dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            bool_dtype,
            out_total,
            OperationKind::Storage,
        )?),
        device: t.buffer.device.clone(),
        device_id,
    };
    if numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), out_shape));
    }

    let numel_i32 = checked_i32(numel, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
    let depth_i32 = checked_i32(depth, "depth")?;
    let offset_i32 = checked_i32(t.offset_elements, "input offset")?;
    let block_size = 256u32;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: Launches one_hot over a fresh bool allocation with validated
    // strides and offsets.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*t.buffer.data)
            .arg(&mut *out_u8)
            .arg(&numel_i32)
            .arg(&rank_i32)
            .arg(&depth_i32)
            .arg(&offset_i32)
            .arg(&in_contig_dev)
            .arg(&in_strides_dev)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA one_hot launch failed: {e:?}")))?;
    }
    Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_bincount(t: &CudaStorage, bins: usize) -> Result<CudaStorage> {
    if bins == 0 {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::Bincount,
            reason: "bincount needs at least one bin to count into",
        }));
    }
    if t.buffer.dtype.builtin_id() != Some(DTypeId::I64) {
        return Err(Error::UnsupportedDType {
            dtype: t.buffer.dtype,
            backend: "Cuda",
            op: "bincount",
        });
    }
    let rank = t.shape.len();
    let numel = crate::bytes::checked_numel(&t.shape)?;
    let bins_i32_val = checked_i32(bins, "bins")?;

    let device_id = t.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_bincount")?;
    let stream = t.buffer.device.default_stream();

    let in_contig = crate::layout::contiguous_strides(&t.shape)
        .strides()
        .to_vec();
    let in_contig_dev = dev_i32_arg(&stream, &in_contig, "stride")?;
    let in_strides_dev = dev_i32_arg(&stream, t.strides.strides(), "stride")?;

    let byte_len = crate::bytes::byte_len(DTypeId::I64, bins, OperationKind::Storage)?;
    let mut counts_buffer =
        CudaBuffer {
            len: bins,
            dtype: DTypeId::I64.descriptor(),
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                Error::Msg(format!("CUDA bincount output allocation failed: {e:?}"))
            })?),
            device: t.buffer.device.clone(),
            device_id,
        };
    if numel == 0 {
        return Ok(CudaStorage::new(Arc::new(counts_buffer), alloc::vec![bins]));
    }

    let error_flag_dev = stream
        .alloc_zeros::<u32>(1)
        .map_err(|e| Error::Msg(format!("CUDA error flag allocation failed: {e:?}")))?;

    let numel_i32 = checked_i32(numel, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
    let offset_i32 = checked_i32(t.offset_elements, "input offset")?;
    let block_size = 256u32;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: Launches bincount with bounds-checked parameters and a device
    // error flag; counts is a fresh zeroed i64 allocation.
    unsafe {
        let out_u8 = Arc::get_mut(&mut counts_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*t.buffer.data)
            .arg(&mut *out_u8)
            .arg(&error_flag_dev)
            .arg(&numel_i32)
            .arg(&rank_i32)
            .arg(&bins_i32_val)
            .arg(&offset_i32)
            .arg(&in_contig_dev)
            .arg(&in_strides_dev)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA bincount launch failed: {e:?}")))?;
    }

    let mut host_err = [0u32; 1];
    stream
        .memcpy_dtoh(&error_flag_dev, &mut host_err)
        .map_err(|e| Error::Msg(format!("CUDA error flag readback failed: {e:?}")))?;
    if host_err[0] != 0 {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::Bincount,
            reason: "bincount index is not a whole number inside the bin range",
        }));
    }
    Ok(CudaStorage::new(Arc::new(counts_buffer), alloc::vec![bins]))
}

/// Deterministic `scatter_add` forward: builds the destination/source map,
/// then accumulates each output position in one fixed thread so the result
/// matches CPU bitwise. Returns `(output, scratch)`; the scratch pairs stay
/// alive for the tape entry's backward.
#[cfg(feature = "cuda")]
pub(crate) fn launch_scatter_add_ordered(
    input: &CudaStorage,
    axis: usize,
    index: &CudaStorage,
    src: &CudaStorage,
) -> Result<(CudaStorage, CudaStorage)> {
    let dtype_id = input.buffer.dtype.builtin_id();
    let dtype_code = match dtype_id {
        Some(DTypeId::F32) => 0i32,
        Some(DTypeId::I64) => 1i32,
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: input.buffer.dtype,
                backend: "Cuda",
                op: "scatter_add",
            });
        }
    };
    if src.buffer.dtype != input.buffer.dtype {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::ScatterAdd,
            reason: "scatter_add source operand must share the input's dtype",
        }));
    }
    if index.buffer.dtype.builtin_id() != Some(DTypeId::I64) {
        return Err(Error::UnsupportedDType {
            dtype: index.buffer.dtype,
            backend: "Cuda",
            op: "scatter_add",
        });
    }
    let rank = input.shape.len();
    if axis >= rank {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::ScatterAdd,
            reason: "scatter_add axis is outside the operand's rank",
        }));
    }
    if index.shape != input.shape || src.shape != input.shape {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::ScatterAdd,
            reason: "scatter_add expects the index and source operands to share the input's shape",
        }));
    }

    let out_shape = input.shape.to_vec();
    let out_numel = crate::bytes::checked_numel(&out_shape)?;
    let src_numel = crate::bytes::checked_numel(&src.shape)?;
    let device_id = input.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let map_fn = dispatcher.get_function("index_ops", "incin_cuda_scatter_add_map")?;
    let ord_fn = dispatcher.get_function("index_ops", "incin_cuda_scatter_add_ordered")?;
    let stream = input.buffer.device.default_stream();

    let out_contig = crate::layout::contiguous_strides(&out_shape)
        .strides()
        .to_vec();
    let out_contig_dev = dev_i32_arg(&stream, &out_contig, "stride")?;
    let in_strides_dev = dev_i32_arg(&stream, input.strides.strides(), "stride")?;
    let idx_contig_dev = dev_i32_arg(&stream, &out_contig, "stride")?;
    let idx_strides_dev = dev_i32_arg(&stream, index.strides.strides(), "stride")?;
    let src_strides_dev = dev_i32_arg(&stream, src.strides.strides(), "stride")?;
    let out_shape_dev = dev_i32_arg(&stream, &out_shape, "shape")?;

    // Scratch pairs `[dest; src_numel][src_phys; src_numel]`, zeroed so an
    // empty source leaves only consulted-free zeros (the ordered kernel's
    // inner loop runs `j < numel_src` times and never reads past it).
    let scratch_len = src_numel.max(1) * 2;
    let scratch_byte_len =
        crate::bytes::byte_len(DTypeId::I64, scratch_len, OperationKind::Storage)?;
    let scratch_buffer = CudaBuffer {
        len: scratch_len,
        dtype: DTypeId::I64.descriptor(),
        data: Arc::new(stream.alloc_zeros::<u8>(scratch_byte_len).map_err(|e| {
            Error::Msg(format!("CUDA scatter_add scratch allocation failed: {e:?}"))
        })?),
        device: input.buffer.device.clone(),
        device_id,
    };
    let scratch = CudaStorage::new(Arc::new(scratch_buffer), alloc::vec![scratch_len]);

    let out_byte_len =
        crate::bytes::byte_len(input.buffer.dtype, out_numel, OperationKind::Storage)?;
    let out_buffer = CudaBuffer {
        len: out_numel,
        dtype: input.buffer.dtype,
        data: Arc::new(stream.alloc_zeros::<u8>(out_byte_len).map_err(|e| {
            Error::Msg(format!("CUDA scatter_add output allocation failed: {e:?}"))
        })?),
        device: input.buffer.device.clone(),
        device_id,
    };

    let out_numel_i32 = checked_i32(out_numel, "element count")?;
    let src_numel_i32 = checked_i32(src_numel, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
    let axis_i32 = checked_i32(axis, "axis")?;
    let idx_offset_i32 = checked_i32(index.offset_elements, "index offset")?;
    let src_offset_i32 = checked_i32(src.offset_elements, "source offset")?;
    let in_offset_i32 = checked_i32(input.offset_elements, "input offset")?;
    let block_size = 256u32;

    // Pass 1 runs even when the output is empty, so every source position's
    // destination resolves to `-1` (an empty buffer accepts nothing) and the
    // tape's backward sees the drops; its grid covers the source, not the
    // output. With an empty source the zeros are never consulted: the ordered
    // kernel's inner loop runs `j < 0` times.
    if src_numel > 0 {
        let map_grid = (src_numel_i32 as u32).div_ceil(block_size);
        let map_config = cudarc::driver::LaunchConfig {
            grid_dim: (map_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: Launches the map kernel over the source with validated
        // strides/offsets, writing the scratch pair allocation.
        unsafe {
            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&map_fn)
                .arg(&*index.buffer.data)
                .arg(&*scratch.buffer.data)
                .arg(&src_numel_i32)
                .arg(&out_numel_i32)
                .arg(&rank_i32)
                .arg(&axis_i32)
                .arg(&idx_offset_i32)
                .arg(&src_offset_i32)
                .arg(&out_shape_dev)
                .arg(&idx_contig_dev)
                .arg(&idx_strides_dev)
                .arg(&src_strides_dev)
                .arg(&out_contig_dev)
                .launch(map_config)
                .map_err(|e| Error::Msg(format!("CUDA scatter_add map launch failed: {e:?}")))?;
        }
    }
    if out_numel == 0 {
        return Ok((CudaStorage::new(Arc::new(out_buffer), out_shape), scratch));
    }

    // Pass 2: one thread per output position.
    let ord_grid = (out_numel_i32 as u32).div_ceil(block_size);
    let ord_config = cudarc::driver::LaunchConfig {
        grid_dim: (ord_grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut out_buffer = out_buffer;
    // SAFETY: Launches the ordered accumulation over the fresh output with
    // validated strides/offsets; dtype_code was checked against the buffers.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&ord_fn)
            .arg(&*input.buffer.data)
            .arg(&*src.buffer.data)
            .arg(&mut *out_u8)
            .arg(&*scratch.buffer.data)
            .arg(&out_numel_i32)
            .arg(&src_numel_i32)
            .arg(&rank_i32)
            .arg(&dtype_code)
            .arg(&in_offset_i32)
            .arg(&out_contig_dev)
            .arg(&in_strides_dev)
            .launch(ord_config)
            .map_err(|e| Error::Msg(format!("CUDA scatter_add ordered launch failed: {e:?}")))?;
    }
    Ok((CudaStorage::new(Arc::new(out_buffer), out_shape), scratch))
}

/// `scatter_add`'s source cotangent: gather `grad_out` at each recorded
/// destination, zero where the forward dropped the write. The scratch pairs
/// come from [`launch_scatter_add_ordered`].
#[cfg(feature = "cuda")]
pub(crate) fn launch_scatter_add_backward_src(
    grad_out: &CudaStorage,
    scratch: &CudaStorage,
    source_shape: &[usize],
) -> Result<CudaStorage> {
    let rank = source_shape.len();
    let numel_src = crate::bytes::checked_numel(source_shape)?;
    let dtype_code = match grad_out.buffer.dtype.builtin_id() {
        Some(DTypeId::F32) => 0i32,
        Some(DTypeId::I64) => 1i32,
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: grad_out.buffer.dtype,
                backend: "Cuda",
                op: "scatter_add backward",
            });
        }
    };
    let device_id = grad_out.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_scatter_add_backward")?;
    let stream = grad_out.buffer.device.default_stream();

    let out_contig = crate::layout::contiguous_strides(source_shape)
        .strides()
        .to_vec();
    let out_contig_dev = dev_i32_arg(&stream, &out_contig, "stride")?;
    let grad_strides_dev = dev_i32_arg(&stream, grad_out.strides.strides(), "stride")?;

    let byte_len =
        crate::bytes::byte_len(grad_out.buffer.dtype, numel_src, OperationKind::Storage)?;
    let mut out_buffer = CudaBuffer {
        len: numel_src,
        dtype: grad_out.buffer.dtype,
        data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
            Error::Msg(format!(
                "CUDA scatter_add backward allocation failed: {e:?}"
            ))
        })?),
        device: grad_out.buffer.device.clone(),
        device_id,
    };
    if numel_src == 0 {
        return Ok(CudaStorage::new(
            Arc::new(out_buffer),
            source_shape.to_vec(),
        ));
    }

    let numel_i32 = checked_i32(numel_src, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
    let offset_i32 = checked_i32(grad_out.offset_elements, "grad offset")?;
    let block_size = 256u32;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: Launches the backward gather over a fresh allocation with
    // validated strides and offsets; scratch is the paired i64 scratch.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*grad_out.buffer.data)
            .arg(&mut *out_u8)
            .arg(&*scratch.buffer.data)
            .arg(&numel_i32)
            .arg(&rank_i32)
            .arg(&dtype_code)
            .arg(&offset_i32)
            .arg(&out_contig_dev)
            .arg(&grad_strides_dev)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA scatter_add backward launch failed: {e:?}")))?;
    }
    Ok(CudaStorage::new(
        Arc::new(out_buffer),
        source_shape.to_vec(),
    ))
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
        pad_before.push(checked_i32(before, "padding")?);
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

    let out_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&out_shape).strides(),
        "stride",
    )?;
    let in_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&input.shape).strides(),
        "stride",
    )?;
    let out_shape_i32 = checked_i32_vec(&out_shape, "shape")?;
    let in_shape_i32 = checked_i32_vec(&input.shape, "shape")?;

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
    // The grid is `u32` while the count is `usize`: converting with `as`
    // would wrap past `u32::MAX` and launch an undersized kernel over a
    // buffer the allocation already proved that large.
    let grid_size = checked_u32(out_numel, "element count")?.div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_numel_i32 = checked_i32(out_numel, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
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

    let out_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&out_shape).strides(),
        "stride",
    )?;
    let in_strides = checked_i32_vec(
        crate::layout::contiguous_strides(&input.shape).strides(),
        "stride",
    )?;
    let out_shape_i32 = checked_i32_vec(&out_shape, "shape")?;
    let in_shape_i32 = checked_i32_vec(&input.shape, "shape")?;

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
    // The grid is `u32` while the count is `usize`: converting with `as`
    // would wrap past `u32::MAX` and launch an undersized kernel over a
    // buffer the allocation already proved that large.
    let grid_size = checked_u32(out_numel, "element count")?.div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let out_numel_i32 = checked_i32(out_numel, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;

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

/// `repeat`'s cotangent: each source element's gradient is the sum over the
/// output tiles that copied it. See `incin_cuda_repeat_backward` for why the
/// tile walk is ordered the way it is and why the f64 sums are bitwise
/// CPU-identical.
#[cfg(feature = "cuda")]
pub(crate) fn launch_repeat_backward(
    grad_out: &CudaStorage,
    repeats: &[usize],
    source_shape: &[usize],
) -> Result<CudaStorage> {
    let rank = source_shape.len();
    if repeats.len() != rank {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::Repeat,
            reason: "repeat factors must match tensor rank",
        }));
    }
    let numel_in = crate::bytes::checked_numel(source_shape)?;
    let num_tiles = repeats.iter().try_fold(1usize, |acc, &factor| {
        acc.checked_mul(factor)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Repeat,
                expression: "CUDA repeat tile count",
            })
    })?;

    let device_id = grad_out.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_repeat_backward")?;
    let stream = grad_out.buffer.device.default_stream();

    let in_shape_dev = dev_i32_arg(&stream, source_shape, "shape")?;
    let repeats_dev = dev_i32_arg(&stream, repeats, "repeats")?;
    let grad_strides_dev = dev_i32_arg(&stream, grad_out.strides.strides(), "stride")?;

    let byte_len = crate::bytes::byte_len(DTypeId::F32, numel_in, OperationKind::Storage)?;
    let mut out_buffer =
        CudaBuffer {
            len: numel_in,
            dtype: DTypeId::F32.descriptor(),
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                Error::Msg(format!("CUDA repeat backward allocation failed: {e:?}"))
            })?),
            device: grad_out.buffer.device.clone(),
            device_id,
        };
    // An empty source, or an empty tile set (some factor is 0), leaves every
    // source gradient at zero - and keeps `cols`-style divisions inside the
    // kernel off the zero-extent path the same way the forward does.
    if numel_in == 0 || num_tiles == 0 {
        return Ok(CudaStorage::new(
            Arc::new(out_buffer),
            source_shape.to_vec(),
        ));
    }

    let numel_i32 = checked_i32(numel_in, "element count")?;
    let rank_i32 = checked_i32(rank, "rank")?;
    let num_tiles_i32 = checked_i32(num_tiles, "tile count")?;
    let offset_i32 = checked_i32(grad_out.offset_elements, "grad offset")?;
    let block_size = 256u32;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: Launches the tile-sum backward over a fresh allocation with
    // validated strides, offsets, and a tile count the fold above proved
    // non-overflowing.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*grad_out.buffer.data)
            .arg(&mut *out_u8)
            .arg(&numel_i32)
            .arg(&rank_i32)
            .arg(&num_tiles_i32)
            .arg(&offset_i32)
            .arg(&in_shape_dev)
            .arg(&repeats_dev)
            .arg(&grad_strides_dev)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA repeat backward launch failed: {e:?}")))?;
    }
    Ok(CudaStorage::new(
        Arc::new(out_buffer),
        source_shape.to_vec(),
    ))
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
        let out_dim = n
            .checked_add(diag_abs)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Diag,
                expression: "CUDA diagonal output dimension",
            })?;
        let out_shape = vec![out_dim, out_dim];
        let out_numel = out_dim
            .checked_mul(out_dim)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Diag,
                expression: "CUDA diagonal output element count",
            })?;
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
            // The grid is `u32` while the count is `usize`: converting with `as`
            // would wrap past `u32::MAX` and launch an undersized kernel over a
            // buffer the allocation already proved that large.
            let grid_size = checked_u32(out_numel, "element count")?.div_ceil(block_size);
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = checked_i32(n, "extent")?;
            let out_dim_i32 = checked_i32(out_dim, "extent")?;

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
            let grid_size = checked_u32(out_len, "element count")?.div_ceil(block_size);
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let rows_i32 = checked_i32(rows, "extent")?;
            let cols_i32 = checked_i32(cols, "extent")?;
            let out_len_i32 = checked_i32(out_len, "element count")?;

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

/// `diag`'s cotangent for the EXTRACT form: zeros over the rank-2 input with
/// the outgoing cotangent placed back on the extracted diagonal. The
/// CONSTRUCT form's adjoint is the extract itself and goes through
/// [`launch_diag`], keeping the two forms on one kernel and one
/// out-length rule.
#[cfg(feature = "cuda")]
pub(crate) fn launch_diag_backward(
    grad_out: &CudaStorage,
    diagonal: i32,
    input_shape: &[usize],
) -> Result<CudaStorage> {
    if input_shape.len() != 2 || grad_out.shape.len() != 1 {
        return Err(Error::ShapeMismatch {
            op: "diag",
            expected: vec![1, 2],
            got: vec![input_shape.len()],
            msg: "diag backward needs a rank-2 input and a rank-1 cotangent".into(),
        });
    }
    let cols = input_shape[1];
    let numel_in = crate::bytes::checked_numel(input_shape)?;
    let out_len = grad_out.shape[0];

    let device_id = grad_out.buffer.device_id;
    ensure_index_ops_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("index_ops", "incin_cuda_diag_backward")?;
    let stream = grad_out.buffer.device.default_stream();

    let grad_strides_dev = dev_i32_arg(&stream, grad_out.strides.strides(), "stride")?;

    let byte_len = crate::bytes::byte_len(DTypeId::F32, numel_in, OperationKind::Storage)?;
    let mut out_buffer = CudaBuffer {
        len: numel_in,
        dtype: DTypeId::F32.descriptor(),
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(byte_len)
                .map_err(|e| Error::Msg(format!("CUDA diag backward allocation failed: {e:?}")))?,
        ),
        device: grad_out.buffer.device.clone(),
        device_id,
    };
    // No cell to fill (empty input) or no cotangent to place: the zeros
    // already are the answer, and skipping the launch also keeps `idx /
    // cols` off a zero `cols` the way every other launcher here does.
    if numel_in == 0 || out_len == 0 {
        return Ok(CudaStorage::new(Arc::new(out_buffer), input_shape.to_vec()));
    }

    let numel_i32 = checked_i32(numel_in, "element count")?;
    let cols_i32 = checked_i32(cols, "extent")?;
    let out_len_i32 = checked_i32(out_len, "element count")?;
    let offset_i32 = checked_i32(grad_out.offset_elements, "grad offset")?;
    let block_size = 256u32;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: Launches the diagonal scatter over a fresh allocation with
    // validated extents and offsets; `numel_in > 0` proved `cols > 0`.
    unsafe {
        let out_u8 = Arc::get_mut(&mut out_buffer.data)
            .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*grad_out.buffer.data)
            .arg(&mut *out_u8)
            .arg(&numel_i32)
            .arg(&cols_i32)
            .arg(&out_len_i32)
            .arg(&diagonal)
            .arg(&offset_i32)
            .arg(&grad_strides_dev)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA diag backward launch failed: {e:?}")))?;
    }
    Ok(CudaStorage::new(Arc::new(out_buffer), input_shape.to_vec()))
}
