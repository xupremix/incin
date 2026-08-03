use super::alloc_zeroed_bytes;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use alloc::vec::Vec;
use incin_core::prelude::Result;
use incin_core::prelude::{OperationKind, ShapeBuf, ShapeError};

/// Packs the `[u32; 21]` params buffer `kernels/shape.cu`'s `shape_op` kernel
/// expects: `[op_mode, rank, n_elements, out_shape(6), inp_shape(6), aux(6)]`,
/// shapes right-aligned/padded with leading `1`s to a fixed rank-6 layout.
/// Direct port of `wgpu/dispatch.rs::prepare_shape_params` — same op_mode
/// values (0=narrow, 2=transpose, 3=broadcast), same `aux` semantics (narrow
/// start offsets, or transpose's per-output-dim source-dim map, offset by
/// the output's padding amount so it indexes correctly into `multi_idx`).
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

/// Shared launcher for `narrow`/`paste`/`transpose`/`broadcast_as` — all are
/// the same per-thread strided gather-or-scatter, differing only in how a
/// thread's index maps to an input/output flat offset (see `shape.cu`).
/// `launch_n` is the thread count: the output's element count for
/// narrow/transpose/broadcast, but the (smaller) *input*'s element count
/// for paste, which scatters into a larger, pre-zeroed output — see
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

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("shape", "shape_op")?;
    let stream = t_buf.device.default_stream();

    let n_elements: usize = incin_core::prelude::ShapeBuf::from_slice(&(out_shape))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
    let launch_n_u32 = crate::cuda::checked_u32(launch_n, "CUDA shape-op grid dimension")?;
    let params = prepare_shape_params(op_mode, launch_n_u32, &out_shape, &t.shape, aux)?;
    let params_u8: &[u8] = bytemuck::cast_slice(&params);
    let params_dev = stream.clone_htod(params_u8).map_err(|e| {
        incin_core::prelude::Error::Msg(format!("shape params upload failed: {e:?}"))
    })?;

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

    unsafe {
        let in_f32 = t_buf.data.transmute::<f32>(t_buf.len).unwrap();
        let params_f32 = params_dev.transmute::<u32>(21).unwrap();
        // out_b.data was allocated once immediately above and never cloned,
        // so it stays uniquely owned (refcount 1) here — Arc::get_mut
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
            .map_err(|e| {
                incin_core::prelude::Error::Msg(format!("shape_op launch failed: {e:?}"))
            })?;
    }

    let strides = crate::cpu::stride::contiguous_strides(&out_shape);
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0)
}

/// Narrows dimension `dim` of `t` to the half-open range `[start, start+len)`.
/// Materializes a fresh contiguous buffer (unlike CPU's metadata-only
/// `narrow`) — CUDA's elementwise/matmul/reduce kernels read flat contiguous
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

/// Broadcasts `t` to `target_shape`. Materializes (see `launch_narrow`'s doc
/// for why). Caller must validate shape compatibility first — this function
/// assumes `target_shape` is already a legal broadcast target of `t.shape`.
#[cfg(feature = "cuda")]
pub(crate) fn launch_broadcast(t: &CudaStorage, target_shape: &[usize]) -> Result<CudaStorage> {
    let launch_n = ShapeBuf::from_slice(target_shape).checked_numel(OperationKind::Broadcast)?;
    launch_shape_op(3, t, target_shape.to_vec(), &[], launch_n)
}

/// Scatters `values` into a fresh, zero-initialized buffer of shape
/// `original_shape` at `region_start` — `narrow`'s backward (the gradient
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
        return Err(incin_core::prelude::Error::Msg(
            "concat: empty tensor list".into(),
        ));
    }

    let first_buf = &*tensors[0].buffer;
    let device_id = first_buf.device_id;
    ensure_concat_loaded(device_id)?;

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("concat", "concat_f32")?;
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

    let outer_size: usize = incin_core::prelude::ShapeBuf::from_slice(&(out_shape[0..dim]))
        .checked_numel(incin_core::prelude::OperationKind::Storage)?;
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
                    incin_core::prelude::Error::Msg(format!("concat launch failed: {e:?}"))
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

    let strides = crate::cpu::stride::contiguous_strides(&out_shape);
    CudaStorage::try_from_parts(Arc::new(out_b), out_shape, strides, 0)
}
