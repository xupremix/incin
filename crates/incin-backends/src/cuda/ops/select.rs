//! Mask-driven selection, the counterpart to `compare`'s mask *producers*.
//!
//! Same rationale as `compare` for being a dedicated kernel rather than a
//! trip through the generic pointwise pipeline: `where_cond` reads three
//! operands of two different dtypes (a `bool` mask, two `float` values), and
//! `masked_fill` reads a `bool` mask beside a `float` operand and a scalar
//! attribute — neither shape fits `launch_binary_op`, which reads and writes
//! exactly two operands of one shared dtype. Both kernels are compiled once
//! from `kernels/select.cu` and cached under the module name `select`, same
//! pattern as `concat`/`shape_op`/`compare`.
//!
//! Only `f32` for the value operands, and only the contiguous,
//! identically-shaped case, for the same byte-width reason `compare`
//! documents: `kernels/select.cu` reads every value pointer as a hardcoded
//! `float*`.

use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DTypeId;

#[cfg(feature = "cuda")]
const SELECT_SRC: &str = include_str!("kernels/select.cu");

#[cfg(feature = "cuda")]
fn ensure_select_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "select").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel("select", SELECT_SRC, "select")?;
    }
    Ok(())
}

/// Every operand's shape, dtype and device must already agree exactly: the
/// caller (the executor, for `where_cond`; the descriptor contract itself,
/// for `masked_fill`) is responsible for broadcasting before either launcher
/// below is called.
fn require_bool_mask(mask: &CudaStorage, operation: &'static str) -> Result<()> {
    if mask.buffer.dtype.builtin_id() != Some(DTypeId::Bool) {
        return Err(Error::Msg(format!(
            "CUDA {operation} requires a bool mask, got {:?}",
            mask.buffer.dtype
        )));
    }
    Ok(())
}

fn require_same_device(a: &CudaStorage, b: &CudaStorage) -> Result<()> {
    if a.buffer.device_id != b.buffer.device_id {
        return Err(Error::DeviceMismatch {
            left: incin_core::tensor::device::DeviceId::cuda(a.buffer.device_id),
            right: incin_core::tensor::device::DeviceId::cuda(b.buffer.device_id),
        });
    }
    Ok(())
}

/// Broadcasts a `bool` mask to `target_shape`.
///
/// Not `crate::cuda::ops::shape::launch_broadcast`: that launches
/// `shape.cu`'s `shape_op`, whose data pointers are a hardcoded
/// `float*`/`float*` — the same byte-width assumption this session already
/// narrowed CUDA's `broadcast`/`broadcast_training` capability rows to
/// `F32_ONLY` over. Narrowing the row stops `BroadcastAs` from reaching it
/// on a non-`f32` dtype; it does nothing about a direct internal call like
/// `where_cond`'s own mask broadcast, which is exactly why this function
/// exists instead of reusing that one. It reuses
/// `crate::cuda::ops::shape::prepare_shape_params` for the index arithmetic
/// (identical to `shape_op`'s `op_mode == 3`) and launches `select.cu`'s
/// `broadcast_bool_op`, the `unsigned char` port of that case, against the
/// mask's own raw byte buffer directly — a `bool` element is already one
/// byte, so unlike `shape_op`'s `f32` path there is no `transmute` involved
/// on either data pointer, only on the uploaded `params` buffer.
#[cfg(feature = "cuda")]
pub(crate) fn launch_broadcast_bool_mask(
    mask: &CudaStorage,
    target_shape: &[usize],
) -> Result<CudaStorage> {
    require_bool_mask(mask, "where_cond")?;

    let mask_b = &*mask.buffer;
    let device_id = mask_b.device_id;
    ensure_select_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("select", "broadcast_bool_op")?;
    let stream = mask_b.device.default_stream();

    let n_elements = crate::bytes::checked_numel(target_shape)?;
    let n_elements_u32 = crate::cuda::checked_u32(n_elements, "CUDA broadcast-mask element count")?;
    let params = crate::cuda::ops::shape::prepare_shape_params(
        3,
        n_elements_u32,
        target_shape,
        &mask.shape,
        &[],
    )?;
    let params_u8: &[u8] = bytemuck::cast_slice(&params);
    let params_dev = stream
        .clone_htod(params_u8)
        .map_err(|e| Error::Msg(format!("broadcast-mask params upload failed: {e:?}")))?;

    let bool_dtype = DTypeId::Bool.descriptor();
    let mut out_b = CudaBuffer {
        len: n_elements,
        dtype: bool_dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            bool_dtype,
            n_elements,
            OperationKind::Broadcast,
        )?),
        device: mask_b.device.clone(),
        device_id,
    };

    if n_elements == 0 {
        return Ok(CudaStorage::new(Arc::new(out_b), target_shape.to_vec()));
    }

    let block_size: u32 = 256;
    let grid_size = n_elements_u32.div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let params_u32 = params_dev.transmute::<u32>(21).ok_or_else(|| {
            Error::Msg("broadcast-mask params buffer was not 21 u32s wide".into())
        })?;
        let out_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .ok_or_else(|| {
                Error::Msg("fresh CUDA broadcast-mask output was unexpectedly shared".into())
            })?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*mask_b.data)
            .arg(&mut *out_slice)
            .arg(&params_u32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA broadcast-mask launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_b), target_shape.to_vec()))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_where_cond(
    mask: &CudaStorage,
    on_true: &CudaStorage,
    on_false: &CudaStorage,
) -> Result<CudaStorage> {
    if mask.shape != on_true.shape || mask.shape != on_false.shape {
        return Err(Error::Msg(format!(
            "CUDA where_cond requires identically-shaped operands after broadcast; got mask {:?}, on_true {:?}, on_false {:?}",
            mask.shape, on_true.shape, on_false.shape
        )));
    }
    require_bool_mask(mask, "where_cond")?;
    crate::cuda::backend::cuda_require_f32(on_true.buffer.dtype, "where_cond")?;
    crate::cuda::backend::cuda_require_f32(on_false.buffer.dtype, "where_cond")?;
    require_same_device(mask, on_true)?;
    require_same_device(mask, on_false)?;

    let numel = crate::bytes::checked_numel(&mask.shape)?;
    let device_id = mask.buffer.device_id;
    ensure_select_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("select", "where_cond_op")?;
    let stream = mask.buffer.device.default_stream();

    let f32_dtype = DTypeId::F32.descriptor();
    let mut out_b = CudaBuffer {
        len: numel,
        dtype: f32_dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            f32_dtype,
            numel,
            OperationKind::Pointwise,
        )?),
        device: mask.buffer.device.clone(),
        device_id,
    };

    if numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_b), mask.shape.to_vec()));
    }

    let numel_i32 = crate::cuda::checked_i32(numel, "element count")?;
    let block_size: u32 = 256;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let out_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .ok_or_else(|| {
                Error::Msg("fresh CUDA where_cond output was unexpectedly shared".into())
            })?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*mask.buffer.data)
            .arg(&*on_true.buffer.data)
            .arg(&*on_false.buffer.data)
            .arg(&mut *out_slice)
            .arg(&numel_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA where_cond launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_b), mask.shape.to_vec()))
}

/// No tape entry is pushed here, matching `cpu::ops::shape_ops::masked_fill_storage`
/// exactly: neither backend's `masked_fill` currently routes a gradient back
/// to `input`, so a masked-fill result is a leaf on the tape on both. That is
/// an existing, pre-`compare`/`select` gap shared by both backends rather
/// than something this module introduces — fixing it would mean adding the
/// same backward (`grad_input = masked_fill(grad_out, mask, 0.0)`) to CPU
/// too, which is separate work, not a CUDA-only patch that would leave the
/// two backends disagreeing on what `masked_fill` differentiates through.
#[cfg(feature = "cuda")]
pub(crate) fn launch_masked_fill(
    input: &CudaStorage,
    mask: &CudaStorage,
    value: f64,
) -> Result<CudaStorage> {
    if input.shape != mask.shape {
        return Err(Error::Msg(format!(
            "CUDA masked_fill requires input and mask to share one shape; got input {:?}, mask {:?}",
            input.shape, mask.shape
        )));
    }
    require_bool_mask(mask, "masked_fill")?;
    crate::cuda::backend::cuda_require_f32(input.buffer.dtype, "masked_fill")?;
    require_same_device(mask, input)?;

    let numel = crate::bytes::checked_numel(&input.shape)?;
    let device_id = input.buffer.device_id;
    ensure_select_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("select", "masked_fill_op")?;
    let stream = input.buffer.device.default_stream();

    let f32_dtype = DTypeId::F32.descriptor();
    let mut out_b = CudaBuffer {
        len: numel,
        dtype: f32_dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            f32_dtype,
            numel,
            OperationKind::Pointwise,
        )?),
        device: input.buffer.device.clone(),
        device_id,
    };

    if numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_b), input.shape.to_vec()));
    }

    let numel_i32 = crate::cuda::checked_i32(numel, "element count")?;
    let value_f32 = value as f32;
    let block_size: u32 = 256;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let out_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .ok_or_else(|| {
                Error::Msg("fresh CUDA masked_fill output was unexpectedly shared".into())
            })?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*input.buffer.data)
            .arg(&*mask.buffer.data)
            .arg(&value_f32)
            .arg(&mut *out_slice)
            .arg(&numel_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA masked_fill launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_b), input.shape.to_vec()))
}
