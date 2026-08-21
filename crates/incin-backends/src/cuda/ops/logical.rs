//! Logical connectives over `bool` storage: `LogicalAnd`/`LogicalOr`/`LogicalNot`.
//!
//! Same rationale as `compare`/`select` for a dedicated kernel: these read
//! and write `bool` throughout, one byte per element, which the generic
//! pointwise pipeline (float-typed) cannot answer. Unlike `compare`
//! (f32 -> bool) or `select` (bool + f32 -> f32), every operand here is
//! `bool`, so the capability row is `Bool`-only - no dtype union needed the
//! way `where_cond`/`masked_fill`'s `F32_AND_BOOL` was.
//!
//! Broadcasting the two binary operands to one shape is the caller's job
//! (the executor), the same precondition `compare`/`select` state; the
//! caller must broadcast through `cuda::ops::select::launch_broadcast_bool_mask`
//! rather than `shape::launch_broadcast`, for the same byte-width reason.

use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DTypeId;

#[cfg(feature = "cuda")]
const LOGICAL_SRC: &str = include_str!("kernels/logical.cu");

#[cfg(feature = "cuda")]
fn ensure_logical_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "logical").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel("logical", LOGICAL_SRC, "logical")?;
    }
    Ok(())
}

fn require_bool(storage: &CudaStorage, operation: &'static str) -> Result<()> {
    if storage.buffer.dtype.builtin_id() != Some(DTypeId::Bool) {
        return Err(Error::Msg(format!(
            "CUDA {operation} requires bool storage, got {:?}",
            storage.buffer.dtype
        )));
    }
    Ok(())
}

fn alloc_bool_output(
    stream: &alloc::sync::Arc<cudarc::driver::CudaStream>,
    template: &CudaStorage,
    numel: usize,
) -> Result<CudaBuffer> {
    let bool_dtype = DTypeId::Bool.descriptor();
    Ok(CudaBuffer {
        len: numel,
        dtype: bool_dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            stream,
            bool_dtype,
            numel,
            OperationKind::Pointwise,
        )?),
        device: template.buffer.device.clone(),
        device_id: template.buffer.device_id,
    })
}

#[cfg(feature = "cuda")]
fn launch_logical_binary(
    entry_point: &'static str,
    lhs: &CudaStorage,
    rhs: &CudaStorage,
) -> Result<CudaStorage> {
    if lhs.shape != rhs.shape {
        return Err(Error::Msg(format!(
            "CUDA {entry_point} requires identically-shaped operands after broadcast; got {:?} vs {:?}",
            lhs.shape, rhs.shape
        )));
    }
    require_bool(lhs, entry_point)?;
    require_bool(rhs, entry_point)?;
    if lhs.buffer.device_id != rhs.buffer.device_id {
        return Err(Error::DeviceMismatch {
            left: incin_core::tensor::device::DeviceId::cuda(lhs.buffer.device_id),
            right: incin_core::tensor::device::DeviceId::cuda(rhs.buffer.device_id),
        });
    }

    let numel = crate::bytes::checked_numel(&lhs.shape)?;
    let device_id = lhs.buffer.device_id;
    ensure_logical_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("logical", entry_point)?;
    let stream = lhs.buffer.device.default_stream();

    let mut out_b = alloc_bool_output(&stream, lhs, numel)?;
    if numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_b), lhs.shape.to_vec()));
    }

    let numel_i32 = crate::cuda::checked_i32(numel, "element count")?;
    let block_size: u32 = 256;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: numel is checked for the launch and out_b is a fresh unique
    // allocation, so the kernel receives valid input/output device views.
    unsafe {
        let out_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "fresh CUDA {entry_point} output was unexpectedly shared"
                ))
            })?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*lhs.buffer.data)
            .arg(&*rhs.buffer.data)
            .arg(&mut *out_slice)
            .arg(&numel_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA {entry_point} launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_b), lhs.shape.to_vec()))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_logical_and(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    launch_logical_binary("logical_and_op", lhs, rhs)
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_logical_or(lhs: &CudaStorage, rhs: &CudaStorage) -> Result<CudaStorage> {
    launch_logical_binary("logical_or_op", lhs, rhs)
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_logical_not(input: &CudaStorage) -> Result<CudaStorage> {
    require_bool(input, "logical_not_op")?;

    let numel = crate::bytes::checked_numel(&input.shape)?;
    let device_id = input.buffer.device_id;
    ensure_logical_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("logical", "logical_not_op")?;
    let stream = input.buffer.device.default_stream();

    let mut out_b = alloc_bool_output(&stream, input, numel)?;
    if numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_b), input.shape.to_vec()));
    }

    let numel_i32 = crate::cuda::checked_i32(numel, "element count")?;
    let block_size: u32 = 256;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: logical_not validates numel and launch dimensions; the fresh
    // output allocation is uniquely owned for the duration of this launch.
    unsafe {
        let out_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .ok_or_else(|| {
                Error::Msg("fresh CUDA logical_not_op output was unexpectedly shared".into())
            })?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*input.buffer.data)
            .arg(&mut *out_slice)
            .arg(&numel_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA logical_not_op launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_b), input.shape.to_vec()))
}
