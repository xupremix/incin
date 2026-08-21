//! Elementwise numeric comparison, producing `bool` storage.
//!
//! Deliberately not built on the generic pointwise machinery
//! (`launch_binary_op`/`render_binary_strategy`/`CudaScalarSpec`): that
//! pipeline always writes its output in the same dtype it read, and giving
//! it a genuinely different output dtype would mean touching every other
//! caller of the packed/tuned strategy selection to keep them from picking
//! it up by accident. A dedicated kernel, compiled once and cached the same
//! way `concat`/`shape_op` are, keeps that machinery untouched and keeps
//! this one narrow enough to read start to finish.
//!
//! Only `f32`, and only the contiguous, identically-shaped case: the
//! executor broadcasts both operands to a common shape with
//! `CudaBackendImpl::broadcast_as` first, which always materializes a fresh
//! contiguous buffer, so this launcher never has to reason about strides or
//! offsets the way the general elementwise path does. Widening past `f32`
//! would mean widening `kernels/compare.cu`'s hardcoded `float*` signature
//! first, the same `bytemuck`-style byte-width trap this session already
//! found and fixed for `broadcast_as` itself.

use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DTypeId;

#[cfg(feature = "cuda")]
const COMPARE_SRC: &str = include_str!("kernels/compare.cu");

/// The six numeric comparisons this module answers, in the exact order
/// `kernels/compare.cu`'s `op_mode` switch expects.
#[derive(Clone, Copy)]
pub(crate) enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    const fn mode(self) -> i32 {
        match self {
            Self::Eq => 0,
            Self::Ne => 1,
            Self::Lt => 2,
            Self::Le => 3,
            Self::Gt => 4,
            Self::Ge => 5,
        }
    }
}

#[cfg(feature = "cuda")]
fn ensure_compare_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "compare").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel("compare", COMPARE_SRC, "compare")?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_compare(
    op: CompareOp,
    lhs: &CudaStorage,
    rhs: &CudaStorage,
) -> Result<CudaStorage> {
    if lhs.shape != rhs.shape {
        return Err(Error::Msg(format!(
            "CUDA compare requires identically-shaped operands after broadcast; got {:?} vs {:?}",
            lhs.shape, rhs.shape
        )));
    }
    let (lhs_b, rhs_b) = (&*lhs.buffer, &*rhs.buffer);
    if lhs_b.dtype != rhs_b.dtype {
        return Err(Error::DTypeStorageMismatch {
            expected: lhs_b.dtype,
            got: rhs_b.dtype,
        });
    }
    crate::cuda::backend::cuda_require_f32(lhs_b.dtype, "compare")?;
    if lhs_b.device_id != rhs_b.device_id {
        return Err(Error::DeviceMismatch {
            left: incin_core::tensor::device::DeviceId::cuda(lhs_b.device_id),
            right: incin_core::tensor::device::DeviceId::cuda(rhs_b.device_id),
        });
    }

    let numel = crate::bytes::checked_numel(&lhs.shape)?;
    let device_id = lhs_b.device_id;
    ensure_compare_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("compare", "compare_op")?;
    let stream = lhs_b.device.default_stream();

    let bool_dtype = DTypeId::Bool.descriptor();
    let mut out_b = CudaBuffer {
        len: numel,
        dtype: bool_dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            bool_dtype,
            numel,
            OperationKind::Pointwise,
        )?),
        device: lhs_b.device.clone(),
        device_id,
    };

    if numel == 0 {
        return Ok(CudaStorage::new(Arc::new(out_b), lhs.shape.to_vec()));
    }

    let numel_i32 = crate::cuda::checked_i32(numel, "element count")?;
    let mode = op.mode();
    let block_size: u32 = 256;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: numel and launch dimensions were checked above; out_b is freshly
    // allocated, so its unique u8 storage can be passed to this typed kernel.
    unsafe {
        let out_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .ok_or_else(|| {
                Error::Msg("fresh CUDA compare output was unexpectedly shared".into())
            })?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*lhs_b.data)
            .arg(&*rhs_b.data)
            .arg(&mut *out_slice)
            .arg(&mode)
            .arg(&numel_i32)
            .launch(config)
            .map_err(|e| Error::Msg(format!("CUDA compare launch failed: {e:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(out_b), lhs.shape.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_op_modes_match_the_kernel_switch_order() {
        assert_eq!(CompareOp::Eq.mode(), 0);
        assert_eq!(CompareOp::Ne.mode(), 1);
        assert_eq!(CompareOp::Lt.mode(), 2);
        assert_eq!(CompareOp::Le.mode(), 3);
        assert_eq!(CompareOp::Gt.mode(), 4);
        assert_eq!(CompareOp::Ge.mode(), 5);
    }
}
