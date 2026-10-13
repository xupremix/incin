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
//! Only the contiguous, identically-shaped case, and every float
//! storage dtype the kernel exports an entry point for
//! (`f32`/`f64`/`f16`/`bf16`, issue #90 slice 4), selected by the
//! fail-closed map below so a buffer can never reach a kernel typed for a
//! different dtype - the same shape `select.rs` took in slice 3. The
//! executor broadcasts both operands to a common shape with
//! `shape::launch_broadcast` first, which is width-parametric and moves
//! bytes for every dtype, so this launcher never has to reason about
//! strides or offsets the way the general elementwise path does.
//!
//! The WGPU twin (`wgpu/backend/compare.rs`, f32 re-checked by name) and
//! Metal (no comparison kernel at all) stay exactly as they are: widening
//! them is out of scope, and their `F32_ONLY` rows keep stating it.

use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

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

/// The kernel entry point for an operand dtype, or a typed refusal for
/// anything `compare.cu` does not export a name for. Fail-closed by
/// construction: the launcher launches exactly this name, so a dtype
/// without an entry can never reach a buffer reinterpreted as the wrong
/// type. Same contract as `where_cond_entry_point` in
/// `cuda/ops/select.rs`.
#[cfg(feature = "cuda")]
fn compare_entry_point(dtype: DTypeDescriptor) -> Result<&'static str> {
    match dtype.builtin_id() {
        Some(DTypeId::F32) => Ok("compare_op"),
        Some(DTypeId::F64) => Ok("compare_op_f64"),
        Some(DTypeId::F16) => Ok("compare_op_f16"),
        Some(DTypeId::BF16) => Ok("compare_op_bf16"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Cuda",
            op: "compare",
        }),
    }
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
    // Selects the entry point and refuses every dtype without one, before
    // any buffer is reinterpreted.
    let entry_point = compare_entry_point(lhs_b.dtype)?;
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
    let function = dispatcher.get_function("compare", entry_point)?;
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
    // The operand views pass through as byte buffers - the driver only needs
    // the device address, the kernel is typed by the entry point, and
    // `compare_entry_point` above established that the entry's type really
    // is these buffers' dtype.
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

/// Pure entry-point selection: runs wherever the `cuda` feature is on,
/// with no device required. The launcher launches exactly the name
/// `compare_entry_point` returns, so proving the mapping here is proving
/// a buffer can never reach a kernel typed for a different dtype.
#[cfg(all(test, feature = "cuda"))]
mod compare_entry_point_tests {
    use super::compare_entry_point;
    use incin_core::error::Error;
    use incin_core::tensor::dtype::DTypeId;

    #[test]
    fn compare_entry_point_maps_each_float_and_refuses_the_rest() {
        assert_eq!(
            compare_entry_point(DTypeId::F32.descriptor()).unwrap(),
            "compare_op"
        );
        assert_eq!(
            compare_entry_point(DTypeId::F64.descriptor()).unwrap(),
            "compare_op_f64"
        );
        assert_eq!(
            compare_entry_point(DTypeId::F16.descriptor()).unwrap(),
            "compare_op_f16"
        );
        assert_eq!(
            compare_entry_point(DTypeId::BF16.descriptor()).unwrap(),
            "compare_op_bf16"
        );

        for id in [
            DTypeId::I64,
            DTypeId::Bool,
            DTypeId::U8,
            DTypeId::U32,
            DTypeId::Q8_0,
        ] {
            let error = match compare_entry_point(id.descriptor()) {
                Ok(name) => panic!("{id:?} must be refused, got entry point {name:?}"),
                Err(error) => error,
            };
            assert!(
                matches!(
                    error,
                    Error::UnsupportedDType {
                        backend: "Cuda",
                        op: "compare",
                        ..
                    }
                ),
                "{id:?}: expected UnsupportedDType naming Cuda/compare, got {error:?}"
            );
        }
    }
}
