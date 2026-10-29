//! Mask-driven selection, the counterpart to `compare`'s mask *producers*.
//!
//! Same rationale as `compare` for being a dedicated kernel rather than a
//! trip through the generic pointwise pipeline: `where_cond` reads three
//! operands of two different dtypes (a `bool` mask, two float values), and
//! `masked_fill` reads a `bool` mask beside a float operand and a scalar
//! attribute - neither shape fits `launch_binary_op`, which reads and writes
//! exactly two operands of one shared dtype. Both kernels are compiled once
//! from `kernels/select.cu` and cached under the module name `select`, same
//! pattern as `concat`/`shape_op`/`compare`.
//!
//! Every float storage dtype the kernel exports an entry point for
//! (`f32`/`f64`/`f16`/`bf16`, issue #90 slice 3), selected by the
//! fail-closed maps below so a buffer can never reach a kernel typed for a
//! different dtype. Only the contiguous, identically-shaped case: the
//! executor broadcasts every operand to one shared shape first, the same
//! precondition `compare` documents.

use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

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

/// Zero-filled contiguous storage in `sample`'s dtype on `sample`'s device.
///
/// The `where_cond` backward needs a zeros operand in the *value* dtype, not
/// `f32`: all-zero bytes are `+0.0` in every float encoding, so one
/// zeroed allocation serves `f32`/`f64`/`f16`/`bf16` alike without a
/// host fill per dtype.
#[cfg(feature = "cuda")]
pub(crate) fn zeros_like(sample: &CudaStorage, shape: &[usize]) -> Result<CudaStorage> {
    let numel = crate::bytes::checked_numel(shape)?;
    let stream = sample.buffer.device.default_stream();
    let data = crate::cuda::ops::alloc_zeroed_bytes(
        &stream,
        sample.buffer.dtype,
        numel,
        OperationKind::Storage,
    )?;
    let buffer = CudaBuffer {
        len: numel,
        dtype: sample.buffer.dtype,
        data: Arc::new(data),
        device: sample.buffer.device.clone(),
        device_id: sample.buffer.device_id,
    };
    Ok(CudaStorage::new(Arc::new(buffer), shape.to_vec()))
}

/// The kernel entry point for a value dtype, or a typed refusal for anything
/// `select.cu` does not export a `where_cond` name for. Fail-closed by
/// construction: the launcher launches exactly this name, so a dtype without
/// an entry can never reach a buffer reinterpreted as the wrong type. Same
/// contract as `matmul_entry_point` in `cuda/ops/matmul.rs`.
#[cfg(feature = "cuda")]
fn where_cond_entry_point(dtype: DTypeDescriptor) -> Result<&'static str> {
    match dtype.builtin_id() {
        Some(DTypeId::F32) => Ok("where_cond_op"),
        Some(DTypeId::F64) => Ok("where_cond_op_f64"),
        Some(DTypeId::F16) => Ok("where_cond_op_f16"),
        Some(DTypeId::BF16) => Ok("where_cond_op_bf16"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Cuda",
            op: "where_cond",
        }),
    }
}

/// The kernel entry point for a value dtype, or a typed refusal for anything
/// `select.cu` does not export a `masked_fill` name for. Same fail-closed
/// contract as [`where_cond_entry_point`].
#[cfg(feature = "cuda")]
fn masked_fill_entry_point(dtype: DTypeDescriptor) -> Result<&'static str> {
    match dtype.builtin_id() {
        Some(DTypeId::F32) => Ok("masked_fill_op"),
        Some(DTypeId::F64) => Ok("masked_fill_op_f64"),
        Some(DTypeId::F16) => Ok("masked_fill_op_f16"),
        Some(DTypeId::BF16) => Ok("masked_fill_op_bf16"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Cuda",
            op: "masked_fill",
        }),
    }
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
    // One typed kernel reads both value operands, so a mixed pair is refused
    // rather than reinterpreted - the same `DTypeMismatch` contract
    // `launch_matmul` keeps.
    if on_true.buffer.dtype != on_false.buffer.dtype {
        return Err(Error::DTypeMismatch {
            operation: "where_cond",
            expected: on_true.buffer.dtype,
            actual: on_false.buffer.dtype,
        });
    }
    // Selects the entry point and refuses every dtype without one, before
    // any buffer is reinterpreted.
    let entry_point = where_cond_entry_point(on_true.buffer.dtype)?;
    require_same_device(mask, on_true)?;
    require_same_device(mask, on_false)?;

    let numel = crate::bytes::checked_numel(&mask.shape)?;
    let device_id = mask.buffer.device_id;
    ensure_select_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("select", entry_point)?;
    let stream = mask.buffer.device.default_stream();

    let value_dtype = on_true.buffer.dtype;
    let mut out_b = CudaBuffer {
        len: numel,
        dtype: value_dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            value_dtype,
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

    // SAFETY: checked broadcast shape and numel bound kernel views; out_b is
    // fresh, so Arc::get_mut supplies the only mutable device handle. The
    // views pass through as byte buffers - the driver only needs the device
    // address, the kernel is typed by the entry point, and
    // `where_cond_entry_point` above established that the entry's type really
    // is this buffer's dtype.
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

/// No tape entry is pushed here by design, not by omission: `launch_where_cond`
/// follows the same split, and the `Execute<op::MaskedFill>` implementation in
/// `cuda/executor.rs` pushes the backward (`grad_input = masked_fill(grad_out,
/// mask, 0.0)`, mirroring CPU's `masked_fill_storage`) around this launch.
/// Keeping the launch itself tape-free is what lets both the forward-only and
/// the tracked call sites share it.
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
    // Selects the entry point and refuses every dtype without one, before
    // any buffer is reinterpreted.
    let entry_point = masked_fill_entry_point(input.buffer.dtype)?;
    require_same_device(mask, input)?;

    let numel = crate::bytes::checked_numel(&input.shape)?;
    let device_id = input.buffer.device_id;
    ensure_select_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("select", entry_point)?;
    let stream = input.buffer.device.default_stream();

    let value_dtype = input.buffer.dtype;
    let mut out_b = CudaBuffer {
        len: numel,
        dtype: value_dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            value_dtype,
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
    let block_size: u32 = 256;
    let grid_size = (numel_i32 as u32).div_ceil(block_size);
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // The scalar at the kernel's own precision: `f32`/`f64` narrow from the
    // `f64` attribute directly, while the half types round once through
    // `f32` and pass their two-byte encoding - the driver needs the value
    // width to match the entry point's parameter exactly.
    //
    // SAFETY: checked mask/input/output shapes and numel bound this launch;
    // out_b is freshly allocated and uniquely owned.
    macro_rules! launch_with_scalar {
        ($scalar:ident, $out:ident) => {{
            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&function)
                .arg(&*input.buffer.data)
                .arg(&*mask.buffer.data)
                .arg(&$scalar)
                .arg(&mut *$out)
                .arg(&numel_i32)
                .launch(config)
                .map_err(|e| Error::Msg(format!("CUDA masked_fill launch failed: {e:?}")))?;
        }};
    }
    // SAFETY: the output buffer was allocated fresh for this launch, so it has
    // no other owners and `Arc::get_mut` either succeeds or fails closed.
    unsafe {
        let out_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .ok_or_else(|| {
                Error::Msg("fresh CUDA masked_fill output was unexpectedly shared".into())
            })?;
        match value_dtype.builtin_id() {
            Some(DTypeId::F32) => {
                let scalar = value as f32;
                launch_with_scalar!(scalar, out_slice);
            }
            Some(DTypeId::F64) => {
                let scalar = value;
                launch_with_scalar!(scalar, out_slice);
            }
            Some(DTypeId::F16) => {
                let scalar = half::f16::from_f32(value as f32).to_bits();
                launch_with_scalar!(scalar, out_slice);
            }
            Some(DTypeId::BF16) => {
                let scalar = half::bf16::from_f32(value as f32).to_bits();
                launch_with_scalar!(scalar, out_slice);
            }
            // Unreachable: `masked_fill_entry_point` above already refused
            // every dtype without an entry, so this arm only repeats its
            // refusal to keep the match total without a panic.
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: value_dtype,
                    backend: "Cuda",
                    op: "masked_fill",
                });
            }
        }
    }

    Ok(CudaStorage::new(Arc::new(out_b), input.shape.to_vec()))
}

/// Pure entry-point selection: runs wherever the `cuda` feature is on,
/// with no device required. The launcher launches exactly the name
/// `where_cond_entry_point` returns, so proving the mapping here is proving
/// a buffer can never reach a kernel typed for a different dtype.
#[cfg(all(test, feature = "cuda"))]
mod where_cond_entry_point_tests {
    use super::where_cond_entry_point;
    use incin_core::error::Error;
    use incin_core::tensor::dtype::DTypeId;

    #[test]
    fn where_cond_entry_point_maps_each_float_and_refuses_the_rest() {
        assert_eq!(
            where_cond_entry_point(DTypeId::F32.descriptor()).unwrap(),
            "where_cond_op"
        );
        assert_eq!(
            where_cond_entry_point(DTypeId::F64.descriptor()).unwrap(),
            "where_cond_op_f64"
        );
        assert_eq!(
            where_cond_entry_point(DTypeId::F16.descriptor()).unwrap(),
            "where_cond_op_f16"
        );
        assert_eq!(
            where_cond_entry_point(DTypeId::BF16.descriptor()).unwrap(),
            "where_cond_op_bf16"
        );

        for id in [
            DTypeId::I64,
            DTypeId::Bool,
            DTypeId::U8,
            DTypeId::U32,
            DTypeId::Q8_0,
        ] {
            let error = match where_cond_entry_point(id.descriptor()) {
                Ok(name) => panic!("{id:?} must be refused, got entry point {name:?}"),
                Err(error) => error,
            };
            assert!(
                matches!(
                    error,
                    Error::UnsupportedDType {
                        backend: "Cuda",
                        op: "where_cond",
                        ..
                    }
                ),
                "{id:?}: expected UnsupportedDType naming Cuda/where_cond, got {error:?}"
            );
        }
    }
}

/// The `masked_fill` half of the entry-point contract: the launcher launches
/// exactly the name `masked_fill_entry_point` returns, so proving the
/// mapping here is proving a buffer can never reach a kernel typed for a
/// different dtype. Runs wherever `cuda` is on, with no device.
#[cfg(all(test, feature = "cuda"))]
mod masked_fill_entry_point_tests {
    use super::masked_fill_entry_point;
    use incin_core::error::Error;
    use incin_core::tensor::dtype::DTypeId;

    #[test]
    fn masked_fill_entry_point_maps_each_float_and_refuses_the_rest() {
        assert_eq!(
            masked_fill_entry_point(DTypeId::F32.descriptor()).unwrap(),
            "masked_fill_op"
        );
        assert_eq!(
            masked_fill_entry_point(DTypeId::F64.descriptor()).unwrap(),
            "masked_fill_op_f64"
        );
        assert_eq!(
            masked_fill_entry_point(DTypeId::F16.descriptor()).unwrap(),
            "masked_fill_op_f16"
        );
        assert_eq!(
            masked_fill_entry_point(DTypeId::BF16.descriptor()).unwrap(),
            "masked_fill_op_bf16"
        );

        for id in [
            DTypeId::I64,
            DTypeId::Bool,
            DTypeId::U8,
            DTypeId::U32,
            DTypeId::Q8_0,
        ] {
            let error = match masked_fill_entry_point(id.descriptor()) {
                Ok(name) => panic!("{id:?} must be refused, got entry point {name:?}"),
                Err(error) => error,
            };
            assert!(
                matches!(
                    error,
                    Error::UnsupportedDType {
                        backend: "Cuda",
                        op: "masked_fill",
                        ..
                    }
                ),
                "{id:?}: expected UnsupportedDType naming Cuda/masked_fill, got {error:?}"
            );
        }
    }
}
