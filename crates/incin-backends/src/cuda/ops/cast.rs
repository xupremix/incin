//! On-device dtype cast backing `Execute<op::ToDType>` (issue #106(a)).
//!
//! Replaces the previous host round-trip (download bytes, convert on the CPU,
//! upload) with a single runtime type-code kernel. Support is decided by the
//! dtype pair, never by the values, and matches CPU's refusal set for targets:
//! Q8_0 is block-packed, Bool has no CPU conversion, and U8/U32 are not CUDA
//! storage dtypes - all fail loudly rather than producing zeros.

use super::alloc_zeroed_bytes;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::shapes::OperationKind;
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

#[cfg(feature = "cuda")]
const CAST_SRC: &str = include_str!("kernels/cast.cu");

/// Source type code for the cast kernel (`cast.cu`'s `src_code`).
#[cfg(feature = "cuda")]
fn source_code(dtype: DTypeDescriptor) -> Result<i32> {
    match dtype.builtin_id() {
        Some(DTypeId::F32) => Ok(0),
        Some(DTypeId::F64) => Ok(1),
        Some(DTypeId::F16) => Ok(2),
        Some(DTypeId::BF16) => Ok(3),
        Some(DTypeId::I64) => Ok(4),
        Some(DTypeId::Bool) => Ok(5),
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Cuda",
            op: "to_dtype",
        }),
    }
}

/// Target type code for the cast kernel. Deliberately narrower than the
/// source set: bool/u8/u32/Q8_0 targets have no honest CUDA path (and CPU
/// itself refuses Q8_0 and unknown/bool targets).
#[cfg(feature = "cuda")]
fn target_code(dtype: DTypeDescriptor) -> Result<i32> {
    match dtype.builtin_id() {
        Some(DTypeId::F32) => Ok(0),
        Some(DTypeId::F64) => Ok(1),
        Some(DTypeId::F16) => Ok(2),
        Some(DTypeId::BF16) => Ok(3),
        Some(DTypeId::I64) => Ok(4),
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Cuda",
            op: "to_dtype",
        }),
    }
}

#[cfg(feature = "cuda")]
fn ensure_cast_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "cast").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel("cast", CAST_SRC, "cast")?;
    }
    Ok(())
}

/// Linear cast of a contiguous `input` into a fresh buffer of `target_dtype`.
///
/// Refuses strided views (the kernel reads linearly from one base pointer) and
/// every target outside `{f32,f64,f16,bf16,i64}`. A zero-element tensor skips
/// the launch. The output is always contiguous with offset 0, matching the
/// shape-and-strides contract the previous host path used.
#[cfg(feature = "cuda")]
pub(crate) fn launch_cast(
    input: &CudaStorage,
    target_dtype: DTypeDescriptor,
) -> Result<CudaStorage> {
    let source_dtype = input.buffer.dtype;
    if input.buffer.dtype == target_dtype {
        return Ok(input.clone());
    }
    let src_code = source_code(source_dtype)?;
    let dst_code = target_code(target_dtype)?;
    if input.strides != crate::layout::contiguous_strides(&input.shape) {
        return Err(Error::Msg(format!(
            "CUDA to_dtype requires a contiguous operand, got strides {:?} for shape {:?}",
            input.strides, input.shape
        )));
    }
    let total = input.shape.iter().product::<usize>();
    let device_id = input.buffer.device_id;
    ensure_cast_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("cast", "to_dtype_cast")?;
    let stream = input.buffer.device.default_stream();

    let mut out_buffer = CudaBuffer {
        len: total,
        dtype: target_dtype,
        data: Arc::new(alloc_zeroed_bytes(
            &stream,
            target_dtype,
            total,
            OperationKind::ToDType,
        )?),
        device: input.buffer.device.clone(),
        device_id,
    };

    if total > 0 {
        let block_size = 256u32;
        let grid_size =
            crate::cuda::checked_u32(total, "to_dtype launch grid")?.div_ceil(block_size);
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i64 = i64::try_from(total)
            .map_err(|_| Error::Msg("to_dtype element count exceeds i64".into()))?;
        let offset_i64 = i64::try_from(input.offset_elements)
            .map_err(|_| Error::Msg("to_dtype offset exceeds i64".into()))?;

        // SAFETY: Launches the cast kernel with the verified element count and
        // freshly allocated output buffer.
        unsafe {
            let out_u8 = Arc::get_mut(&mut out_buffer.data)
                .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&function)
                .arg(&*input.buffer.data)
                .arg(&mut *out_u8)
                .arg(&n_i64)
                .arg(&src_code)
                .arg(&dst_code)
                .arg(&offset_i64)
                .launch(config)
                .map_err(|e| Error::Msg(alloc::format!("CUDA to_dtype launch failed: {e:?}")))?;
        }
    }
    let strides = crate::layout::contiguous_strides(&input.shape)
        .strides()
        .to_vec();
    CudaStorage::try_from_parts(Arc::new(out_buffer), input.shape.to_vec(), strides, 0)
}
