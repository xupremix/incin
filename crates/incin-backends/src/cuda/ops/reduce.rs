use crate::cuda::storage::{CudaBuffer, CudaStorage};
use crate::iteration::OperandIteration;
use alloc::sync::Arc;
use incin_core::exec::PrecisionRequest;
use incin_core::prelude::{DTypeId, Error, OperationKind, Result};

fn checked_i32(value: usize, field: &'static str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::Msg(format!("CUDA reduction {field} {value} exceeds i32 ABI")))
}

fn checked_i32_vec(values: &[usize], field: &'static str) -> Result<Vec<i32>> {
    values
        .iter()
        .map(|&value| checked_i32(value, field))
        .collect()
}

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

#[cfg(feature = "cuda")]
pub(crate) fn launch_reduce_with_indices_op(
    op_name: &'static str,
    storage: &CudaStorage,
    axis: usize,
    keepdim: bool,
) -> Result<(CudaStorage, CudaStorage)> {
    let (buffer, reduce_dim_size) = validate_reduction(storage, axis, op_name)?;
    let (keepdim_shape, final_shape) = reduction_shapes(&storage.shape, axis, keepdim);
    let out_numel = keepdim_shape.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| Error::Msg("CUDA indexed reduction output size overflow".into()))
    })?;
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, op_name)?;
    let kernel = crate::kernel::render_cuda_reduction(op_name, builtin_id, true, false)?;
    let launch_selection =
        reduction_launch_selection(&buffer.device, &kernel, out_numel, reduce_dim_size, false)?;
    let stream = buffer.device.default_stream();
    let mut value_buffer = CudaBuffer {
        len: out_numel,
        dtype: buffer.dtype,
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(crate::bytes::byte_len(
                    kernel.dtype,
                    out_numel,
                    OperationKind::Reduction,
                )?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA reduction value allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    let mut index_buffer = CudaBuffer {
        len: out_numel,
        dtype: DTypeId::U32.descriptor(),
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(crate::bytes::byte_len(
                    DTypeId::U32.descriptor(),
                    out_numel,
                    OperationKind::Reduction,
                )?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA reduction index allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    if out_numel == 0 {
        return Ok((
            CudaStorage::new(Arc::new(value_buffer), final_shape.clone()),
            CudaStorage::new(Arc::new(index_buffer), final_shape),
        ));
    }

    ensure_reduction_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;
    let in_strides = checked_i32_vec(&storage.strides, "input stride")?;
    let out_shape = checked_i32_vec(&keepdim_shape, "output shape")?;
    let out_strides_host = crate::layout::contiguous_strides(&keepdim_shape);
    let out_strides = checked_i32_vec(&out_strides_host, "output stride")?;
    let in_strides_dev = stream.clone_htod(&in_strides).map_err(|error| {
        Error::Msg(format!(
            "CUDA indexed reduction stride upload failed: {error:?}"
        ))
    })?;
    let out_shape_dev = stream.clone_htod(&out_shape).map_err(|error| {
        Error::Msg(format!(
            "CUDA indexed reduction shape upload failed: {error:?}"
        ))
    })?;
    let out_strides_dev = stream.clone_htod(&out_strides).map_err(|error| {
        Error::Msg(format!(
            "CUDA indexed reduction output-stride upload failed: {error:?}"
        ))
    })?;
    let out_numel_i32 = checked_i32(out_numel, "output element count")?;
    let work_items = u32::try_from(out_numel).map_err(|_| {
        Error::Msg(format!(
            "CUDA indexed reduction work-item count {out_numel} exceeds u32 grid ABI"
        ))
    })?;

    unsafe {
        let value_u8 = Arc::get_mut(&mut value_buffer.data).ok_or_else(|| {
            Error::Msg("fresh CUDA reduction value buffer was unexpectedly shared".into())
        })?;
        let index_u8 = Arc::get_mut(&mut index_buffer.data).ok_or_else(|| {
            Error::Msg("fresh CUDA reduction index buffer was unexpectedly shared".into())
        })?;
        use cudarc::driver::PushKernelArg;
        let in_offset = checked_i32(storage.offset_elements, "input offset")?;
        let axis_i32 = checked_i32(axis, "axis")?;
        let reduce_dim = checked_i32(reduce_dim_size, "axis length")?;
        let ndim = checked_i32(storage.shape.len(), "rank")?;
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
                .arg(&mut *value_u8)
                .arg(&mut *index_u8)
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
                    Error::Msg(format!(
                        "CUDA indexed {op_name} reduction launch failed: {error:?}"
                    ))
                })
        };
        let launch_candidate =
            empirically_select_reduction_candidate(&stream, launch_selection, false, &mut launch)?;
        launch(launch_candidate)?;
    }

    Ok((
        CudaStorage::new(Arc::new(value_buffer), final_shape.clone()),
        CudaStorage::new(Arc::new(index_buffer), final_shape),
    ))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_reduce_with_indices_host(
    op_name: &'static str,
    storage: &CudaStorage,
    axis: usize,
    keepdim: bool,
) -> Result<(CudaStorage, Vec<usize>)> {
    let (value_storage, index_storage) =
        launch_reduce_with_indices_op(op_name, storage, axis, keepdim)?;
    let buffer = &*index_storage.buffer;
    if buffer.dtype != DTypeId::U32.descriptor() {
        return Err(Error::DTypeStorageMismatch {
            expected: DTypeId::U32.descriptor(),
            got: buffer.dtype,
        });
    }
    let indices = unsafe {
        let device_values = buffer.data.transmute::<u32>(buffer.len).ok_or_else(|| {
            Error::Msg("CUDA reduction index view has invalid byte length".into())
        })?;
        buffer
            .device
            .default_stream()
            .clone_dtoh(&device_values)
            .map_err(|error| {
                Error::Msg(format!("CUDA reduction index download failed: {error:?}"))
            })?
    };
    Ok((
        value_storage,
        indices.into_iter().map(|index| index as usize).collect(),
    ))
}

/// Converts a `U32`-dtype index `CudaStorage` (what `launch_reduce_with_indices_op`
/// produces) into an `I64`-dtype one, matching CPU/WGPU's `argmax`/`argmin`
/// convention (`CpuBuffer::I64`) so downstream consumers that assume
/// integer index tensors are `I64` (e.g. `embedding`, `cross_entropy_loss`)
/// keep working uniformly across backends. Small buffer (one index per
/// reduced output position, not per input element), so a host round-trip
/// here is cheap regardless of the input tensor's size.
#[cfg(feature = "cuda")]
pub(crate) fn indices_u32_to_i64(idx: &CudaStorage) -> Result<CudaStorage> {
    let buf = &*idx.buffer;
    if buf.dtype != DTypeId::U32.descriptor() {
        return Err(Error::DTypeStorageMismatch {
            expected: DTypeId::U32.descriptor(),
            got: buf.dtype,
        });
    }
    let stream = buf.device.default_stream();
    let host_u32: alloc::vec::Vec<u32> = unsafe {
        let view = buf
            .data
            .transmute::<u32>(buf.len)
            .ok_or_else(|| Error::Msg("indices_u32_to_i64: invalid byte length".into()))?;
        stream
            .clone_dtoh(&view)
            .map_err(|e| Error::Msg(format!("indices_u32_to_i64: download failed: {e:?}")))?
    };
    let host_i64: alloc::vec::Vec<i64> = host_u32.into_iter().map(|v| v as i64).collect();
    crate::cuda::backend::cuda_from_bytes(
        &idx.shape,
        DTypeId::I64.descriptor(),
        buf.device_id,
        bytemuck::cast_slice(&host_i64),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
