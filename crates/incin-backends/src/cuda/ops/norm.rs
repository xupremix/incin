//! CUDA `layer_norm` and `batch_norm`.
//!
//! `softmax` and `rms_norm` are not here: both are answered in
//! `cuda::executor` by composing already-implemented pointwise and reduction
//! `Execute<O>` calls rather than by a dedicated kernel, and their capability
//! rows say `Composed` because of it. `layer_norm` and `batch_norm` are real
//! kernels because a Welford row reduction and a per-channel affine pass are
//! each one launch; composing either from primitives would mean materializing
//! an intermediate the size of the input for no reason a fused kernel needs.

use crate::cuda::checked_i32;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::exec::PrecisionRequest;
use incin_core::shapes::OperationKind;

fn validate_contiguous(storage: &CudaStorage, name: &'static str) -> Result<usize> {
    let numel = crate::bytes::checked_numel(&storage.shape)?;
    if storage.strides != crate::layout::contiguous_strides(&storage.shape) {
        return Err(Error::Msg(format!(
            "CUDA normalization requires contiguous {name} storage"
        )));
    }
    let end = storage
        .offset_elements
        .checked_add(numel)
        .ok_or_else(|| Error::Msg(format!("CUDA normalization {name} storage bound overflow")))?;
    if end > storage.buffer.len {
        return Err(Error::Msg(format!(
            "CUDA normalization {name} view ends at {end}, but buffer length is {}",
            storage.buffer.len
        )));
    }
    Ok(numel)
}

fn validate_parameter(
    input: &CudaStorage,
    parameter: &CudaStorage,
    needed: usize,
    name: &'static str,
) -> Result<()> {
    if input.buffer.dtype != parameter.buffer.dtype {
        return Err(Error::DTypeStorageMismatch {
            expected: input.buffer.dtype,
            got: parameter.buffer.dtype,
        });
    }
    if input.buffer.device_id != parameter.buffer.device_id {
        return Err(Error::DeviceMismatch {
            left: incin_core::tensor::device::DeviceId::cuda(input.buffer.device_id),
            right: incin_core::tensor::device::DeviceId::cuda(parameter.buffer.device_id),
        });
    }
    let numel = validate_contiguous(parameter, name)?;
    if numel < needed {
        return Err(Error::Msg(format!(
            "CUDA normalization {name} has {numel} elements, but {needed} are required"
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn ensure_normalization_loaded(
    device_id: usize,
    kernel: &crate::kernel::RenderedKernel,
) -> Result<()> {
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

struct NormalizationLaunchSelection {
    candidate: crate::tuning::LaunchCandidate,
    #[cfg(feature = "autotune")]
    tuning_permit: Option<crate::tuning::TuningPermit>,
}

fn normalization_launch_selection(
    context: &cudarc::driver::CudaContext,
    kernel: &crate::kernel::RenderedKernel,
    batch_size: usize,
    norm_size: usize,
    is_layer_norm: bool,
) -> Result<NormalizationLaunchSelection> {
    let candidates = crate::tuning::normalization_candidates(is_layer_norm);
    let fallback = crate::tuning::default_normalization_candidate(&candidates)?;
    #[cfg(feature = "autotune")]
    {
        let key = crate::tuning::TuningKey::new(
            crate::tuning::identity::TuningEnvironmentFingerprint::<
                incin_core::tensor::device::Cuda,
            >::from_cuda_context(context)?
            .erase(),
            &kernel.key,
            crate::tuning::WorkloadBucket::normalization(batch_size, norm_size),
        );
        match crate::tuning::claim_tuning(key, &candidates)? {
            crate::tuning::TuningDecision::Cached(tuned) => Ok(NormalizationLaunchSelection {
                candidate: tuned.candidate,
                tuning_permit: None,
            }),
            crate::tuning::TuningDecision::Measure(permit) => Ok(NormalizationLaunchSelection {
                candidate: fallback,
                tuning_permit: Some(permit),
            }),
        }
    }
    #[cfg(not(feature = "autotune"))]
    {
        let _ = (context, kernel, batch_size, norm_size, is_layer_norm);
        Ok(NormalizationLaunchSelection {
            candidate: fallback,
        })
    }
}

#[cfg(feature = "cuda")]
fn empirically_select_normalization_candidate<F>(
    stream: &cudarc::driver::CudaStream,
    selection: NormalizationLaunchSelection,
    is_layer_norm: bool,
    mut launch: F,
) -> Result<crate::tuning::LaunchCandidate>
where
    F: FnMut(crate::tuning::LaunchCandidate) -> Result<()>,
{
    #[cfg(feature = "autotune")]
    if let Some(permit) = selection.tuning_permit {
        let candidates = crate::tuning::normalization_candidates(is_layer_norm);
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
    let _ = (stream, is_layer_norm, &mut launch);
    Ok(selection.candidate)
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_layer_norm(
    input: &CudaStorage,
    weight: &CudaStorage,
    bias: Option<&CudaStorage>,
    eps: f32,
) -> Result<CudaStorage> {
    let buffer = &*input.buffer;
    let input_numel = validate_contiguous(input, "input")?;
    let norm_size = *input
        .shape
        .last()
        .ok_or_else(|| Error::Msg("CUDA layer norm requires rank >= 1".into()))?;
    if norm_size == 0 {
        return Err(Error::Msg(
            "CUDA layer norm is undefined for an empty normalized axis".into(),
        ));
    }
    validate_parameter(input, weight, norm_size, "weight")?;
    if let Some(bias) = bias {
        validate_parameter(input, bias, norm_size, "bias")?;
    }
    let req = PrecisionRequest::new(
        incin_core::shapes::error::OperationKind::Normalization,
        buffer.dtype,
        buffer.dtype,
        incin_core::exec::LayoutClass::Contiguous,
        1,
        false,
        incin_core::exec::MathMode::Fast,
    );
    let policy = crate::cuda::backend::native_precision(&req)?;
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, "layer_norm")?;
    let kernel = crate::kernel::render_cuda_normalization("layer_norm", builtin_id)?;
    if Some(kernel.dtype) != buffer.dtype.builtin_id()
        || kernel.element_size
            != buffer
                .dtype
                .encoding()
                .scalar_bytes()
                .ok_or_else(|| Error::Msg("invalid scalar bytes".into()))?
    {
        return Err(Error::Msg(
            "CUDA layer norm kernel/storage ABI mismatch".into(),
        ));
    }
    let batch_size = input_numel / norm_size;
    let stream = buffer.device.default_stream();
    let mut output = CudaBuffer {
        len: input_numel,
        dtype: buffer.dtype,
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(crate::bytes::byte_len(
                    kernel.dtype,
                    input_numel,
                    OperationKind::Normalization,
                )?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA layer norm allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    if input_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(output), input.shape.to_vec()));
    }

    let selection =
        normalization_launch_selection(&buffer.device, &kernel, batch_size, norm_size, true)?;
    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;
    let bias_storage = bias.unwrap_or(weight);
    let has_bias = i32::from(bias.is_some());

    // SAFETY: the selected kernel's checked launch candidate and validated
    // tensor metadata bound all views; output is a fresh unique allocation.
    unsafe {
        let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
            Error::Msg("fresh CUDA layer norm output was unexpectedly shared".into())
        })?;
        use cudarc::driver::PushKernelArg;
        let mut launch = |candidate: crate::tuning::LaunchCandidate| -> Result<()> {
            let block_size = u32::from(candidate.block_size);
            let warp_count = (block_size as usize) / 32;
            let acc_bytes = policy
                .accumulator
                .encoding()
                .scalar_bytes()
                .ok_or_else(|| Error::Msg("CUDA layer norm invalid accumulator encoding".into()))?;
            let shared_bytes = warp_count
                .checked_mul(acc_bytes)
                .and_then(|bytes| bytes.checked_mul(2))
                .and_then(|bytes| bytes.checked_add(warp_count * core::mem::size_of::<i32>()))
                .and_then(|bytes| u32::try_from(bytes).ok())
                .ok_or_else(|| Error::Msg("CUDA layer norm shared-memory size overflow".into()))?;
            let grid = u32::try_from(batch_size).map_err(|_| {
                Error::Msg("CUDA layer norm batch count exceeds u32 grid ABI".into())
            })?;
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            stream
                .launch_builder(&function)
                .arg(&*buffer.data)
                .arg(&*weight.buffer.data)
                .arg(&*bias_storage.buffer.data)
                .arg(&mut *output_u8)
                .arg(&eps)
                .arg(&checked_i32(norm_size, "normalized axis length")?)
                .arg(&has_bias)
                .arg(&checked_i32(batch_size, "batch count")?)
                .arg(&checked_i32(input.offset_elements, "input offset")?)
                .arg(&checked_i32(weight.offset_elements, "weight offset")?)
                .arg(&checked_i32(bias_storage.offset_elements, "bias offset")?)
                .launch(config)
                .map(|_| ())
                .map_err(|error| Error::Msg(format!("CUDA layer norm launch failed: {error:?}")))
        };
        let candidate =
            empirically_select_normalization_candidate(&stream, selection, true, &mut launch)?;
        launch(candidate)?;
    }

    Ok(CudaStorage::new(Arc::new(output), input.shape.to_vec()))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_batch_norm(
    input: &CudaStorage,
    weight: Option<&CudaStorage>,
    bias: Option<&CudaStorage>,
    running_mean: Option<&CudaStorage>,
    running_variance: Option<&CudaStorage>,
    eps: f32,
) -> Result<CudaStorage> {
    let buffer = &*input.buffer;
    let total_elements = validate_contiguous(input, "input")?;
    if input.shape.is_empty() {
        return Err(Error::Msg("CUDA batch norm requires rank >= 1".into()));
    }
    let channel_axis = usize::from(input.shape.len() > 1);
    let num_channels = input.shape[channel_axis];
    let spatial_size = if input.shape.len() > 2 {
        crate::bytes::checked_numel(&input.shape[2..])?
    } else {
        1
    };
    for (parameter, name) in [
        (weight, "weight"),
        (bias, "bias"),
        (running_mean, "running mean"),
        (running_variance, "running variance"),
    ] {
        if let Some(parameter) = parameter {
            validate_parameter(input, parameter, num_channels, name)?;
        }
    }
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, "batch_norm")?;
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, "batch_norm")?;
    let kernel = crate::kernel::render_cuda_normalization("batch_norm", builtin_id)?;
    let selection = normalization_launch_selection(
        &buffer.device,
        &kernel,
        total_elements,
        num_channels,
        false,
    )?;
    let stream = buffer.device.default_stream();
    let mut output = CudaBuffer {
        len: total_elements,
        dtype: buffer.dtype,
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(crate::bytes::byte_len(
                    kernel.dtype,
                    total_elements,
                    OperationKind::Normalization,
                )?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA batch norm allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    if total_elements == 0 {
        return Ok(CudaStorage::new(Arc::new(output), input.shape.to_vec()));
    }

    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;
    let weight_storage = weight.unwrap_or(input);
    let bias_storage = bias.unwrap_or(input);
    let mean_storage = running_mean.unwrap_or(input);
    let variance_storage = running_variance.unwrap_or(input);

    // SAFETY: validated batch-norm metadata and the checked launch candidate
    // bound device accesses; output is freshly allocated and uniquely owned.
    unsafe {
        let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
            Error::Msg("fresh CUDA batch norm output was unexpectedly shared".into())
        })?;
        use cudarc::driver::PushKernelArg;
        let mut launch = |candidate: crate::tuning::LaunchCandidate| -> Result<()> {
            let block_size = u32::from(candidate.block_size);
            let work_items = u32::try_from(total_elements).map_err(|_| {
                Error::Msg("CUDA batch norm element count exceeds u32 grid ABI".into())
            })?;
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (work_items.div_ceil(block_size), 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            stream
                .launch_builder(&function)
                .arg(&*buffer.data)
                .arg(&*weight_storage.buffer.data)
                .arg(&*bias_storage.buffer.data)
                .arg(&*mean_storage.buffer.data)
                .arg(&*variance_storage.buffer.data)
                .arg(&mut *output_u8)
                .arg(&eps)
                .arg(&checked_i32(num_channels, "channel count")?)
                .arg(&checked_i32(spatial_size, "spatial size")?)
                .arg(&checked_i32(total_elements, "element count")?)
                .arg(&i32::from(weight.is_some()))
                .arg(&i32::from(bias.is_some()))
                .arg(&i32::from(running_mean.is_some()))
                .arg(&i32::from(running_variance.is_some()))
                .arg(&checked_i32(input.offset_elements, "input offset")?)
                .arg(&checked_i32(
                    weight_storage.offset_elements,
                    "weight offset",
                )?)
                .arg(&checked_i32(bias_storage.offset_elements, "bias offset")?)
                .arg(&checked_i32(mean_storage.offset_elements, "mean offset")?)
                .arg(&checked_i32(
                    variance_storage.offset_elements,
                    "variance offset",
                )?)
                .launch(config)
                .map(|_| ())
                .map_err(|error| Error::Msg(format!("CUDA batch norm launch failed: {error:?}")))
        };
        let candidate =
            empirically_select_normalization_candidate(&stream, selection, false, &mut launch)?;
        launch(candidate)?;
    }

    Ok(CudaStorage::new(Arc::new(output), input.shape.to_vec()))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_rms_norm(
    input: &CudaStorage,
    weight: &CudaStorage,
    eps: f32,
) -> Result<CudaStorage> {
    let buffer = &*input.buffer;
    let input_numel = validate_contiguous(input, "input")?;
    let norm_size = *input
        .shape
        .last()
        .ok_or_else(|| Error::Msg("CUDA RMS norm requires rank >= 1".into()))?;
    if norm_size == 0 {
        return Err(Error::Msg(
            "CUDA RMS norm is undefined for an empty normalized axis".into(),
        ));
    }
    validate_parameter(input, weight, norm_size, "weight")?;
    crate::cuda::backend::validate_cuda_storage_dtype(buffer.dtype, "rms_norm")?;
    let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, "rms_norm")?;
    let kernel = crate::kernel::render_cuda_normalization("rms_norm", builtin_id)?;
    let batch_size = input_numel / norm_size;
    let stream = buffer.device.default_stream();
    let mut output = CudaBuffer {
        len: input_numel,
        dtype: buffer.dtype,
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(crate::bytes::byte_len(
                    kernel.dtype,
                    input_numel,
                    OperationKind::Normalization,
                )?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA RMS norm allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    if input_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(output), input.shape.to_vec()));
    }

    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
    let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;

    // SAFETY: Input, weight, and output buffers are validated for contiguous bounds and data types;
    // launch configuration uses checked block and grid dimensions with exclusive output buffer access.
    unsafe {
        let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
            Error::Msg("fresh CUDA RMS norm output was unexpectedly shared".into())
        })?;
        use cudarc::driver::PushKernelArg;
        let block_size = 256u32;
        let warp_count = (block_size as usize).div_ceil(32);
        let shared_bytes = (warp_count * core::mem::size_of::<f32>()) as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        stream
            .launch_builder(&function)
            .arg(&*buffer.data)
            .arg(&*weight.buffer.data)
            .arg(&mut *output_u8)
            .arg(&eps)
            .arg(&checked_i32(norm_size, "norm size")?)
            .arg(&checked_i32(batch_size, "batch size")?)
            .arg(&checked_i32(input.offset_elements, "input offset")?)
            .arg(&checked_i32(weight.offset_elements, "weight offset")?)
            .launch(config)
            .map_err(|error| Error::Msg(format!("CUDA RMS norm launch failed: {error:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(output), input.shape.to_vec()))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_softmax(input: &CudaStorage, dim: usize) -> Result<CudaStorage> {
    let buffer = &*input.buffer;
    let input_numel = validate_contiguous(input, "input")?;
    let rank = input.shape.len();
    if dim >= rank {
        return Err(Error::Msg(format!(
            "CUDA softmax dim {dim} is out of bounds for rank {rank}"
        )));
    }
    // If softmax is along the last dimension, run the fast fused warp-reduction kernel
    if dim == rank - 1 {
        let norm_size = input.shape[dim];
        if norm_size == 0 {
            return Err(Error::Msg(
                "CUDA softmax is undefined for empty axis".into(),
            ));
        }
        let builtin_id = crate::cuda::backend::require_cuda_builtin_dtype(buffer.dtype, "softmax")?;
        let kernel = crate::kernel::render_cuda_normalization("softmax", builtin_id)?;
        let batch_size = input_numel / norm_size;
        let stream = buffer.device.default_stream();
        let mut output = CudaBuffer {
            len: input_numel,
            dtype: buffer.dtype,
            data: Arc::new(
                stream
                    .alloc_zeros::<u8>(crate::bytes::byte_len(
                        kernel.dtype,
                        input_numel,
                        OperationKind::Normalization,
                    )?)
                    .map_err(|error| {
                        Error::Msg(format!("CUDA softmax allocation failed: {error:?}"))
                    })?,
            ),
            device: buffer.device.clone(),
            device_id: buffer.device_id,
        };
        if input_numel == 0 {
            return Ok(CudaStorage::new(Arc::new(output), input.shape.to_vec()));
        }

        ensure_normalization_loaded(buffer.device_id, &kernel)?;
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id)?;
        let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;

        // SAFETY: Input and output buffers are validated for contiguous bounds and data types;
        // kernel execution is bounded by batch size and norm size dimensions.
        unsafe {
            let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
                Error::Msg("fresh CUDA softmax output was unexpectedly shared".into())
            })?;
            use cudarc::driver::PushKernelArg;
            let block_size = 256u32;
            let warp_count = (block_size as usize).div_ceil(32);
            let shared_bytes = (warp_count * core::mem::size_of::<f32>()) as u32;
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (batch_size as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: shared_bytes,
            };
            stream
                .launch_builder(&function)
                .arg(&*buffer.data)
                .arg(&mut *output_u8)
                .arg(&checked_i32(norm_size, "norm size")?)
                .arg(&checked_i32(batch_size, "batch size")?)
                .arg(&checked_i32(input.offset_elements, "input offset")?)
                .launch(config)
                .map_err(|error| Error::Msg(format!("CUDA softmax launch failed: {error:?}")))?;
        }

        Ok(CudaStorage::new(Arc::new(output), input.shape.to_vec()))
    } else {
        // Fallback for non-last dimension: safe composed path
        let max = crate::cuda::ops::reduce::launch_reduce_op("max", input, dim, true)?;
        let diff = crate::cuda::backend::cuda_sub_storage(input, &max)?;
        let exp = crate::cuda::backend::cuda_exp_storage(
            &diff,
            crate::kernel::KernelSpecialization::NONE,
        )?;
        let sum = crate::cuda::ops::reduce::launch_reduce_op("sum", &exp, dim, true)?;
        crate::cuda::backend::cuda_div_storage(&exp, &sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_metadata_checks_reject_overflow() {
        assert!(crate::bytes::checked_numel(&[usize::MAX, 2]).is_err());
        assert!(
            crate::bytes::byte_len(
                incin_core::tensor::dtype::DTypeId::F16,
                usize::MAX,
                OperationKind::Normalization
            )
            .is_err()
        );
    }
}
