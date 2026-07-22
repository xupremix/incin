use crate::cuda::storage::{CudaBuffer, CudaStorage};
use crate::dtype_policy::{BackendFamily, OperationFamily, resolve_dtype_policy};
use alloc::sync::Arc;
use kindle_core::prelude::{Error, Result};

fn checked_i32(value: usize, field: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::Msg(format!(
            "CUDA normalization {field} {value} exceeds i32 ABI"
        ))
    })
}

fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| Error::Msg("CUDA normalization element count overflow".into()))
    })
}

fn checked_byte_len(numel: usize, element_size: usize) -> Result<usize> {
    numel.checked_mul(element_size).ok_or_else(|| {
        Error::Msg(format!(
            "CUDA normalization allocation overflow: {numel} elements of {element_size} bytes"
        ))
    })
}

fn validate_contiguous(storage: &CudaStorage, name: &'static str) -> Result<usize> {
    let numel = checked_numel(&storage.shape)?;
    if storage.strides != crate::cpu::stride::contiguous_strides(&storage.shape) {
        return Err(Error::Msg(format!(
            "CUDA normalization requires contiguous {name} storage"
        )));
    }
    let end = storage
        .offset
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
            left: kindle_core::prelude::DeviceId::cuda(input.buffer.device_id),
            right: kindle_core::prelude::DeviceId::cuda(parameter.buffer.device_id),
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
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        dispatcher.compile_and_load_kernel(
            &kernel.entry_point,
            &kernel.source,
            &kernel.cache_key,
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_layer_norm(
    input: &CudaStorage,
    weight: &CudaStorage,
    bias: Option<&CudaStorage>,
    eps: f32,
) -> Result<CudaStorage> {
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
    let buffer = &*input.buffer;
    let policy = resolve_dtype_policy(
        BackendFamily::Cuda,
        OperationFamily::Normalization,
        buffer.dtype,
        "layer_norm",
    )?;
    let kernel = crate::kernel::render_cuda_normalization("layer_norm", buffer.dtype)?;
    if kernel.dtype != buffer.dtype || kernel.element_size != buffer.dtype.element_size() {
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
                .alloc_zeros::<u8>(checked_byte_len(input_numel, kernel.element_size)?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA layer norm allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    if input_numel == 0 {
        return Ok(CudaStorage::new(Arc::new(output), input.shape.clone()));
    }

    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id);
    let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;
    let warp_count = 256usize / 32;
    let shared_bytes = warp_count
        .checked_mul(policy.accumulator.element_size())
        .and_then(|bytes| bytes.checked_mul(2))
        .and_then(|bytes| bytes.checked_add(warp_count * core::mem::size_of::<i32>()))
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| Error::Msg("CUDA layer norm shared-memory size overflow".into()))?;
    let grid = u32::try_from(batch_size)
        .map_err(|_| Error::Msg("CUDA layer norm batch count exceeds u32 grid ABI".into()))?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let bias_storage = bias.unwrap_or(weight);
    let has_bias = i32::from(bias.is_some());

    unsafe {
        let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
            Error::Msg("fresh CUDA layer norm output was unexpectedly shared".into())
        })?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*buffer.data)
            .arg(&*weight.buffer.data)
            .arg(&*bias_storage.buffer.data)
            .arg(output_u8)
            .arg(&eps)
            .arg(&checked_i32(norm_size, "normalized axis length")?)
            .arg(&has_bias)
            .arg(&checked_i32(batch_size, "batch count")?)
            .arg(&checked_i32(input.offset, "input offset")?)
            .arg(&checked_i32(weight.offset, "weight offset")?)
            .arg(&checked_i32(bias_storage.offset, "bias offset")?)
            .launch(config)
            .map_err(|error| Error::Msg(format!("CUDA layer norm launch failed: {error:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(output), input.shape.clone()))
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
    let total_elements = validate_contiguous(input, "input")?;
    if input.shape.is_empty() {
        return Err(Error::Msg("CUDA batch norm requires rank >= 1".into()));
    }
    let channel_axis = usize::from(input.shape.len() > 1);
    let num_channels = input.shape[channel_axis];
    let spatial_size = if input.shape.len() > 2 {
        checked_numel(&input.shape[2..])?
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
    let buffer = &*input.buffer;
    resolve_dtype_policy(
        BackendFamily::Cuda,
        OperationFamily::Normalization,
        buffer.dtype,
        "batch_norm",
    )?;
    let kernel = crate::kernel::render_cuda_normalization("batch_norm", buffer.dtype)?;
    let stream = buffer.device.default_stream();
    let mut output = CudaBuffer {
        len: total_elements,
        dtype: buffer.dtype,
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(checked_byte_len(total_elements, kernel.element_size)?)
                .map_err(|error| {
                    Error::Msg(format!("CUDA batch norm allocation failed: {error:?}"))
                })?,
        ),
        device: buffer.device.clone(),
        device_id: buffer.device_id,
    };
    if total_elements == 0 {
        return Ok(CudaStorage::new(Arc::new(output), input.shape.clone()));
    }

    ensure_normalization_loaded(buffer.device_id, &kernel)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(buffer.device_id);
    let function = dispatcher.get_function(&kernel.cache_key, &kernel.entry_point)?;
    let work_items = u32::try_from(total_elements)
        .map_err(|_| Error::Msg("CUDA batch norm element count exceeds u32 grid ABI".into()))?;
    let config = cudarc::driver::LaunchConfig {
        grid_dim: (work_items.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let weight_storage = weight.unwrap_or(input);
    let bias_storage = bias.unwrap_or(input);
    let mean_storage = running_mean.unwrap_or(input);
    let variance_storage = running_variance.unwrap_or(input);

    unsafe {
        let output_u8 = Arc::get_mut(&mut output.data).ok_or_else(|| {
            Error::Msg("fresh CUDA batch norm output was unexpectedly shared".into())
        })?;
        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&function)
            .arg(&*buffer.data)
            .arg(&*weight_storage.buffer.data)
            .arg(&*bias_storage.buffer.data)
            .arg(&*mean_storage.buffer.data)
            .arg(&*variance_storage.buffer.data)
            .arg(output_u8)
            .arg(&eps)
            .arg(&checked_i32(num_channels, "channel count")?)
            .arg(&checked_i32(spatial_size, "spatial size")?)
            .arg(&checked_i32(total_elements, "element count")?)
            .arg(&i32::from(weight.is_some()))
            .arg(&i32::from(bias.is_some()))
            .arg(&i32::from(running_mean.is_some()))
            .arg(&i32::from(running_variance.is_some()))
            .arg(&checked_i32(input.offset, "input offset")?)
            .arg(&checked_i32(weight_storage.offset, "weight offset")?)
            .arg(&checked_i32(bias_storage.offset, "bias offset")?)
            .arg(&checked_i32(mean_storage.offset, "mean offset")?)
            .arg(&checked_i32(variance_storage.offset, "variance offset")?)
            .launch(config)
            .map_err(|error| Error::Msg(format!("CUDA batch norm launch failed: {error:?}")))?;
    }

    Ok(CudaStorage::new(Arc::new(output), input.shape.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_metadata_checks_reject_overflow() {
        assert_eq!(checked_i32(i32::MAX as usize, "test").unwrap(), i32::MAX);
        assert!(checked_i32(i32::MAX as usize + 1, "test").is_err());
        assert!(checked_numel(&[usize::MAX, 2]).is_err());
        assert!(checked_byte_len(usize::MAX, 2).is_err());
    }
}
