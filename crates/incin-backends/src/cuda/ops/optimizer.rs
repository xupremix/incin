//! CUDA implementation of fused optimizer steps (Adam, AdamW, SGD).

#![allow(dead_code, unused_imports)]

use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::exec::catalog::{AdamAttributes, AdamWAttributes, SgdAttributes};
use incin_core::shapes::OperationKind;

#[cfg(feature = "cuda")]
const OPTIMIZER_SRC: &str = include_str!("kernels/optimizer.cu");

/// One thread per element, in blocks of 256.
///
/// The optimizer kernels are pointwise over the parameter buffer and bounded by
/// memory rather than occupancy, so the block size is not tuned.
#[cfg(feature = "cuda")]
fn optimizer_launch_config(numel: usize) -> cudarc::driver::LaunchConfig {
    const BLOCK: u32 = 256;
    let blocks = numel.div_ceil(BLOCK as usize).max(1) as u32;
    cudarc::driver::LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(feature = "cuda")]
fn ensure_optimizer_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "optimizer").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel("optimizer", OPTIMIZER_SRC, "optimizer")?;
    }
    Ok(())
}

/// Launches the fused AdamW step kernel on CUDA.
pub(crate) fn launch_adamw_step(
    p_in: &CudaStorage,
    grad: &CudaStorage,
    m_in: Option<&CudaStorage>,
    v_in: Option<&CudaStorage>,
    attrs: &AdamWAttributes,
) -> Result<(CudaStorage, CudaStorage, CudaStorage)> {
    let len = p_in.buffer.len;
    let device_id = p_in.buffer.device_id;
    let stream = p_in.buffer.device.default_stream();
    let dtype = p_in.buffer.dtype;

    let mut p_out_buf = CudaBuffer {
        len,
        dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            dtype,
            len,
            OperationKind::AdamWStep,
        )?),
        device: p_in.buffer.device.clone(),
        device_id,
    };

    let mut m_out_buf = CudaBuffer {
        len,
        dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            dtype,
            len,
            OperationKind::AdamWStep,
        )?),
        device: p_in.buffer.device.clone(),
        device_id,
    };

    let mut v_out_buf = CudaBuffer {
        len,
        dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            dtype,
            len,
            OperationKind::AdamWStep,
        )?),
        device: p_in.buffer.device.clone(),
        device_id,
    };

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::PushKernelArg;
        use incin_core::tensor::dtype::DTypeId;

        ensure_optimizer_loaded(device_id)?;
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        let numel = len;
        let config = optimizer_launch_config(numel);
        // A moment buffer that was not supplied is a null pointer, which the
        // kernel reads as "start from zero" -- that is what makes the first
        // step work without the caller allocating state it does not have yet.
        let m_ptr = m_in.map(|s| &*s.buffer.data);
        let v_ptr = v_in.map(|s| &*s.buffer.data);
        let empty =
            crate::cuda::ops::alloc_zeroed_bytes(&stream, dtype, len, OperationKind::AdamWStep)?;
        let m_src = m_ptr.unwrap_or(&empty);
        let v_src = v_ptr.unwrap_or(&empty);

        let p_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut p_out_buf.data)
            .ok_or_else(|| Error::Msg("fresh optimizer output was unexpectedly shared".into()))?;
        let p_ptr = p_slice as *mut cudarc::driver::CudaSlice<u8>;
        let m_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut m_out_buf.data)
            .ok_or_else(|| Error::Msg("fresh optimizer output was unexpectedly shared".into()))?;
        let m_ptr_out = m_slice as *mut cudarc::driver::CudaSlice<u8>;
        let v_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut v_out_buf.data)
            .ok_or_else(|| Error::Msg("fresh optimizer output was unexpectedly shared".into()))?;

        // Scalars must be passed at the kernel's own precision: an `f64` handed
        // to a `const float` parameter shifts every argument after it, which
        // fails silently rather than loudly.
        let step = i32::try_from(attrs.step).unwrap_or(i32::MAX);
        match dtype.builtin_id() {
            Some(DTypeId::F32) => {
                let function = dispatcher.get_function("optimizer", "adamw_step_f32")?;
                #[allow(clippy::cast_possible_truncation)]
                let (lr, beta1, beta2, eps) = (
                    attrs.learning_rate as f32,
                    attrs.beta1 as f32,
                    attrs.beta2 as f32,
                    attrs.epsilon as f32,
                );
                #[allow(clippy::cast_possible_truncation)]
                let weight_decay = attrs.weight_decay as f32;
                #[allow(clippy::cast_possible_truncation)]
                let bias1 = (1.0 - attrs.beta1.powi(step)) as f32;
                #[allow(clippy::cast_possible_truncation)]
                let bias2 = (1.0 - attrs.beta2.powi(step)) as f32;
                // SAFETY: the three outputs are fresh allocations this call
                // uniquely owns and are distinct from each other; every operand
                // is `numel` f32 elements, matching the launch configuration.
                unsafe {
                    stream
                        .launch_builder(&function)
                        .arg(&mut *p_ptr)
                        .arg(&mut *m_ptr_out)
                        .arg(&mut *v_slice)
                        .arg(&*p_in.buffer.data)
                        .arg(&*grad.buffer.data)
                        .arg(m_src)
                        .arg(v_src)
                        .arg(&lr)
                        .arg(&beta1)
                        .arg(&beta2)
                        .arg(&eps)
                        .arg(&weight_decay)
                        .arg(&bias1)
                        .arg(&bias2)
                        .arg(&numel)
                        .launch(config)
                        .map_err(|error| {
                            Error::Msg(alloc::format!("adamw_step launch failed: {error:?}"))
                        })?;
                }
            }
            Some(DTypeId::F64) => {
                let function = dispatcher.get_function("optimizer", "adamw_step_f64")?;
                let (lr, beta1, beta2, eps) =
                    (attrs.learning_rate, attrs.beta1, attrs.beta2, attrs.epsilon);
                let weight_decay = attrs.weight_decay;
                let bias1 = 1.0 - attrs.beta1.powi(step);
                let bias2 = 1.0 - attrs.beta2.powi(step);
                // SAFETY: as above, with f64 elements.
                unsafe {
                    stream
                        .launch_builder(&function)
                        .arg(&mut *p_ptr)
                        .arg(&mut *m_ptr_out)
                        .arg(&mut *v_slice)
                        .arg(&*p_in.buffer.data)
                        .arg(&*grad.buffer.data)
                        .arg(m_src)
                        .arg(v_src)
                        .arg(&lr)
                        .arg(&beta1)
                        .arg(&beta2)
                        .arg(&eps)
                        .arg(&weight_decay)
                        .arg(&bias1)
                        .arg(&bias2)
                        .arg(&numel)
                        .launch(config)
                        .map_err(|error| {
                            Error::Msg(alloc::format!("adamw_step launch failed: {error:?}"))
                        })?;
                }
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    backend: "Cuda",
                    op: "adamw_step",
                });
            }
        }
    }

    let p_out = CudaStorage::new(Arc::new(p_out_buf), p_in.shape.to_vec());
    let m_out = CudaStorage::new(Arc::new(m_out_buf), p_in.shape.to_vec());
    let v_out = CudaStorage::new(Arc::new(v_out_buf), p_in.shape.to_vec());

    Ok((p_out, m_out, v_out))
}

/// Launches the fused standard Adam step kernel on CUDA.
pub(crate) fn launch_adam_step(
    p_in: &CudaStorage,
    grad: &CudaStorage,
    m_in: Option<&CudaStorage>,
    v_in: Option<&CudaStorage>,
    attrs: &AdamAttributes,
) -> Result<(CudaStorage, CudaStorage, CudaStorage)> {
    let len = p_in.buffer.len;
    let device_id = p_in.buffer.device_id;
    let stream = p_in.buffer.device.default_stream();
    let dtype = p_in.buffer.dtype;

    let mut p_out_buf = CudaBuffer {
        len,
        dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            dtype,
            len,
            OperationKind::AdamStep,
        )?),
        device: p_in.buffer.device.clone(),
        device_id,
    };

    let mut m_out_buf = CudaBuffer {
        len,
        dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            dtype,
            len,
            OperationKind::AdamStep,
        )?),
        device: p_in.buffer.device.clone(),
        device_id,
    };

    let mut v_out_buf = CudaBuffer {
        len,
        dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            dtype,
            len,
            OperationKind::AdamStep,
        )?),
        device: p_in.buffer.device.clone(),
        device_id,
    };

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::PushKernelArg;
        use incin_core::tensor::dtype::DTypeId;

        ensure_optimizer_loaded(device_id)?;
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        let numel = len;
        let config = optimizer_launch_config(numel);
        // A moment buffer that was not supplied is a null pointer, which the
        // kernel reads as "start from zero" -- that is what makes the first
        // step work without the caller allocating state it does not have yet.
        let m_ptr = m_in.map(|s| &*s.buffer.data);
        let v_ptr = v_in.map(|s| &*s.buffer.data);
        let empty =
            crate::cuda::ops::alloc_zeroed_bytes(&stream, dtype, len, OperationKind::AdamStep)?;
        let m_src = m_ptr.unwrap_or(&empty);
        let v_src = v_ptr.unwrap_or(&empty);

        let p_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut p_out_buf.data)
            .ok_or_else(|| Error::Msg("fresh optimizer output was unexpectedly shared".into()))?;
        let p_ptr = p_slice as *mut cudarc::driver::CudaSlice<u8>;
        let m_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut m_out_buf.data)
            .ok_or_else(|| Error::Msg("fresh optimizer output was unexpectedly shared".into()))?;
        let m_ptr_out = m_slice as *mut cudarc::driver::CudaSlice<u8>;
        let v_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut v_out_buf.data)
            .ok_or_else(|| Error::Msg("fresh optimizer output was unexpectedly shared".into()))?;

        // Scalars must be passed at the kernel's own precision: an `f64` handed
        // to a `const float` parameter shifts every argument after it, which
        // fails silently rather than loudly.
        let step = i32::try_from(attrs.step).unwrap_or(i32::MAX);
        match dtype.builtin_id() {
            Some(DTypeId::F32) => {
                let function = dispatcher.get_function("optimizer", "adam_step_f32")?;
                #[allow(clippy::cast_possible_truncation)]
                let (lr, beta1, beta2, eps) = (
                    attrs.learning_rate as f32,
                    attrs.beta1 as f32,
                    attrs.beta2 as f32,
                    attrs.epsilon as f32,
                );
                #[allow(clippy::cast_possible_truncation)]
                let bias1 = (1.0 - attrs.beta1.powi(step)) as f32;
                #[allow(clippy::cast_possible_truncation)]
                let bias2 = (1.0 - attrs.beta2.powi(step)) as f32;
                // SAFETY: the three outputs are fresh allocations this call
                // uniquely owns and are distinct from each other; every operand
                // is `numel` f32 elements, matching the launch configuration.
                unsafe {
                    stream
                        .launch_builder(&function)
                        .arg(&mut *p_ptr)
                        .arg(&mut *m_ptr_out)
                        .arg(&mut *v_slice)
                        .arg(&*p_in.buffer.data)
                        .arg(&*grad.buffer.data)
                        .arg(m_src)
                        .arg(v_src)
                        .arg(&lr)
                        .arg(&beta1)
                        .arg(&beta2)
                        .arg(&eps)
                        .arg(&bias1)
                        .arg(&bias2)
                        .arg(&numel)
                        .launch(config)
                        .map_err(|error| {
                            Error::Msg(alloc::format!("adam_step launch failed: {error:?}"))
                        })?;
                }
            }
            Some(DTypeId::F64) => {
                let function = dispatcher.get_function("optimizer", "adam_step_f64")?;
                let (lr, beta1, beta2, eps) =
                    (attrs.learning_rate, attrs.beta1, attrs.beta2, attrs.epsilon);
                let bias1 = 1.0 - attrs.beta1.powi(step);
                let bias2 = 1.0 - attrs.beta2.powi(step);
                // SAFETY: as above, with f64 elements.
                unsafe {
                    stream
                        .launch_builder(&function)
                        .arg(&mut *p_ptr)
                        .arg(&mut *m_ptr_out)
                        .arg(&mut *v_slice)
                        .arg(&*p_in.buffer.data)
                        .arg(&*grad.buffer.data)
                        .arg(m_src)
                        .arg(v_src)
                        .arg(&lr)
                        .arg(&beta1)
                        .arg(&beta2)
                        .arg(&eps)
                        .arg(&bias1)
                        .arg(&bias2)
                        .arg(&numel)
                        .launch(config)
                        .map_err(|error| {
                            Error::Msg(alloc::format!("adam_step launch failed: {error:?}"))
                        })?;
                }
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    backend: "Cuda",
                    op: "adam_step",
                });
            }
        }
    }

    let p_out = CudaStorage::new(Arc::new(p_out_buf), p_in.shape.to_vec());
    let m_out = CudaStorage::new(Arc::new(m_out_buf), p_in.shape.to_vec());
    let v_out = CudaStorage::new(Arc::new(v_out_buf), p_in.shape.to_vec());

    Ok((p_out, m_out, v_out))
}

/// Launches the fused SGD step kernel on CUDA.
pub(crate) fn launch_sgd_step(
    p_in: &CudaStorage,
    grad: &CudaStorage,
    attrs: &SgdAttributes,
) -> Result<CudaStorage> {
    let len = p_in.buffer.len;
    let device_id = p_in.buffer.device_id;
    let stream = p_in.buffer.device.default_stream();
    let dtype = p_in.buffer.dtype;

    let mut p_out_buf = CudaBuffer {
        len,
        dtype,
        data: Arc::new(crate::cuda::ops::alloc_zeroed_bytes(
            &stream,
            dtype,
            len,
            OperationKind::SgdStep,
        )?),
        device: p_in.buffer.device.clone(),
        device_id,
    };

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::PushKernelArg;

        ensure_optimizer_loaded(device_id)?;
        use incin_core::tensor::dtype::DTypeId;

        ensure_optimizer_loaded(device_id)?;
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        let out_slice: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut p_out_buf.data)
            .ok_or_else(|| Error::Msg("fresh optimizer output was unexpectedly shared".into()))?;
        let numel = len;
        let config = optimizer_launch_config(numel);

        // The scalar arguments must be passed at the kernel's own precision.
        // A `f64` handed to a `const float` parameter does not convert -- it
        // shifts every argument after it, so the kernel reads a denormal for the
        // learning rate and garbage for the element count. That failure is
        // silent: the parameters come back unchanged rather than wrong.
        match dtype.builtin_id() {
            Some(DTypeId::F32) => {
                let function = dispatcher.get_function("optimizer", "sgd_step_f32")?;
                #[allow(clippy::cast_possible_truncation)]
                let lr = attrs.learning_rate as f32;
                // SAFETY: the output is a fresh allocation this call uniquely
                // owns, and every operand is `numel` f32 elements, which the
                // launch configuration matches.
                unsafe {
                    stream
                        .launch_builder(&function)
                        .arg(&mut *out_slice)
                        .arg(&*p_in.buffer.data)
                        .arg(&*grad.buffer.data)
                        .arg(&lr)
                        .arg(&numel)
                        .launch(config)
                        .map_err(|error| {
                            Error::Msg(alloc::format!("sgd_step launch failed: {error:?}"))
                        })?;
                }
            }
            Some(DTypeId::F64) => {
                let function = dispatcher.get_function("optimizer", "sgd_step_f64")?;
                let lr = attrs.learning_rate;
                // SAFETY: as above, with f64 elements.
                unsafe {
                    stream
                        .launch_builder(&function)
                        .arg(&mut *out_slice)
                        .arg(&*p_in.buffer.data)
                        .arg(&*grad.buffer.data)
                        .arg(&lr)
                        .arg(&numel)
                        .launch(config)
                        .map_err(|error| {
                            Error::Msg(alloc::format!("sgd_step launch failed: {error:?}"))
                        })?;
                }
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype,
                    backend: "Cuda",
                    op: "sgd_step",
                });
            }
        }
    }

    let p_out = CudaStorage::new(Arc::new(p_out_buf), p_in.shape.to_vec());

    Ok(p_out)
}
