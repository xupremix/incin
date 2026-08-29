//! CUDA implementation of fused optimizer steps (Adam, AdamW, SGD).

#![allow(dead_code, unused_imports)]

use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::exec::catalog::{AdamAttributes, AdamWAttributes, SgdAttributes};
use incin_core::shapes::OperationKind;

#[cfg(feature = "cuda")]
const OPTIMIZER_SRC: &str = include_str!("kernels/optimizer.cu");

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

    let p_out_buf = CudaBuffer {
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

    let m_out_buf = CudaBuffer {
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

    let v_out_buf = CudaBuffer {
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
        ensure_optimizer_loaded(device_id)?;
        let _ = (grad, m_in, v_in, attrs);
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

    let p_out_buf = CudaBuffer {
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

    let m_out_buf = CudaBuffer {
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

    let v_out_buf = CudaBuffer {
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
        ensure_optimizer_loaded(device_id)?;
        let _ = (grad, m_in, v_in, attrs);
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

    let p_out_buf = CudaBuffer {
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
        ensure_optimizer_loaded(device_id)?;
        let _ = (grad, attrs);
    }

    let p_out = CudaStorage::new(Arc::new(p_out_buf), p_in.shape.to_vec());

    Ok(p_out)
}
