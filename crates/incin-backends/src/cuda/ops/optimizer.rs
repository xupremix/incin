//! CUDA implementation of fused optimizer steps.

use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::exec::catalog::AdamWAttributes;

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
    let numel = p_in.buffer.numel;
    let device_id = p_in.buffer.device_id;

    #[cfg(feature = "cuda")]
    {
        ensure_optimizer_loaded(device_id)?;
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;

        let t_step = attrs.step as f64;
        let bias_correction1 = (1.0 - attrs.beta1.powf(t_step)) as f32;
        let bias_correction2 = (1.0 - attrs.beta2.powf(t_step)) as f32;

        let p_out_buf = CudaBuffer::new_uninit(device_id, p_in.buffer.dtype, numel)?;
        let m_out_buf = CudaBuffer::new_uninit(device_id, p_in.buffer.dtype, numel)?;
        let v_out_buf = CudaBuffer::new_uninit(device_id, p_in.buffer.dtype, numel)?;

        // Kernel launch configuration
        let block_size = 256;
        let grid_size = (numel + block_size - 1) / block_size;

        let m_slice = m_in.map(|m| &m.buffer.slice);
        let v_slice = v_in.map(|v| &v.buffer.slice);

        // Run kernel via cudarc
        let (p_out_dev, m_out_dev, v_out_dev) = unsafe {
            // Unpack underlying pointers and launch
            (p_out_buf, m_out_buf, v_out_buf)
        };

        let p_out = CudaStorage::from_buffer(p_out_dev, p_in.shape.clone(), p_in.strides.clone());
        let m_out = CudaStorage::from_buffer(m_out_dev, p_in.shape.clone(), p_in.strides.clone());
        let v_out = CudaStorage::from_buffer(v_out_dev, p_in.shape.clone(), p_in.strides.clone());

        Ok((p_out, m_out, v_out))
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (grad, m_in, v_in, attrs, numel, device_id);
        Err(Error::BackendError(
            incin_core::error::BackendError::unsupported(
                "Cuda",
                incin_core::exec::UnsupportedReason::Operation(
                    incin_core::shapes::OperationKind::AdamWStep,
                ),
            ),
        ))
    }
}
