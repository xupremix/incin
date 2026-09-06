//! CUDA launchers for Q8_0 quantization kernels.
//!
//! `quantize_q8_0`'s kernel reads its input as `const float*`, so only f32
//! storage may reach it. `launch_quantize_q8_0` refuses anything else loudly
//! rather than reinterpreting narrower/wider bytes as floats; extending the
//! kernel to bf16/f16/f64 is future work, not a silent reinterpretation.

use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use incin_core::error::{Error, Result};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DTypeId;

#[cfg(feature = "cuda")]
const QUANT_SRC: &str = include_str!("kernels/quant.cu");

#[cfg(feature = "cuda")]
fn ensure_quant_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "quant").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel("quant", QUANT_SRC, "quant")?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_quantize_q8_0(input: &CudaStorage) -> Result<CudaStorage> {
    // The kernel below reads `const float*`: a bf16/f16 input would hand the
    // transmute fewer bytes than it asks for (driver panic), an f64 input
    // would hand it more (first half read as f32, wrong values, no error).
    // Both become the same typed refusal here.
    crate::cuda::backend::cuda_require_f32(input.buffer.dtype, "quantize")?;
    let total_numel = input.shape.iter().product::<usize>();
    if total_numel % 32 != 0 {
        return Err(Error::Msg(format!(
            "CUDA Q8_0 quantize requires element count to be a multiple of 32, got {total_numel}"
        )));
    }
    let num_blocks = total_numel / 32;
    let byte_len = num_blocks * 34; // 2-byte half scale + 32-byte int8 values

    let device_id = input.buffer.device_id;
    ensure_quant_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("quant", "quantize_q8_0")?;
    let stream = input.buffer.device.default_stream();

    let mut out_buffer = CudaBuffer {
        len: total_numel,
        dtype: DTypeId::Q8_0.descriptor(),
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(byte_len)
                .map_err(|e| Error::Msg(format!("CUDA quantize allocation failed: {e:?}")))?,
        ),
        device: input.buffer.device.clone(),
        device_id,
    };

    if num_blocks > 0 {
        let block_size = 256u32;
        let grid_size =
            crate::cuda::checked_u32(num_blocks, "quantize launch grid")?.div_ceil(block_size);
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let num_blocks_i32 = crate::cuda::checked_i32(num_blocks, "quantize block count")?;

        // SAFETY: Launches Q8_0 quantize kernel with verified block count and output byte allocation.
        unsafe {
            let out_u8 = Arc::get_mut(&mut out_buffer.data)
                .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&function)
                .arg(&*input.buffer.data)
                .arg(&mut *out_u8)
                .arg(&num_blocks_i32)
                .launch(config)
                .map_err(|e| Error::Msg(format!("CUDA quantize launch failed: {e:?}")))?;
        }
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), input.shape.to_vec()))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_dequantize_q8_0(input: &CudaStorage) -> Result<CudaStorage> {
    let total_numel = input.shape.iter().product::<usize>();
    if total_numel % 32 != 0 {
        return Err(Error::Msg(format!(
            "CUDA Q8_0 dequantize requires element count to be a multiple of 32, got {total_numel}"
        )));
    }
    let num_blocks = total_numel / 32;
    let byte_len = crate::bytes::byte_len(DTypeId::F32, total_numel, OperationKind::Storage)?;

    let device_id = input.buffer.device_id;
    ensure_quant_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("quant", "dequantize_q8_0")?;
    let stream = input.buffer.device.default_stream();

    let mut out_buffer = CudaBuffer {
        len: total_numel,
        dtype: DTypeId::F32.descriptor(),
        data: Arc::new(
            stream
                .alloc_zeros::<u8>(byte_len)
                .map_err(|e| Error::Msg(format!("CUDA dequantize allocation failed: {e:?}")))?,
        ),
        device: input.buffer.device.clone(),
        device_id,
    };

    if num_blocks > 0 {
        let block_size = 256u32;
        let grid_size =
            crate::cuda::checked_u32(num_blocks, "quantize launch grid")?.div_ceil(block_size);
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let num_blocks_i32 = crate::cuda::checked_i32(num_blocks, "quantize block count")?;

        // SAFETY: Launches Q8_0 dequantize kernel with verified block count and output byte allocation.
        unsafe {
            let out_u8 = Arc::get_mut(&mut out_buffer.data)
                .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&function)
                .arg(&*input.buffer.data)
                .arg(&mut *out_u8)
                .arg(&num_blocks_i32)
                .launch(config)
                .map_err(|e| Error::Msg(format!("CUDA dequantize launch failed: {e:?}")))?;
        }
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), input.shape.to_vec()))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_quantized_matmul_q8_0(
    lhs: &CudaStorage,
    rhs: &CudaStorage,
) -> Result<CudaStorage> {
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        return Err(Error::Msg(format!(
            "CUDA quantized_matmul requires 2D matrices, got lhs {:?}, rhs {:?}",
            lhs.shape, rhs.shape
        )));
    }
    let (m, k) = (lhs.shape[0], lhs.shape[1]);
    let (k2, n) = (rhs.shape[0], rhs.shape[1]);
    if k != k2 {
        return Err(Error::Msg(format!(
            "CUDA quantized_matmul dimension mismatch: lhs shape {:?}, rhs shape {:?}",
            lhs.shape, rhs.shape
        )));
    }
    if k % 32 != 0 {
        return Err(Error::Msg(format!(
            "CUDA quantized_matmul requires K to be a multiple of 32, got {k}"
        )));
    }

    let out_shape = vec![m, n];
    let out_numel = m * n;
    let byte_len = crate::bytes::byte_len(DTypeId::F32, out_numel, OperationKind::MatMul)?;

    let device_id = lhs.buffer.device_id;
    ensure_quant_loaded(device_id)?;
    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let function = dispatcher.get_function("quant", "quantized_matmul_q8_0")?;
    let stream = lhs.buffer.device.default_stream();

    let mut out_buffer =
        CudaBuffer {
            len: out_numel,
            dtype: DTypeId::F32.descriptor(),
            data: Arc::new(stream.alloc_zeros::<u8>(byte_len).map_err(|e| {
                Error::Msg(format!("CUDA quantized matmul allocation failed: {e:?}"))
            })?),
            device: lhs.buffer.device.clone(),
            device_id,
        };

    if out_numel > 0 {
        let grid_x = crate::cuda::checked_u32(n, "quantized matmul grid x")?.div_ceil(32);
        let grid_y = crate::cuda::checked_u32(m, "quantized matmul grid y")?.div_ceil(32);
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: 0,
        };
        let m_i32 = crate::cuda::checked_i32(m, "quantized matmul m")?;
        let n_i32 = crate::cuda::checked_i32(n, "quantized matmul n")?;
        let k_i32 = crate::cuda::checked_i32(k, "quantized matmul k")?;

        // SAFETY: Launches Q8_0 quantized matmul kernel with verified matrix dimensions.
        unsafe {
            let out_u8 = Arc::get_mut(&mut out_buffer.data)
                .ok_or_else(|| Error::Msg("Output buffer unexpectedly shared".into()))?;
            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&function)
                .arg(&*lhs.buffer.data)
                .arg(&*rhs.buffer.data)
                .arg(&mut *out_u8)
                .arg(&m_i32)
                .arg(&n_i32)
                .arg(&k_i32)
                .launch(config)
                .map_err(|e| Error::Msg(format!("CUDA quantized matmul launch failed: {e:?}")))?;
        }
    }

    Ok(CudaStorage::new(Arc::new(out_buffer), out_shape))
}
