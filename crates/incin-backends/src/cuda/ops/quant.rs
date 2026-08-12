use super::alloc_zeroed_bytes;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use crate::quant::BlockQ8_0;
use alloc::sync::Arc;
use incin_core::prelude::Result;
use incin_core::prelude::{OperationKind, ShapeBuf};

#[cfg(feature = "cuda")]
pub(crate) fn launch_quantize(inp: &CudaStorage) -> Result<CudaStorage> {
    let b_inp = &*inp.buffer;
    let device_id = b_inp.device_id;

    if crate::cuda::gpu::cuda_cache::get_module(device_id, "quant").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel(
            "quantize_q8_0",
            crate::cuda::ops::kernels::QUANT_KERNEL,
            "quant",
        )?;
    }

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("quant", "quantize_q8_0")?;
    let stream = b_inp.device.default_stream();

    let n = ShapeBuf::from_slice(&inp.shape).checked_numel(OperationKind::Storage)?;
    if n % 32 != 0 {
        return Err(incin_core::prelude::Error::Msg(alloc::format!(
            "quantize requires length multiple of 32, got {}",
            n
        )));
    }

    let num_blocks = n / 32;
    debug_assert_eq!(
        core::mem::size_of::<BlockQ8_0>(),
        incin_core::prelude::DTypeId::Q8_0
            .encoding()
            .bytes_per_block()
    );

    // `len` is a logical element count for every other CUDA buffer, and
    // `CudaStorage` bounds-checks the shape's span against it. Recording the
    // block count here instead made a `[2, 32]` quantized tensor claim a
    // two-element allocation, which is what the hardware run rejected.
    let mut out_buf = CudaBuffer {
        len: n,
        dtype: incin_core::prelude::DTypeId::Q8_0.descriptor(),
        data: Arc::new(alloc_zeroed_bytes(
            &stream,
            incin_core::prelude::DTypeId::Q8_0.descriptor(),
            n,
            OperationKind::Storage,
        )?),
        device: b_inp.device.clone(),
        device_id,
    };

    let num_blocks_u32 = crate::cuda::checked_u32(num_blocks, "CUDA quantization grid dimension")?;
    let num_blocks_i32 = crate::cuda::checked_i32(num_blocks, "CUDA quantization block count")?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks_u32.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let inp_ptr = b_inp.data.transmute::<f32>(b_inp.len).unwrap();
        // out_buf.data was just allocated above (Arc::new), so it is uniquely
        // owned here (refcount 1) and Arc::get_mut succeeds without cloning first.
        let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_buf.data)
            .expect("out_buf.data is freshly allocated and uniquely owned here");

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&inp_ptr)
            .arg(out_u8)
            .arg(&num_blocks_i32)
            .launch(cfg)
            .map_err(|e| {
                incin_core::prelude::Error::Msg(alloc::format!(
                    "quantize_q8_0 launch failed: {:?}",
                    e
                ))
            })?;
    }

    Ok(CudaStorage::new(
        alloc::sync::Arc::new(out_buf),
        inp.shape.to_vec(),
    ))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_dequantize(inp: &CudaStorage) -> Result<CudaStorage> {
    let b_inp = &*inp.buffer;
    let device_id = b_inp.device_id;

    if crate::cuda::gpu::cuda_cache::get_module(device_id, "quant").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
        dispatcher.compile_and_load_kernel(
            "dequantize_q8_0",
            crate::cuda::ops::kernels::QUANT_KERNEL,
            "quant",
        )?;
    }

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id)?;
    let f = dispatcher.get_function("quant", "dequantize_q8_0")?;
    let stream = b_inp.device.default_stream();

    let n = ShapeBuf::from_slice(&inp.shape).checked_numel(OperationKind::Storage)?;
    if n % 32 != 0 {
        return Err(incin_core::prelude::Error::Msg(alloc::format!(
            "dequantize requires length multiple of 32, got {}",
            n
        )));
    }
    // Derived from the shape rather than read from the buffer, so the block
    // count follows the tensor being dequantized rather than the unit its
    // allocation happened to be recorded in.
    let num_blocks = n / 32;
    let out_numel = n;

    let mut out_buf = CudaBuffer {
        len: out_numel,
        dtype: incin_core::prelude::DTypeId::F32.descriptor(),
        data: Arc::new(alloc_zeroed_bytes(
            &stream,
            incin_core::prelude::DTypeId::F32.descriptor(),
            out_numel,
            OperationKind::Storage,
        )?),
        device: b_inp.device.clone(),
        device_id,
    };

    let num_blocks_u32 =
        crate::cuda::checked_u32(num_blocks, "CUDA dequantization grid dimension")?;
    let num_blocks_i32 = crate::cuda::checked_i32(num_blocks, "CUDA dequantization block count")?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks_u32.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        // out_buf.data was just allocated above (Arc::new), so it is uniquely
        // owned here (refcount 1) and Arc::get_mut succeeds without cloning first.
        let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_buf.data)
            .expect("out_buf.data is freshly allocated and uniquely owned here");
        let mut out_ptr = out_u8.transmute_mut::<f32>(out_numel).unwrap();

        use cudarc::driver::PushKernelArg;
        stream
            .launch_builder(&f)
            .arg(&*b_inp.data)
            .arg(&mut out_ptr)
            .arg(&num_blocks_i32)
            .launch(cfg)
            .map_err(|e| {
                incin_core::prelude::Error::Msg(alloc::format!(
                    "dequantize_q8_0 launch failed: {:?}",
                    e
                ))
            })?;
    }

    Ok(CudaStorage::new(
        alloc::sync::Arc::new(out_buf),
        inp.shape.to_vec(),
    ))
}
