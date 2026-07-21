use crate::cpu::storage::BlockQ8_0;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
pub fn launch_quantize(inp: &CudaStorage) -> Result<CudaStorage> {
    if true {
        let b_inp = &*inp.buffer;
        let device_id = b_inp.device_id;

        if crate::cuda::gpu::cuda_cache::get_module(device_id, "quant").is_none() {
            let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
            dispatcher.compile_and_load_kernel(
                "quant",
                crate::cuda::ops::kernels::QUANT_KERNEL,
                "quantize_q8_0",
            )?;
        }

        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        let f = dispatcher.get_function("quant", "quantize_q8_0")?;
        let stream = b_inp.device.default_stream();

        let n = inp.shape.iter().product::<usize>();
        if n % 32 != 0 {
            return Err(kindle_core::prelude::Error::Msg(alloc::format!(
                "quantize requires length multiple of 32, got {}",
                n
            )));
        }

        let num_blocks = n / 32;
        let out_bytes = num_blocks * core::mem::size_of::<BlockQ8_0>();

        let mut out_buf = CudaBuffer {
            len: num_blocks,
            data: Arc::new(stream.alloc_zeros::<u8>(out_bytes).unwrap()),
            device: b_inp.device.clone(),
            device_id,
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((num_blocks as u32 + 255) / 256, 1, 1),
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
                .arg(&(num_blocks as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(alloc::format!(
                        "quantize_q8_0 launch failed: {:?}",
                        e
                    ))
                })?;
        }

        return Ok(CudaStorage::new(
            alloc::sync::Arc::new(out_buf),
            inp.shape.clone(),
        ));
    }
    Err(kindle_core::prelude::Error::Msg(
        "quantize requires cuda buffers".to_string(),
    ))
}

#[cfg(feature = "cuda")]
pub fn launch_dequantize(inp: &CudaStorage) -> Result<CudaStorage> {
    if true {
        let b_inp = &*inp.buffer;
        let device_id = b_inp.device_id;

        if crate::cuda::gpu::cuda_cache::get_module(device_id, "quant").is_none() {
            let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
            dispatcher.compile_and_load_kernel(
                "quant",
                crate::cuda::ops::kernels::QUANT_KERNEL,
                "dequantize_q8_0",
            )?;
        }

        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        let f = dispatcher.get_function("quant", "dequantize_q8_0")?;
        let stream = b_inp.device.default_stream();

        let n = inp.shape.iter().product::<usize>();
        let num_blocks = b_inp.len;
        let out_numel = n;

        let mut out_buf = CudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: b_inp.device.clone(),
            device_id,
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((num_blocks as u32 + 255) / 256, 1, 1),
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
                .arg(&(num_blocks as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(alloc::format!(
                        "dequantize_q8_0 launch failed: {:?}",
                        e
                    ))
                })?;
        }

        return Ok(CudaStorage::new(
            alloc::sync::Arc::new(out_buf),
            inp.shape.clone(),
        ));
    }
    Err(kindle_core::prelude::Error::Msg(
        "dequantize requires cuda buffers".to_string(),
    ))
}
