use crate::cuda::storage::{CudaBuffer, CudaStorage, TensorId};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
const CONCAT_SRC: &str = include_str!("kernels/concat.cu");

#[cfg(feature = "cuda")]
fn ensure_concat_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "concat").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        dispatcher.compile_and_load_kernel("concat", CONCAT_SRC, "concat")?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_concat(tensors: &[&CudaStorage], dim: usize) -> Result<CudaStorage> {
    if tensors.is_empty() {
        return Err(kindle_core::prelude::Error::Msg(
            "concat: empty tensor list".into(),
        ));
    }

    let first_buf = match &*tensors[0].buffer {
        b => b,
        _ => {
            return Err(kindle_core::prelude::Error::Msg(
                "concat: inputs must be CUDA buffers".into(),
            ));
        }
    };
    let device_id = first_buf.device_id;
    ensure_concat_loaded(device_id)?;

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
    let f = dispatcher.get_function("concat", "concat_f32")?;
    let stream = first_buf.device.default_stream();

    let mut out_shape = tensors[0].shape.clone();
    let out_dim_total: usize = tensors.iter().map(|t| t.shape[dim]).sum();
    out_shape[dim] = out_dim_total;

    let total = out_shape.iter().product::<usize>();
    let mut out_b = CudaBuffer {
        len: total,
        data: Arc::new(stream.alloc_zeros::<u8>(total * 4).unwrap()),
        device: first_buf.device.clone(),
        device_id,
    };

    let outer_size: usize = out_shape[0..dim].iter().product();
    let inner_size: usize = if dim + 1 < out_shape.len() {
        out_shape[dim + 1..].iter().product()
    } else {
        1
    };

    let mut current_offset: u32 = 0;
    for t in tensors {
        let t_buf = &*t.buffer;

        let in_dim_size = t.shape[dim];
        let elements = outer_size * in_dim_size * inner_size;
        if elements == 0 {
            current_offset += in_dim_size as u32;
            continue;
        }

        let block_size: u32 = 256;
        let grid_size = (elements as u32 + block_size - 1) / block_size;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let in_f32 = t_buf.data.transmute::<f32>(t_buf.len).unwrap();
            // out_b.data was allocated once before this loop and never cloned, so it
            // stays uniquely owned (refcount 1) across every iteration and
            // Arc::get_mut succeeds without cloning first.
            let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
                .expect("out_b.data is uniquely owned for the lifetime of this loop");
            let mut out_f32 = out_u8.transmute_mut::<f32>(total).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&in_f32)
                .arg(&mut out_f32)
                .arg(&(outer_size as u32))
                .arg(&(in_dim_size as u32))
                .arg(&(out_dim_total as u32))
                .arg(&(inner_size as u32))
                .arg(&current_offset)
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("concat launch failed: {e:?}"))
                })?;
        }

        current_offset += in_dim_size as u32;
    }

    let strides = crate::cpu::stride::contiguous_strides(&out_shape);
    Ok(CudaStorage {
        buffer: Arc::new(out_b),
        shape: out_shape,
        strides,
        offset: 0,
        id: TensorId::next(),
    })
}
