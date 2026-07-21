use crate::cuda::ops::kernels::NORM_KERNEL;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use cudarc::driver::PushKernelArg;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
fn ensure_norm_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "norm").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        dispatcher.compile_and_load_kernel("norm", NORM_KERNEL, "layer_norm")?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_layer_norm(
    t: &CudaStorage,
    weight: &CudaStorage,
    bias: Option<&CudaStorage>,
    eps: f32,
) -> Result<CudaStorage> {
    let w_buf = &*weight.buffer;

    let device_id = w_buf.device_id;
    ensure_norm_loaded(device_id)?;

    let t_buf = &*t.buffer;

    let has_bias = if bias.is_some() { 1i32 } else { 0i32 };

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
    let f = dispatcher.get_function("norm", "layer_norm")?;
    let stream = w_buf.device.default_stream();

    let rank = t.shape.len();
    let norm_size = t.shape[rank - 1];
    let batch_size = t.shape[..rank - 1].iter().product::<usize>();

    let mut out_b = CudaBuffer {
        len: t_buf.len,
        data: Arc::new(stream.alloc_zeros::<u8>(t_buf.len * 4).unwrap()),
        device: w_buf.device.clone(),
        device_id,
    };

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    unsafe {
        let t_ptr = t_buf.data.transmute::<f32>(t_buf.len).unwrap();
        let w_ptr = w_buf.data.transmute::<f32>(w_buf.len).unwrap();

        // out_b.data was just allocated above (Arc::new), so it is uniquely owned
        // here (refcount 1) and Arc::get_mut succeeds without cloning first.
        let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .expect("out_b.data is freshly allocated and uniquely owned here");
        let mut out_ptr = out_u8.transmute_mut::<f32>(t_buf.len).unwrap();

        if let Some(b) = bias {
            if true {
                let b_buf = &*b.buffer;
                let b_ptr = b_buf.data.transmute::<f32>(b_buf.len).unwrap();
                stream
                    .launch_builder(&f)
                    .arg(&t_ptr)
                    .arg(&w_ptr)
                    .arg(&b_ptr)
                    .arg(&mut out_ptr)
                    .arg(&eps)
                    .arg(&(norm_size as i32))
                    .arg(&{ has_bias })
                    .arg(&(batch_size as i32))
                    .launch(cfg)
                    .map_err(|e| {
                        kindle_core::prelude::Error::Msg(alloc::format!(
                            "layer_norm launch error: {:?}",
                            e
                        ))
                    })?;
            }
        } else {
            // Re-transmute w_ptr to use as a dummy pointer for b_ptr
            let b_ptr = w_buf.data.transmute::<f32>(w_buf.len).unwrap();
            stream
                .launch_builder(&f)
                .arg(&t_ptr)
                .arg(&w_ptr)
                .arg(&b_ptr)
                .arg(&mut out_ptr)
                .arg(&eps)
                .arg(&(norm_size as i32))
                .arg(&{ has_bias })
                .arg(&(batch_size as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(alloc::format!(
                        "layer_norm launch error: {:?}",
                        e
                    ))
                })?;
        }
    }

    Ok(CudaStorage::new(
        alloc::sync::Arc::new(out_b),
        t.shape.clone(),
    ))
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_batch_norm(
    t: &CudaStorage,
    w: Option<&CudaStorage>,
    b: Option<&CudaStorage>,
    rm: Option<&CudaStorage>,
    rv: Option<&CudaStorage>,
    eps: f32,
) -> Result<CudaStorage> {
    let t_buf = &*t.buffer;

    let device_id = t_buf.device_id;
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "batch_norm").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        dispatcher.compile_and_load_kernel("batch_norm", NORM_KERNEL, "batch_norm")?;
    }

    let has_w = if w.is_some() { 1i32 } else { 0i32 };
    let has_b = if b.is_some() { 1i32 } else { 0i32 };
    let has_rm = if rm.is_some() { 1i32 } else { 0i32 };
    let has_rv = if rv.is_some() { 1i32 } else { 0i32 };

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
    let f = dispatcher.get_function("batch_norm", "batch_norm")?;
    let stream = t_buf.device.default_stream();

    let rank = t.shape.len();
    let channel_dim = if rank > 1 { 1 } else { 0 };
    let num_channels = t.shape[channel_dim];

    let spatial_size = if rank > 2 {
        t.shape[2..].iter().product::<usize>()
    } else {
        1
    };

    let total_elements = t.shape.iter().product::<usize>();

    let mut out_b = CudaBuffer {
        len: t_buf.len,
        data: Arc::new(stream.alloc_zeros::<u8>(t_buf.len * 4).unwrap()),
        device: t_buf.device.clone(),
        device_id,
    };

    let block_size = 256;
    let grid_size = (total_elements as u32).div_ceil(block_size);

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let t_ptr = t_buf.data.transmute::<f32>(t_buf.len).unwrap();

        // out_b.data was just allocated above (Arc::new), so it is uniquely owned
        // here (refcount 1) and Arc::get_mut succeeds without cloning first.
        let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .expect("out_b.data is freshly allocated and uniquely owned here");
        let mut out_ptr = out_u8.transmute_mut::<f32>(t_buf.len).unwrap();

        let w_ptr = w
            .and_then(|s| {
                if true {
                    let buf = &*s.buffer;
                    Some(buf.data.transmute::<f32>(buf.len).unwrap())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| t_buf.data.transmute::<f32>(t_buf.len).unwrap());
        let b_ptr = b
            .and_then(|s| {
                if true {
                    let buf = &*s.buffer;
                    Some(buf.data.transmute::<f32>(buf.len).unwrap())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| t_buf.data.transmute::<f32>(t_buf.len).unwrap());
        let rm_ptr = rm
            .and_then(|s| {
                if true {
                    let buf = &*s.buffer;
                    Some(buf.data.transmute::<f32>(buf.len).unwrap())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| t_buf.data.transmute::<f32>(t_buf.len).unwrap());
        let rv_ptr = rv
            .and_then(|s| {
                if true {
                    let buf = &*s.buffer;
                    Some(buf.data.transmute::<f32>(buf.len).unwrap())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| t_buf.data.transmute::<f32>(t_buf.len).unwrap());

        stream
            .launch_builder(&f)
            .arg(&t_ptr)
            .arg(&w_ptr)
            .arg(&b_ptr)
            .arg(&rm_ptr)
            .arg(&rv_ptr)
            .arg(&mut out_ptr)
            .arg(&eps)
            .arg(&(num_channels as i32))
            .arg(&(spatial_size as i32))
            .arg(&(total_elements as i32))
            .arg(&{ has_w })
            .arg(&{ has_b })
            .arg(&{ has_rm })
            .arg(&{ has_rv })
            .launch(cfg)
            .map_err(|e| {
                kindle_core::prelude::Error::Msg(alloc::format!("batch_norm launch error: {:?}", e))
            })?;
    }

    Ok(CudaStorage::new(
        alloc::sync::Arc::new(out_b),
        t.shape.clone(),
    ))
}
