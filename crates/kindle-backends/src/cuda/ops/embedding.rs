use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
const EMBEDDING_SRC: &str = include_str!("kernels/embedding.cu");

#[cfg(feature = "cuda")]
fn ensure_embedding_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "embedding").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        dispatcher.compile_and_load_kernel("embedding", EMBEDDING_SRC, "embedding")?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_embedding_forward(
    weight: &CudaStorage,
    indices: &CudaStorage,
) -> Result<CudaStorage> {
    if true {
        let (w_b, i_b) = (&*weight.buffer, &*indices.buffer);
        let device_id = w_b.device_id;
        if device_id != i_b.device_id {
            return Err(kindle_core::prelude::Error::Msg(
                "embedding_forward: weight and indices must be on the same CUDA device".into(),
            ));
        }
        ensure_embedding_loaded(device_id)?;

        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        let f = dispatcher.get_function("embedding", "embedding_forward")?;
        let stream = w_b.device.default_stream();

        let vocab_size = weight.shape[0];
        let hidden_size = weight.shape[1];
        let num_indices = indices.shape.iter().product::<usize>();

        let mut out_shape = indices.shape.clone();
        out_shape.push(hidden_size);
        let out_numel = num_indices * hidden_size;

        let mut out_b = CudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: w_b.device.clone(),
            device_id,
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_indices as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let i_ptr = i_b.data.transmute::<i64>(i_b.len).unwrap();
            let w_ptr = w_b.data.transmute::<f32>(w_b.len).unwrap();
            // out_b.data was just allocated above (Arc::new), so it is uniquely owned
            // here (refcount 1) and Arc::get_mut succeeds without cloning first.
            let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
                .expect("out_b.data is freshly allocated and uniquely owned here");
            let mut out_ptr = out_u8.transmute_mut::<f32>(out_numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&i_ptr)
                .arg(&w_ptr)
                .arg(&mut out_ptr)
                .arg(&(num_indices as usize))
                .arg(&(vocab_size as usize))
                .arg(&(hidden_size as usize))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!(
                        "embedding_forward launch failed: {e:?}"
                    ))
                })?;
        }

        let out_strides = crate::cpu::stride::contiguous_strides(&out_shape);
        Ok(CudaStorage {
            buffer: Arc::new(out_b),
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            id: crate::cuda::storage::TensorId::next(),
        })
    } else {
        Err(kindle_core::prelude::Error::Msg(
            "launch_embedding_forward: inputs must be on CUDA".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_embedding_backward(
    grad_output: &CudaStorage,
    indices: &CudaStorage,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<CudaStorage> {
    if true {
        let (go_b, i_b) = (&*grad_output.buffer, &*indices.buffer);
        let device_id = go_b.device_id;
        ensure_embedding_loaded(device_id)?;

        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        let f = dispatcher.get_function("embedding", "embedding_backward")?;
        let stream = go_b.device.default_stream();

        let num_indices = indices.shape.iter().product::<usize>();
        let out_numel = vocab_size * hidden_size;

        let mut grad_w_b = CudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: go_b.device.clone(),
            device_id,
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_indices as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let go_ptr = go_b.data.transmute::<f32>(go_b.len).unwrap();
            let i_ptr = i_b.data.transmute::<i64>(i_b.len).unwrap();
            // grad_w_b.data was just allocated above (Arc::new), so it is uniquely
            // owned here (refcount 1) and Arc::get_mut succeeds without cloning first.
            let gw_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut grad_w_b.data)
                .expect("grad_w_b.data is freshly allocated and uniquely owned here");
            let mut gw_ptr = gw_u8.transmute_mut::<f32>(out_numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&go_ptr)
                .arg(&i_ptr)
                .arg(&mut gw_ptr)
                .arg(&(num_indices as usize))
                .arg(&(hidden_size as usize))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!(
                        "embedding_backward launch failed: {e:?}"
                    ))
                })?;
        }

        let out_shape = vec![vocab_size, hidden_size];
        let out_strides = crate::cpu::stride::contiguous_strides(&out_shape);
        Ok(CudaStorage {
            buffer: Arc::new(grad_w_b),
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            id: crate::cuda::storage::TensorId::next(),
        })
    } else {
        Err(kindle_core::prelude::Error::Msg(
            "launch_embedding_backward: inputs must be on CUDA".into(),
        ))
    }
}
