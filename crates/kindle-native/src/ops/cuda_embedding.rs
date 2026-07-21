use crate::storage::{NativeBuffer, NativeCudaBuffer, NativeStorage};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
pub const EMBEDDING_FWD_KERNEL: &str = r#"
extern "C" __global__ void embedding_fwd(
    const float* weight,
    const long long* indices,
    float* output,
    int vocab_size,
    int hidden_size,
    int total_indices
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = total_indices * hidden_size;
    if (out_idx < total_out) {
        int token_pos = out_idx / hidden_size;
        int h = out_idx % hidden_size;
        long long row_idx = indices[token_pos];
        if (row_idx >= 0 && row_idx < vocab_size) {
            output[out_idx] = weight[row_idx * hidden_size + h];
        } else {
            output[out_idx] = 0.0f;
        }
    }
}
"#;

#[cfg(feature = "cuda")]
pub const EMBEDDING_BWD_KERNEL: &str = r#"
extern "C" __global__ void embedding_bwd(
    const float* grad_out,
    const long long* indices,
    float* grad_weight,
    int vocab_size,
    int hidden_size,
    int total_indices
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = total_indices * hidden_size;
    if (out_idx < total_out) {
        int token_pos = out_idx / hidden_size;
        int h = out_idx % hidden_size;
        long long row_idx = indices[token_pos];
        if (row_idx >= 0 && row_idx < vocab_size) {
            atomicAdd(&grad_weight[row_idx * hidden_size + h], grad_out[out_idx]);
        }
    }
}
"#;

#[cfg(feature = "cuda")]
pub fn launch_embedding_fwd(
    indices: &NativeStorage,
    weight: &NativeStorage,
) -> Result<NativeStorage> {
    let vocab_size = weight.shape[0];
    let hidden_size = weight.shape[1];
    let total_indices: usize = indices.shape.iter().product();
    let out_numel = total_indices * hidden_size;

    if let (NativeBuffer::Cuda(w_b), NativeBuffer::Cuda(idx_b)) = (&*weight.buffer, &*indices.buffer) {
        let device_id = w_b.device_id;
        let dispatcher = crate::gpu::NativeCudaDispatcher::new(device_id);

        let kernel_name = "embedding_fwd";
        if crate::gpu::cuda_cache::get_module(device_id, kernel_name).is_none() {
            dispatcher.compile_and_load_kernel(kernel_name, EMBEDDING_FWD_KERNEL, kernel_name)?;
        }
        let f = dispatcher.get_function(kernel_name, kernel_name)?;
        let stream = w_b.device.default_stream();

        let out_b = NativeCudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: w_b.device.clone(),
            device_id,
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((out_numel as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let weight_f32 = w_b.data.transmute::<f32>(w_b.len).unwrap();
            let indices_i64 = idx_b.data.transmute::<i64>(idx_b.len).unwrap();
            let mut out_data_arc = out_b.data.clone();
            let out_slice_u8: &mut cudarc::driver::CudaSlice<u8> =
                Arc::get_mut(&mut out_data_arc).unwrap();
            let mut out_f32 = out_slice_u8.transmute_mut::<f32>(out_numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&weight_f32)
                .arg(&indices_i64)
                .arg(&mut out_f32)
                .arg(&(vocab_size as i32))
                .arg(&(hidden_size as i32))
                .arg(&(total_indices as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("embedding_fwd launch failed: {:?}", e))
                })?;
        }

        let mut out_shape = indices.shape.clone();
        out_shape.push(hidden_size);
        let out_strides = crate::stride::contiguous_strides(&out_shape);

        Ok(NativeStorage {
            buffer: Arc::new(NativeBuffer::Cuda(out_b)),
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            id: crate::storage::TensorId::next(),
        })
    } else {
        Err(kindle_core::prelude::Error::Msg(
            "launch_embedding_fwd: both weight and indices must be on CUDA".into()
        ))
    }
}

#[cfg(feature = "cuda")]
pub fn launch_embedding_bwd(
    grad_out: &NativeStorage,
    indices_dev: &cudarc::driver::CudaSlice<u8>,
    device: &Arc<cudarc::driver::CudaContext>,
    device_id: usize,
    vocab_size: usize,
    hidden_size: usize,
    total_indices: usize,
) -> Result<NativeStorage> {
    let out_numel = total_indices * hidden_size;
    let w_numel = vocab_size * hidden_size;

    let dispatcher = crate::gpu::NativeCudaDispatcher::new(device_id);
    let kernel_name = "embedding_bwd";
    if crate::gpu::cuda_cache::get_module(device_id, kernel_name).is_none() {
        dispatcher.compile_and_load_kernel(kernel_name, EMBEDDING_BWD_KERNEL, kernel_name)?;
    }
    let f = dispatcher.get_function(kernel_name, kernel_name)?;
    let stream = device.default_stream();

    let grad_w_b = NativeCudaBuffer {
        len: w_numel,
        data: Arc::new(stream.alloc_zeros::<u8>(w_numel * 4).unwrap()),
        device: device.clone(),
        device_id,
    };

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: ((out_numel as u32 + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    if let NativeBuffer::Cuda(go_b) = &*grad_out.buffer {
        unsafe {
            let grad_out_f32 = go_b.data.transmute::<f32>(go_b.len).unwrap();
            let indices_i64 = indices_dev.transmute::<i64>(total_indices).unwrap();
            let mut gw_data_arc = grad_w_b.data.clone();
            let gw_slice_u8: &mut cudarc::driver::CudaSlice<u8> =
                Arc::get_mut(&mut gw_data_arc).unwrap();
            let mut gw_f32 = gw_slice_u8.transmute_mut::<f32>(w_numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&grad_out_f32)
                .arg(&indices_i64)
                .arg(&mut gw_f32)
                .arg(&(vocab_size as i32))
                .arg(&(hidden_size as i32))
                .arg(&(total_indices as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("embedding_bwd launch failed: {:?}", e))
                })?;
        }
    }

    let w_strides = crate::stride::contiguous_strides(&[vocab_size, hidden_size]);
    Ok(NativeStorage {
        buffer: Arc::new(NativeBuffer::Cuda(grad_w_b)),
        shape: vec![vocab_size, hidden_size],
        strides: w_strides,
        offset: 0,
        id: crate::storage::TensorId::next(),
    })
}
