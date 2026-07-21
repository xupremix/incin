use crate::storage::{NativeBuffer, NativeCudaBuffer, NativeStorage};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
pub const REDUCE_TEMPLATE: &str = r#"
extern "C" __global__ void reduce_op_{OP_NAME}(
    const float* input,
    float* output,
    const int* in_shape,
    const int* in_strides,
    const int* out_shape,
    const int* out_strides,
    int in_offset,
    int out_offset,
    int reduce_axis,
    int reduce_dim_size,
    int ndim,
    int out_numel
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < out_numel) {
        int temp = out_idx;
        int out_flat = out_offset;
        int base_in_flat = in_offset;
        
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_idx = temp % out_shape[i];
            temp /= out_shape[i];
            out_flat += dim_idx * out_strides[i];
            if (i != reduce_axis) {
                base_in_flat += dim_idx * in_strides[i];
            }
        }
        
        float acc = {INIT_VAL};
        for (int i = 0; i < reduce_dim_size; i++) {
            int in_flat = base_in_flat + i * in_strides[reduce_axis];
            float val = input[in_flat];
            {UPDATE_OP};
        }
        output[out_flat] = acc;
    }
}
"#;

#[cfg(feature = "cuda")]
pub const REDUCE_WITH_INDICES_TEMPLATE: &str = r#"
extern "C" __global__ void reduce_with_indices_op_{OP_NAME}(
    const float* input,
    float* out_vals,
    unsigned int* out_indices,
    const int* in_shape,
    const int* in_strides,
    const int* out_shape,
    const int* out_strides,
    int in_offset,
    int out_offset,
    int reduce_axis,
    int reduce_dim_size,
    int ndim,
    int out_numel
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < out_numel) {
        int temp = out_idx;
        int out_flat = out_offset;
        int base_in_flat = in_offset;
        
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_idx = temp % out_shape[i];
            temp /= out_shape[i];
            out_flat += dim_idx * out_strides[i];
            if (i != reduce_axis) {
                base_in_flat += dim_idx * in_strides[i];
            }
        }
        
        float best_val = {INIT_VAL};
        unsigned int best_idx = 0;
        for (int i = 0; i < reduce_dim_size; i++) {
            int in_flat = base_in_flat + i * in_strides[reduce_axis];
            float val = input[in_flat];
            {UPDATE_OP};
        }
        out_vals[out_flat] = best_val;
        out_indices[out_flat] = best_idx;
    }
}
"#;

#[cfg(feature = "cuda")]
pub fn launch_reduce_op(
    op_name: &str,
    init_val: &str,
    update_op: &str,
    storage: &NativeStorage,
    axis: usize,
    keepdim: bool
) -> Result<NativeStorage> {
    let mut out_shape = storage.shape.clone();
    if keepdim {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }
    
    let out_numel: usize = out_shape.iter().product();
    let out_strides = crate::stride::contiguous_strides(&out_shape);

    if let NativeBuffer::Cuda(b) = &*storage.buffer {
        let device_id = b.device_id;
        let dispatcher = crate::gpu::NativeCudaDispatcher::new(device_id);

        let kernel_name = format!("reduce_op_{}", op_name);
        if crate::gpu::cuda_cache::get_module(device_id, &kernel_name).is_none() {
            let kernel_src = REDUCE_TEMPLATE
                .replace("{OP_NAME}", op_name)
                .replace("{INIT_VAL}", init_val)
                .replace("{UPDATE_OP}", update_op);
            dispatcher.compile_and_load_kernel(&kernel_name, &kernel_src, &kernel_name)?;
        }

        let f = dispatcher.get_function(&kernel_name, &kernel_name)?;
        let stream = b.device.default_stream();

        let mut out_b = NativeCudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: b.device.clone(),
            device_id: device_id,
        };

        // For shapes and strides, use original shape/strides but out_shape for loop if keepdim = true.
        // Wait, the REDUCE_TEMPLATE uses out_shape and out_strides to reconstruct index, and in_strides to find base!
        // We must pass keepdim-equivalent out_shape/strides to the kernel even if keepdim is false, so it has ndims length.
        let mut kernel_out_shape = storage.shape.clone();
        kernel_out_shape[axis] = 1;
        let kernel_out_strides = crate::stride::contiguous_strides(&kernel_out_shape);

        let shape_i32: Vec<i32> = storage.shape.iter().map(|&x| x as i32).collect();
        let in_strides_i32: Vec<i32> = storage.strides.iter().map(|&x| x as i32).collect();
        let out_shape_i32: Vec<i32> = kernel_out_shape.iter().map(|&x| x as i32).collect();
        let out_strides_i32: Vec<i32> = kernel_out_strides.iter().map(|&x| x as i32).collect();

        let in_shape_dev = stream.clone_htod(&shape_i32).unwrap();
        let in_strides_dev = stream.clone_htod(&in_strides_i32).unwrap();
        let out_shape_dev = stream.clone_htod(&out_shape_i32).unwrap();
        let out_strides_dev = stream.clone_htod(&out_strides_i32).unwrap();

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((out_numel as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let input_f32 = b.data.transmute::<f32>(b.len).unwrap();
            let mut out_data_arc = out_b.data.clone();
            let out_slice_u8: &mut cudarc::driver::CudaSlice<u8> =
                Arc::get_mut(&mut out_data_arc).unwrap();
            let mut out_f32 = out_slice_u8.transmute_mut::<f32>(out_numel).unwrap();
            
            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&input_f32)
                .arg(&mut out_f32)
                .arg(&in_shape_dev)
                .arg(&in_strides_dev)
                .arg(&out_shape_dev)
                .arg(&out_strides_dev)
                .arg(&(storage.offset as i32))
                .arg(&(0 as i32))
                .arg(&(axis as i32))
                .arg(&(storage.shape[axis] as i32))
                .arg(&(storage.shape.len() as i32))
                .arg(&(out_numel as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("Kernel launch failed: {:?}", e))
                })?;
        }

        Ok(NativeStorage {
            buffer: Arc::new(NativeBuffer::Cuda(out_b)),
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            id: crate::storage::TensorId::next(),
        })
    } else {
        panic!("launch_reduce_op called on non-CUDA storage")
    }
}

#[cfg(feature = "cuda")]
pub fn launch_reduce_with_indices_op(
    op_name: &str,
    init_val: &str,
    update_op: &str,
    storage: &NativeStorage,
    axis: usize,
    keepdim: bool
) -> Result<(NativeStorage, NativeStorage)> {
    let mut out_shape = storage.shape.clone();
    if keepdim {
        out_shape[axis] = 1;
    } else {
        out_shape.remove(axis);
    }
    
    let out_numel: usize = out_shape.iter().product();
    let out_strides = crate::stride::contiguous_strides(&out_shape);

    if let NativeBuffer::Cuda(b) = &*storage.buffer {
        let device_id = b.device_id;
        let dispatcher = crate::gpu::NativeCudaDispatcher::new(device_id);

        let kernel_name = format!("reduce_with_indices_op_{}", op_name);
        if crate::gpu::cuda_cache::get_module(device_id, &kernel_name).is_none() {
            let kernel_src = REDUCE_WITH_INDICES_TEMPLATE
                .replace("{OP_NAME}", op_name)
                .replace("{INIT_VAL}", init_val)
                .replace("{UPDATE_OP}", update_op);
            dispatcher.compile_and_load_kernel(&kernel_name, &kernel_src, &kernel_name)?;
        }

        let f = dispatcher.get_function(&kernel_name, &kernel_name)?;
        let stream = b.device.default_stream();

        let mut out_b = NativeCudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: b.device.clone(),
            device_id: device_id,
        };
        
        let mut indices_b = NativeCudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: b.device.clone(),
            device_id: device_id,
        };

        let mut kernel_out_shape = storage.shape.clone();
        kernel_out_shape[axis] = 1;
        let kernel_out_strides = crate::stride::contiguous_strides(&kernel_out_shape);

        let shape_i32: Vec<i32> = storage.shape.iter().map(|&x| x as i32).collect();
        let in_strides_i32: Vec<i32> = storage.strides.iter().map(|&x| x as i32).collect();
        let out_shape_i32: Vec<i32> = kernel_out_shape.iter().map(|&x| x as i32).collect();
        let out_strides_i32: Vec<i32> = kernel_out_strides.iter().map(|&x| x as i32).collect();

        let in_shape_dev = stream.clone_htod(&shape_i32).unwrap();
        let in_strides_dev = stream.clone_htod(&in_strides_i32).unwrap();
        let out_shape_dev = stream.clone_htod(&out_shape_i32).unwrap();
        let out_strides_dev = stream.clone_htod(&out_strides_i32).unwrap();

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((out_numel as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let input_f32 = b.data.transmute::<f32>(b.len).unwrap();
            let mut out_data_arc = out_b.data.clone();
            let out_slice_u8: &mut cudarc::driver::CudaSlice<u8> =
                Arc::get_mut(&mut out_data_arc).unwrap();
            let mut out_f32 = out_slice_u8.transmute_mut::<f32>(out_numel).unwrap();
            
            let mut indices_data_arc = indices_b.data.clone();
            let indices_slice_u8: &mut cudarc::driver::CudaSlice<u8> =
                Arc::get_mut(&mut indices_data_arc).unwrap();
            let mut indices_u32 = indices_slice_u8.transmute_mut::<u32>(out_numel).unwrap();
            
            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&input_f32)
                .arg(&mut out_f32)
                .arg(&mut indices_u32)
                .arg(&in_shape_dev)
                .arg(&in_strides_dev)
                .arg(&out_shape_dev)
                .arg(&out_strides_dev)
                .arg(&(storage.offset as i32))
                .arg(&(0 as i32))
                .arg(&(axis as i32))
                .arg(&(storage.shape[axis] as i32))
                .arg(&(storage.shape.len() as i32))
                .arg(&(out_numel as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("Kernel launch failed: {:?}", e))
                })?;
        }

        Ok((
            NativeStorage {
                buffer: Arc::new(NativeBuffer::Cuda(out_b)),
                shape: out_shape.clone(),
                strides: out_strides.clone(),
                offset: 0,
                id: crate::storage::TensorId::next(),
            },
            NativeStorage {
                buffer: Arc::new(NativeBuffer::Cuda(indices_b)),
                shape: out_shape,
                strides: out_strides,
                offset: 0,
                id: crate::storage::TensorId::next(),
            }
        ))
    } else {
        panic!("launch_reduce_with_indices_op called on non-CUDA storage")
    }
}


#[cfg(feature = "cuda")]
pub fn launch_reduce_with_indices_host(
    op_name: &str,
    init_val: &str,
    update_op: &str,
    storage: &crate::storage::NativeStorage,
    axis: usize,
    keepdim: bool
) -> kindle_core::prelude::Result<(crate::storage::NativeStorage, Vec<usize>)> {
    let (val_storage, idx_storage) = launch_reduce_with_indices_op(
        op_name, init_val, update_op, storage, axis, keepdim
    )?;
    
    // Download the u32 indices to host
    let mut host_indices = vec![0u32; idx_storage.shape.iter().product()];
    if let crate::storage::NativeBuffer::Cuda(b) = &*idx_storage.buffer {
        let stream = b.device.default_stream();
        let dev_data = unsafe { b.data.transmute::<u32>(b.len).unwrap() };
        let downloaded = stream.clone_dtoh(&dev_data).unwrap();
        host_indices = downloaded;
    }
    
    let usize_indices = host_indices.into_iter().map(|x| x as usize).collect();
    Ok((val_storage, usize_indices))
}
