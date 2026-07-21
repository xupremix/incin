use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

/// Auto-generated documentation for ELEMENTWISE_UNARY_TEMPLATE.
pub const ELEMENTWISE_UNARY_TEMPLATE: &str = r#"
extern "C" __global__ void unary_op_{OP_NAME}(
    /// Auto-generated documentation for float.
    const float* input,
    float* output,
    /// Auto-generated documentation for int.
    const int* shape,
    /// Auto-generated documentation for int.
    const int* strides,
    int offset,
    int numel,
    int ndim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int flat_idx = offset;
        int temp = idx;
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_idx = temp % shape[i];
            temp /= shape[i];
            flat_idx += dim_idx * strides[i];
        }
        float x = input[flat_idx];
        float out_val = {OP};
        output[idx] = out_val;
    }
}
"#;

/// Auto-generated documentation for ELEMENTWISE_BINARY_TEMPLATE.
pub const ELEMENTWISE_BINARY_TEMPLATE: &str = r#"
extern "C" __global__ void binary_op_{OP_NAME}(
    /// Auto-generated documentation for float.
    const float* lhs,
    /// Auto-generated documentation for float.
    const float* rhs,
    float* output,
    /// Auto-generated documentation for int.
    const int* out_shape,
    /// Auto-generated documentation for int.
    const int* lhs_shape,
    /// Auto-generated documentation for int.
    const int* rhs_shape,
    /// Auto-generated documentation for int.
    const int* lhs_strides,
    /// Auto-generated documentation for int.
    const int* rhs_strides,
    int lhs_offset,
    int rhs_offset,
    int numel,
    int ndim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int temp = idx;
        int lhs_flat = lhs_offset;
        int rhs_flat = rhs_offset;
        
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_idx = temp % out_shape[i];
            temp /= out_shape[i];
            
            int lhs_dim = lhs_shape[i];
            int lhs_dim_idx = (lhs_dim == 1) ? 0 : dim_idx;
            lhs_flat += lhs_dim_idx * lhs_strides[i];
            
            int rhs_dim = rhs_shape[i];
            int rhs_dim_idx = (rhs_dim == 1) ? 0 : dim_idx;
            rhs_flat += rhs_dim_idx * rhs_strides[i];
        }
        
        float a = lhs[lhs_flat];
        float b = rhs[rhs_flat];
        float out_val = {OP};
        output[idx] = out_val;
    }
}
"#;

#[cfg(feature = "cuda")]
/// Auto-generated documentation for launch_unary_op.
pub fn launch_unary_op(op_name: &str, op_expr: &str, t: &CudaStorage) -> Result<CudaStorage> {
    if true {
        let b = &*t.buffer;
        let device_id = b.device_id;
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);

        let kernel_name = format!("unary_op_{}", op_name);
        if crate::cuda::gpu::cuda_cache::get_module(device_id, &kernel_name).is_none() {
            let kernel_src = ELEMENTWISE_UNARY_TEMPLATE
                .replace("{OP_NAME}", op_name)
                .replace("{OP}", op_expr);
            dispatcher.compile_and_load_kernel(&kernel_name, &kernel_src, &kernel_name)?;
        }

        let f = dispatcher.get_function(&kernel_name, &kernel_name)?;
        let numel: usize = t.shape.iter().product();
        let stream = b.device.default_stream();

        let mut out_b = CudaBuffer {
            len: numel,
            data: Arc::new(stream.alloc_zeros::<u8>(numel * 4).unwrap()),
            device: b.device.clone(),
            device_id: device_id,
        };

        let ndim = t.shape.len();

        // Populate shapes/strides
        // Normally you'd htod_copy, we do a simplistic sync transfer here
        let shape_i32: Vec<i32> = t.shape.iter().map(|&x| x as i32).collect();
        let strides_i32: Vec<i32> = t.strides.iter().map(|&x| x as i32).collect();

        // This is a naive implementation, ideally you want a quick memcpy
        let shape_dev = b.device.default_stream().clone_htod(&shape_i32).unwrap();
        let strides_dev = b.device.default_stream().clone_htod(&strides_i32).unwrap();

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((numel as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let input_f32 = b.data.transmute::<f32>(b.len).unwrap();
            // out_b.data was just allocated above (Arc::new), so it is uniquely owned
            // here (refcount 1) and Arc::get_mut succeeds without cloning first.
            let out_slice_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
                .expect("out_b.data is freshly allocated and uniquely owned here");
            let mut out_f32 = out_slice_u8.transmute_mut::<f32>(numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&input_f32)
                .arg(&mut out_f32)
                .arg(&shape_dev)
                .arg(&strides_dev)
                .arg(&(t.offset as i32))
                .arg(&(numel as i32))
                .arg(&(ndim as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("Kernel launch failed: {:?}", e))
                })?;
        }

        Ok(CudaStorage::new(
            alloc::sync::Arc::new(out_b),
            t.shape.clone(),
        ))
    } else {
        Err(kindle_core::prelude::Error::Msg("Not a CUDA buffer".into()))
    }
}

#[cfg(feature = "cuda")]
/// Auto-generated documentation for launch_binary_op.
pub fn launch_binary_op(
    op_name: &str,
    op_expr: &str,
    lhs: &CudaStorage,
    rhs: &CudaStorage,
    out_shape: &[usize],
) -> Result<CudaStorage> {
    if true {
        let (lhs_b, rhs_b) = (&*lhs.buffer, &*rhs.buffer);
        let device_id = lhs_b.device_id;
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);

        let kernel_name = format!("binary_op_{}", op_name);
        if crate::cuda::gpu::cuda_cache::get_module(device_id, &kernel_name).is_none() {
            let kernel_src = ELEMENTWISE_BINARY_TEMPLATE
                .replace("{OP_NAME}", op_name)
                .replace("{OP}", op_expr);
            dispatcher.compile_and_load_kernel(&kernel_name, &kernel_src, &kernel_name)?;
        }

        let f = dispatcher.get_function(&kernel_name, &kernel_name)?;
        let numel: usize = out_shape.iter().product();
        let stream = lhs_b.device.default_stream();

        let mut out_b = CudaBuffer {
            len: numel,
            data: Arc::new(stream.alloc_zeros::<u8>(numel * 4).unwrap()),
            device: lhs_b.device.clone(),
            device_id: device_id,
        };

        let ndim = out_shape.len();

        let mut lhs_padded_shape = vec![1i32; ndim];
        let mut rhs_padded_shape = vec![1i32; ndim];
        let mut lhs_padded_strides = vec![0i32; ndim];
        let mut rhs_padded_strides = vec![0i32; ndim];

        let lhs_offset = ndim.saturating_sub(lhs.shape.len());
        for i in 0..lhs.shape.len() {
            lhs_padded_shape[lhs_offset + i] = lhs.shape[i] as i32;
            lhs_padded_strides[lhs_offset + i] = lhs.strides[i] as i32;
        }
        let rhs_offset = ndim.saturating_sub(rhs.shape.len());
        for i in 0..rhs.shape.len() {
            rhs_padded_shape[rhs_offset + i] = rhs.shape[i] as i32;
            rhs_padded_strides[rhs_offset + i] = rhs.strides[i] as i32;
        }

        let out_shape_i32: Vec<i32> = out_shape.iter().map(|&x| x as i32).collect();

        let out_shape_dev = lhs_b
            .device
            .default_stream()
            .clone_htod(&out_shape_i32)
            .unwrap();
        let lhs_shape_dev = lhs_b
            .device
            .default_stream()
            .clone_htod(&lhs_padded_shape)
            .unwrap();
        let rhs_shape_dev = lhs_b
            .device
            .default_stream()
            .clone_htod(&rhs_padded_shape)
            .unwrap();
        let lhs_strides_dev = lhs_b
            .device
            .default_stream()
            .clone_htod(&lhs_padded_strides)
            .unwrap();
        let rhs_strides_dev = lhs_b
            .device
            .default_stream()
            .clone_htod(&rhs_padded_strides)
            .unwrap();

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((numel as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let lhs_f32 = lhs_b.data.transmute::<f32>(lhs_b.len).unwrap();
            let rhs_f32 = rhs_b.data.transmute::<f32>(rhs_b.len).unwrap();
            // out_b.data was just allocated above (Arc::new), so it is uniquely owned
            // here (refcount 1) and Arc::get_mut succeeds without cloning first.
            let out_slice_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
                .expect("out_b.data is freshly allocated and uniquely owned here");
            let mut out_f32 = out_slice_u8.transmute_mut::<f32>(numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&lhs_f32)
                .arg(&rhs_f32)
                .arg(&mut out_f32)
                .arg(&out_shape_dev)
                .arg(&lhs_shape_dev)
                .arg(&rhs_shape_dev)
                .arg(&lhs_strides_dev)
                .arg(&rhs_strides_dev)
                .arg(&(lhs.offset as i32))
                .arg(&(rhs.offset as i32))
                .arg(&(numel as i32))
                .arg(&(ndim as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("Kernel launch failed: {:?}", e))
                })?;
        }

        Ok(CudaStorage::new(
            alloc::sync::Arc::new(out_b),
            out_shape.to_vec(),
        ))
    } else {
        Err(kindle_core::prelude::Error::Msg("Not a CUDA buffer".into()))
    }
}
