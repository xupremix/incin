use crate::storage::{NativeBuffer, NativeCudaBuffer, NativeStorage};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
pub const ONE_HOT_KERNEL: &str = r#"
extern "C" __global__ void build_one_hot(
    const long long* targets,
    float* one_hot,
    int batch,
    int classes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        long long class_idx = targets[i];
        if (class_idx >= 0 && class_idx < classes) {
            one_hot[i * classes + class_idx] = 1.0f;
        }
    }
}
"#;

#[cfg(feature = "cuda")]
pub fn launch_one_hot(
    targets: &NativeStorage,
    batch: usize,
    classes: usize,
    like_b: &crate::storage::NativeCudaBuffer,
) -> Result<NativeStorage> {
    if let NativeBuffer::Cuda(t_b) = &*targets.buffer {
        let device_id = t_b.device_id;
        let dispatcher = crate::gpu::NativeCudaDispatcher::new(device_id);

        let kernel_name = "build_one_hot";
        if crate::gpu::cuda_cache::get_module(device_id, kernel_name).is_none() {
            dispatcher.compile_and_load_kernel(kernel_name, ONE_HOT_KERNEL, kernel_name)?;
        }
        let f = dispatcher.get_function(kernel_name, kernel_name)?;
        let stream = t_b.device.default_stream();

        let out_numel = batch * classes;
        let out_b = NativeCudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: t_b.device.clone(),
            device_id,
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((batch as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let targets_i64 = t_b.data.transmute::<i64>(t_b.len).unwrap();
            let mut out_data_arc = out_b.data.clone();
            let out_slice_u8: &mut cudarc::driver::CudaSlice<u8> =
                Arc::get_mut(&mut out_data_arc).unwrap();
            let mut out_f32 = out_slice_u8.transmute_mut::<f32>(out_numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&targets_i64)
                .arg(&mut out_f32)
                .arg(&(batch as i32))
                .arg(&(classes as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("one_hot launch failed: {:?}", e))
                })?;
        }

        let strides = crate::stride::contiguous_strides(&[batch, classes]);
        Ok(NativeStorage {
            buffer: Arc::new(NativeBuffer::Cuda(out_b)),
            shape: vec![batch, classes],
            strides,
            offset: 0,
            id: crate::storage::TensorId::next(),
        })
    } else {
        Err(kindle_core::prelude::Error::Msg(
            "launch_one_hot: targets must be on CUDA".into()
        ))
    }
}
