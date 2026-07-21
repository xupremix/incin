use crate::storage::{NativeBuffer, NativeCudaBuffer, NativeStorage};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
const ONE_HOT_SRC: &str = include_str!("kernels/one_hot.cu");

#[cfg(feature = "cuda")]
pub fn launch_one_hot(
    targets: &NativeStorage,
    classes: usize,
) -> Result<NativeStorage> {
    if let NativeBuffer::Cuda(b) = &*targets.buffer {
        let device_id = b.device_id;
        let kernel_name = "one_hot";
        
        if crate::gpu::cuda_cache::get_module(device_id, kernel_name).is_none() {
            let dispatcher = crate::gpu::NativeCudaDispatcher::new(device_id);
            dispatcher.compile_and_load_kernel(kernel_name, ONE_HOT_SRC, kernel_name)?;
        }

        let dispatcher = crate::gpu::NativeCudaDispatcher::new(device_id);
        let f = dispatcher.get_function(kernel_name, "build_one_hot")?;
        let stream = b.device.default_stream();

        let batch = targets.shape.iter().product::<usize>();
        let out_numel = batch * classes;

        let out_b = NativeCudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: b.device.clone(),
            device_id,
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((batch as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            // Targets are cast to i64 (they come from float conversions or ints)
            let in_ptr = b.data.transmute::<i64>(b.len).unwrap();
            let mut out_data_arc = out_b.data.clone();
            let out_u8: &mut cudarc::driver::CudaSlice<u8> =
                Arc::get_mut(&mut out_data_arc).unwrap();
            let mut out_ptr = out_u8.transmute_mut::<f32>(out_numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&in_ptr)
                .arg(&mut out_ptr)
                .arg(&(batch as i32))
                .arg(&(classes as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("build_one_hot launch failed: {e:?}"))
                })?;
        }

        let mut out_shape = targets.shape.clone();
        out_shape.push(classes);
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
            "launch_one_hot: targets must be on CUDA".into(),
        ))
    }
}
