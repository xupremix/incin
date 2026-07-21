use crate::cpu::storage::{CpuBuffer, CpuCudaBuffer, CpuStorage};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
pub fn launch_nll_loss(
    log_sm: &CpuStorage,
    targets: &CpuStorage,
    classes: usize,
) -> Result<CpuStorage> {
    if let (CpuBuffer::Cuda(b_log_sm), CpuBuffer::Cuda(b_targets)) = (&*log_sm.buffer, &*targets.buffer) {
        let device_id = b_log_sm.device_id;
        let kernel_name = "nll_loss";
        
        if crate::cpu::gpu::cuda_cache::get_module(device_id, kernel_name).is_none() {
            let dispatcher = crate::cpu::gpu::CpuCudaDispatcher::new(device_id);
            dispatcher.compile_and_load_kernel(kernel_name, crate::cpu::ops::cuda_kernels::LOSS_KERNEL, kernel_name)?;
        }

        let dispatcher = crate::cpu::gpu::CpuCudaDispatcher::new(device_id);
        let f = dispatcher.get_function(kernel_name, "nll_loss")?;
        let stream = b_log_sm.device.default_stream();

        let batch = targets.shape.iter().product::<usize>();
        let out_numel = batch;

        let out_b = CpuCudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: b_log_sm.device.clone(),
            device_id,
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((batch as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            // Targets are cast to i64 (they come from float conversions or ints)
            let log_sm_ptr = b_log_sm.data.transmute::<f32>(b_log_sm.len).unwrap();
            let targets_ptr = b_targets.data.transmute::<i64>(b_targets.len).unwrap();
            
            let mut out_data_arc = out_b.data.clone();
            let out_u8: &mut cudarc::driver::CudaSlice<u8> =
                Arc::get_mut(&mut out_data_arc).unwrap();
            let mut out_ptr = out_u8.transmute_mut::<f32>(out_numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&log_sm_ptr)
                .arg(&targets_ptr)
                .arg(&mut out_ptr)
                .arg(&(batch as i32))
                .arg(&(classes as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(alloc::format!("nll_loss launch failed: {:?}", e))
                })?;
        }

        return Ok(CpuStorage::from_contiguous(
            CpuBuffer::Cuda(out_b),
            vec![batch],
        ));
    }
    Err(kindle_core::prelude::Error::Msg("nll_loss requires cuda buffers".to_string()))
}
