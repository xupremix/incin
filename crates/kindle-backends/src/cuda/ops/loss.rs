use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
pub(crate) fn launch_nll_loss(
    log_sm: &CudaStorage,
    targets: &CudaStorage,
    classes: usize,
) -> Result<CudaStorage> {
    let (b_log_sm, b_targets) = (&*log_sm.buffer, &*targets.buffer);
    let device_id = b_log_sm.device_id;
    let kernel_name = "nll_loss";

    if crate::cuda::gpu::cuda_cache::get_module(device_id, kernel_name).is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        dispatcher.compile_and_load_kernel(
            kernel_name,
            crate::cuda::ops::kernels::LOSS_KERNEL,
            kernel_name,
        )?;
    }

    let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
    let f = dispatcher.get_function(kernel_name, "nll_loss")?;
    let stream = b_log_sm.device.default_stream();

    let batch = targets.shape.iter().product::<usize>();
    let out_numel = batch;

    let mut out_b = CudaBuffer {
        len: out_numel,
        dtype: b_log_sm.dtype,
        data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
        device: b_log_sm.device.clone(),
        device_id,
    };

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: ((batch as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        // Targets are cast to i64 (they come from float conversions or ints)
        let log_sm_ptr = b_log_sm.data.transmute::<f32>(b_log_sm.len).unwrap();
        let targets_ptr = b_targets.data.transmute::<i64>(b_targets.len).unwrap();

        // out_b.data was just allocated above (Arc::new), so it is uniquely owned
        // here (refcount 1) and Arc::get_mut succeeds without cloning first.
        let out_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
            .expect("out_b.data is freshly allocated and uniquely owned here");
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

    Ok(CudaStorage::new(alloc::sync::Arc::new(out_b), vec![batch]))
}
