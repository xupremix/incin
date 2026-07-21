use crate::cuda::storage::{CudaBuffer, CudaStorage};
use alloc::sync::Arc;
use kindle_core::prelude::Result;

#[cfg(feature = "cuda")]
const REDUCE_SRC: &str = include_str!("kernels/reduce.cu");

#[cfg(feature = "cuda")]
fn ensure_reduce_loaded(device_id: usize) -> Result<()> {
    if crate::cuda::gpu::cuda_cache::get_module(device_id, "reduce").is_none() {
        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        dispatcher.compile_and_load_kernel("reduce", REDUCE_SRC, "reduce")?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_reduce_op(
    op_name: &str,
    _init_val: &str,
    _update_op: &str,
    storage: &CudaStorage,
    axis: usize,
    keepdim: bool,
) -> Result<CudaStorage> {
    if true {
        let b = &*storage.buffer;
        let device_id = b.device_id;
        ensure_reduce_loaded(device_id)?;

        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        let f = dispatcher.get_function("reduce", op_name)?;
        let stream = b.device.default_stream();

        let mut out_shape = storage.shape.clone();
        if keepdim {
            out_shape[axis] = 1;
        } else {
            out_shape.remove(axis);
        }
        let _out_numel: usize = out_shape.iter().product();
        let ndim = storage.shape.len() as i32;

        let in_shape_h: Vec<i32> = storage.shape.iter().map(|&x| x as i32).collect();
        let in_strides_h: Vec<i32> = storage.strides.iter().map(|&x| x as i32).collect();

        let mut keepdim_shape = storage.shape.clone();
        keepdim_shape[axis] = 1;
        let out_strides_h: Vec<i32> = crate::cpu::stride::contiguous_strides(&keepdim_shape)
            .iter()
            .map(|&x| x as i32)
            .collect();
        let keepdim_shape_h: Vec<i32> = keepdim_shape.iter().map(|&x| x as i32).collect();

        let in_shape_dev = stream.clone_htod(&in_shape_h).unwrap();
        let in_strides_dev = stream.clone_htod(&in_strides_h).unwrap();
        let out_shape_dev = stream.clone_htod(&keepdim_shape_h).unwrap();
        let out_strides_dev = stream.clone_htod(&out_strides_h).unwrap();

        let keepdim_numel: usize = keepdim_shape.iter().product();
        let mut out_b = CudaBuffer {
            len: keepdim_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(keepdim_numel * 4).unwrap()),
            device: b.device.clone(),
            device_id,
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((keepdim_numel as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let in_f32 = b.data.transmute::<f32>(b.len).unwrap();
            // out_b.data was just allocated above (Arc::new), so it is uniquely owned
            // here (refcount 1) and Arc::get_mut succeeds without cloning first.
            let out_slice_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut out_b.data)
                .expect("out_b.data is freshly allocated and uniquely owned here");
            let mut out_f32 = out_slice_u8.transmute_mut::<f32>(keepdim_numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&in_f32)
                .arg(&mut out_f32)
                .arg(&in_shape_dev)
                .arg(&in_strides_dev)
                .arg(&out_shape_dev)
                .arg(&out_strides_dev)
                .arg(&(storage.offset as i32))
                .arg(&0i32)
                .arg(&(axis as i32))
                .arg(&(storage.shape[axis] as i32))
                .arg(&ndim)
                .arg(&(keepdim_numel as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("{op_name} launch failed: {e:?}"))
                })?;
        }

        let out_strides_final = crate::cpu::stride::contiguous_strides(&out_shape);
        let keepdim_storage = CudaStorage {
            buffer: Arc::new(out_b),
            shape: keepdim_shape.clone(),
            strides: crate::cpu::stride::contiguous_strides(&keepdim_shape),
            offset: 0,
            id: crate::cuda::storage::TensorId::next(),
        };

        if keepdim || out_shape == keepdim_shape {
            Ok(keepdim_storage)
        } else {
            Ok(CudaStorage {
                shape: out_shape.clone(),
                strides: out_strides_final,
                ..keepdim_storage
            })
        }
    } else {
        Err(kindle_core::prelude::Error::Msg(
            "launch_reduce_op: input must be on CUDA".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_reduce_with_indices_op(
    op_name: &str,
    _init_val: &str,
    _update_op: &str,
    storage: &CudaStorage,
    axis: usize,
    _keepdim: bool,
) -> Result<(CudaStorage, CudaStorage)> {
    if true {
        let b = &*storage.buffer;
        let device_id = b.device_id;
        ensure_reduce_loaded(device_id)?;

        let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(device_id);
        let f = dispatcher.get_function("reduce", op_name)?;
        let stream = b.device.default_stream();

        let mut out_shape = storage.shape.clone();
        out_shape[axis] = 1;
        let out_numel: usize = out_shape.iter().product();
        let ndim = storage.shape.len() as i32;

        let in_shape_h: Vec<i32> = storage.shape.iter().map(|&x| x as i32).collect();
        let in_strides_h: Vec<i32> = storage.strides.iter().map(|&x| x as i32).collect();
        let out_strides_h: Vec<i32> = crate::cpu::stride::contiguous_strides(&out_shape)
            .iter()
            .map(|&x| x as i32)
            .collect();
        let out_shape_h: Vec<i32> = out_shape.iter().map(|&x| x as i32).collect();

        let in_shape_dev = stream.clone_htod(&in_shape_h).unwrap();
        let in_strides_dev = stream.clone_htod(&in_strides_h).unwrap();
        let out_shape_dev = stream.clone_htod(&out_shape_h).unwrap();
        let out_strides_dev = stream.clone_htod(&out_strides_h).unwrap();

        let mut val_b = CudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: b.device.clone(),
            device_id,
        };
        let mut idx_b = CudaBuffer {
            len: out_numel,
            data: Arc::new(stream.alloc_zeros::<u8>(out_numel * 4).unwrap()),
            device: b.device.clone(),
            device_id,
        };

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((out_numel as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let in_f32 = b.data.transmute::<f32>(b.len).unwrap();
            // val_b.data / idx_b.data were just allocated above (Arc::new), so they
            // are uniquely owned here (refcount 1) and Arc::get_mut succeeds without
            // cloning first.
            let val_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut val_b.data)
                .expect("val_b.data is freshly allocated and uniquely owned here");
            let mut val_f32 = val_u8.transmute_mut::<f32>(out_numel).unwrap();

            let idx_u8: &mut cudarc::driver::CudaSlice<u8> = Arc::get_mut(&mut idx_b.data)
                .expect("idx_b.data is freshly allocated and uniquely owned here");
            let mut idx_u32 = idx_u8.transmute_mut::<u32>(out_numel).unwrap();

            use cudarc::driver::PushKernelArg;
            stream
                .launch_builder(&f)
                .arg(&in_f32)
                .arg(&mut val_f32)
                .arg(&mut idx_u32)
                .arg(&in_shape_dev)
                .arg(&in_strides_dev)
                .arg(&out_shape_dev)
                .arg(&out_strides_dev)
                .arg(&(storage.offset as i32))
                .arg(&0i32)
                .arg(&(axis as i32))
                .arg(&(storage.shape[axis] as i32))
                .arg(&ndim)
                .arg(&(out_numel as i32))
                .launch(cfg)
                .map_err(|e| {
                    kindle_core::prelude::Error::Msg(format!("{op_name} launch failed: {e:?}"))
                })?;
        }

        let out_strides = crate::cpu::stride::contiguous_strides(&out_shape);
        let val_storage = CudaStorage {
            buffer: Arc::new(val_b),
            shape: out_shape.clone(),
            strides: out_strides.clone(),
            offset: 0,
            id: crate::cuda::storage::TensorId::next(),
        };
        let idx_storage = CudaStorage {
            buffer: Arc::new(idx_b),
            shape: out_shape,
            strides: out_strides,
            offset: 0,
            id: crate::cuda::storage::TensorId::next(),
        };
        Ok((val_storage, idx_storage))
    } else {
        Err(kindle_core::prelude::Error::Msg(
            "launch_reduce_with_indices_op: input must be on CUDA".into(),
        ))
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn launch_reduce_with_indices_host(
    op_name: &str,
    init_val: &str,
    update_op: &str,
    storage: &CudaStorage,
    axis: usize,
    keepdim: bool,
) -> Result<(CudaStorage, Vec<usize>)> {
    let (val_storage, idx_storage) =
        launch_reduce_with_indices_op(op_name, init_val, update_op, storage, axis, keepdim)?;

    let host_indices = if true {
        let b = &*idx_storage.buffer;
        let stream = b.device.default_stream();
        let dev_u32 = unsafe { b.data.transmute::<u32>(b.len).unwrap() };
        stream.clone_dtoh(&dev_u32).unwrap()
    } else {
        vec![]
    };

    let usize_indices = host_indices.into_iter().map(|x| x as usize).collect();
    Ok((val_storage, usize_indices))
}
