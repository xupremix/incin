use kindle_core::prelude::{Backend, DType, NumericOps, Result};
use crate::NativeBackend;
use crate::storage::{NativeBuffer, NativeStorage, NativeCudaBuffer};
use alloc::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice};

/// Auto-generated documentation for test_cuda_add.
pub fn test_cuda_add() -> Result<()> {
    let device = CudaDevice::new(0).unwrap();
    let data_a: Vec<f32> = vec![1.0, 2.0];
    let data_b: Vec<f32> = vec![3.0, 4.0];
    let slice_a = device.htod_sync_copy(&data_a).unwrap();
    let slice_b = device.htod_sync_copy(&data_b).unwrap();
    
    let buf_a = NativeBuffer::Cuda(NativeCudaBuffer {
        len: 2,
        data: Arc::new(unsafe { std::mem::transmute(slice_a) }),
        device: device.clone(),
        device_id: 0,
    });
    
    let storage_a = NativeStorage::from_contiguous(buf_a, vec![2]);
    let storage_b = NativeStorage::from_contiguous(NativeBuffer::Cuda(NativeCudaBuffer {
        len: 2,
        data: Arc::new(unsafe { std::mem::transmute(slice_b) }),
        device: device.clone(),
        device_id: 0,
    }), vec![2]);
    
    let _out = NativeBackend::<f32, kindle_core::prelude::Cuda>::add::<f32>(&storage_a, &storage_b).unwrap();
    Ok(())
}
