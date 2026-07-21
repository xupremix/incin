use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(0);

impl TensorId {
    pub fn next() -> Self {
        TensorId(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug)]
pub struct CudaBuffer {
    pub len: usize,
    pub data: Arc<cudarc::driver::CudaSlice<u8>>,
    pub device: Arc<cudarc::driver::CudaContext>,
    pub device_id: usize,
}

impl PartialEq for CudaBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && Arc::ptr_eq(&self.data, &other.data)
    }
}

impl Clone for CudaBuffer {
    fn clone(&self) -> Self {
        CudaBuffer {
            len: self.len,
            data: self.data.clone(),
            device: self.device.clone(),
            device_id: self.device_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaStorage {
    pub buffer: Arc<CudaBuffer>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub id: TensorId,
}

impl CudaStorage {
    pub fn new(buffer: Arc<CudaBuffer>, shape: Vec<usize>) -> Self {
        let ndim = shape.len();
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        Self {
            buffer,
            shape,
            strides,
            id: TensorId::next(),
        }
    }
}
