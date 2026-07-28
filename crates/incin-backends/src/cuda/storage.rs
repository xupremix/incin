use alloc::sync::Arc;
use core::ops::Deref;
use core::sync::atomic::{AtomicU64, Ordering};

use incin_core::exec::{Alignment, TensorMeta};
use incin_core::prelude::{DTypeId, DeviceId, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(0);

impl TensorId {
    pub(crate) fn next() -> Self {
        TensorId(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[derive(Debug)]
pub struct CudaBuffer {
    pub(crate) len: usize,
    pub(crate) dtype: DTypeId,
    pub(crate) data: Arc<cudarc::driver::CudaSlice<u8>>,
    pub(crate) device: Arc<cudarc::driver::CudaContext>,
    pub(crate) device_id: usize,
}

impl PartialEq for CudaBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.dtype == other.dtype
            && self.device_id == other.device_id
            && Arc::ptr_eq(&self.data, &other.data)
    }
}

impl Clone for CudaBuffer {
    fn clone(&self) -> Self {
        CudaBuffer {
            len: self.len,
            dtype: self.dtype,
            data: self.data.clone(),
            device: self.device.clone(),
            device_id: self.device_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaStorage {
    pub(crate) buffer: Arc<CudaBuffer>,
    pub(crate) meta: TensorMeta,
    pub(crate) id: TensorId,
}

impl Deref for CudaStorage {
    type Target = TensorMeta;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl CudaStorage {
    pub fn try_from_parts(
        buffer: Arc<CudaBuffer>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset_elements: usize,
    ) -> Result<Self> {
        // `CudaBuffer` is byte-addressed (`CudaSlice<u8>`), so the Rust type
        // only proves byte alignment. Higher device-pointer alignment can be
        // introduced later when the allocator exposes that guarantee.
        let meta = TensorMeta::try_new(
            shape.as_slice().into(),
            strides.as_slice().into(),
            offset_elements,
            buffer.dtype,
            DeviceId::cuda(buffer.device_id),
            Alignment::BYTE,
            buffer.len,
        )
        .map_err(|error| Error::Msg(format!("invalid CUDA storage metadata: {error}")))?;
        Ok(Self {
            buffer,
            meta,
            id: TensorId::next(),
        })
    }

    pub fn try_new(buffer: Arc<CudaBuffer>, shape: Vec<usize>) -> Result<Self> {
        let meta = TensorMeta::contiguous(
            shape.as_slice().into(),
            buffer.dtype,
            DeviceId::cuda(buffer.device_id),
            Alignment::BYTE,
            buffer.len,
        )
        .map_err(|error| Error::Msg(format!("invalid CUDA storage metadata: {error}")))?;
        Ok(Self {
            buffer,
            meta,
            id: TensorId::next(),
        })
    }

    pub fn new(buffer: Arc<CudaBuffer>, shape: Vec<usize>) -> Self {
        Self::try_new(buffer, shape)
            .expect("backend-created contiguous CUDA storage must match its allocation")
    }

    pub fn metadata(&self) -> &TensorMeta {
        &self.meta
    }
}
