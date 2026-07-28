use alloc::sync::Arc;
use core::ops::Deref;
use incin_core::exec::{Alignment, TensorMeta};
use incin_core::prelude::{DTypeId, DeviceId, Error, OperationKind, Result};
use wgpu::util::DeviceExt;

use crate::wgpu::device::get_device_state;

/// Raw GPU buffer.  All fields are intentionally private — layout and usage
/// flags are an implementation detail and must not be relied upon by callers.
pub(crate) struct WgpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) size: usize,
}

impl WgpuBuffer {
    pub(crate) fn new_zeros(size_bytes: usize) -> Arc<Self> {
        let state = get_device_state();
        let buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WgpuBuffer"),
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Arc::new(Self {
            buffer,
            size: size_bytes,
        })
    }

    /// Allocate a zeroed buffer sized for `elements` values of `dtype`.
    ///
    /// The dtype decides the width and the multiplication is checked, so an
    /// element count whose byte length overflows `usize` is reported instead of
    /// wrapping into an undersized buffer that a shader would then write past.
    pub(crate) fn new_zeros_for(
        dtype: DTypeId,
        elements: usize,
        operation: OperationKind,
    ) -> Result<Arc<Self>> {
        Ok(Self::new_zeros(crate::bytes::byte_len(
            dtype, elements, operation,
        )?))
    }

    pub(crate) fn from_slice<T: bytemuck::Pod>(data: &[T]) -> Arc<Self> {
        let state = get_device_state();
        let bytes = bytemuck::cast_slice(data);
        let buffer = state
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WgpuBuffer Init"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        Arc::new(Self {
            buffer,
            size: bytes.len(),
        })
    }

    pub(crate) fn to_vec<T: bytemuck::Pod>(&self) -> Vec<T> {
        let state = get_device_state();
        let staging = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: self.size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ToVec"),
            });
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size as u64);
        state.queue.submit(core::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        state.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }
}

use core::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(0);

impl TensorId {
    pub fn next() -> Self {
        TensorId(NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// Storage type used by `WgpuBackendImpl` as `Backend::Storage<K>`.
/// The internal buffer and shape are private to prevent construction of
/// invalid states from outside this crate.
#[derive(Clone)]
pub struct WgpuStorage {
    pub(crate) buffer: Arc<WgpuBuffer>,
    pub(crate) meta: TensorMeta,
    pub(crate) id: TensorId,
}

impl Deref for WgpuStorage {
    type Target = TensorMeta;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl WgpuStorage {
    pub(crate) fn try_new(buffer: Arc<WgpuBuffer>, shape: Vec<usize>) -> Result<Self> {
        let capacity = buffer
            .size
            .checked_div(DTypeId::F32.element_size())
            .ok_or_else(|| Error::Msg("WGPU element size must be nonzero".into()))?;
        if buffer.size % DTypeId::F32.element_size() != 0 {
            return Err(Error::Msg(format!(
                "WGPU buffer byte size {} is not a whole number of f32 elements",
                buffer.size
            )));
        }
        let meta = TensorMeta::contiguous(
            shape.as_slice().into(),
            DTypeId::F32,
            DeviceId::wgpu(0),
            Alignment::of::<f32>(),
            capacity,
        )
        .map_err(|error| Error::Msg(format!("invalid WGPU storage metadata: {error}")))?;
        Ok(Self {
            buffer,
            meta,
            id: TensorId::next(),
        })
    }

    pub(crate) fn try_new_packed_q8(buffer: Arc<WgpuBuffer>, shape: Vec<usize>) -> Result<Self> {
        let logical_elements = shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .ok_or_else(|| Error::Msg("WGPU Q8_0 logical element count overflowed".into()))?;
        if !logical_elements.is_multiple_of(32) {
            return Err(Error::Msg(format!(
                "WGPU Q8_0 storage requires a multiple of 32 logical elements, got {logical_elements}"
            )));
        }
        let expected_bytes = (logical_elements / 32)
            .checked_mul(34)
            .ok_or_else(|| Error::Msg("WGPU Q8_0 packed byte length overflowed".into()))?;
        if buffer.size != expected_bytes {
            return Err(Error::Msg(format!(
                "WGPU Q8_0 buffer has {} bytes, expected {expected_bytes}",
                buffer.size
            )));
        }
        let meta = TensorMeta::contiguous(
            shape.as_slice().into(),
            DTypeId::Q8_0,
            DeviceId::wgpu(0),
            Alignment::BYTE,
            logical_elements,
        )
        .map_err(|error| Error::Msg(format!("invalid WGPU Q8_0 storage metadata: {error}")))?;
        Ok(Self {
            buffer,
            meta,
            id: TensorId::next(),
        })
    }

    pub(crate) fn new(buffer: Arc<WgpuBuffer>, shape: Vec<usize>) -> Self {
        Self::try_new(buffer, shape)
            .expect("backend-created contiguous WGPU storage must match its allocation")
    }

    pub fn metadata(&self) -> &TensorMeta {
        &self.meta
    }
}

pub(crate) fn scatter_into_zeros(
    out_shape: &[usize],
    start: &[usize],
    grad_out: &WgpuStorage,
) -> WgpuStorage {
    use crate::wgpu::dispatch;
    let out_n = crate::wgpu::backend::num_elements(out_shape);
    // The tape's backward closure is `Fn(&WgpuStorage) -> Vec<WgpuStorage>`, so
    // this path cannot report. `out_shape` is the recorded shape of a tensor
    // that was already allocated in the forward pass, so its byte length has
    // already been computed successfully once; the same discipline as
    // `WgpuStorage::new` below applies. Making the whole backward signature
    // fallible belongs with the explicit gradient context in GRD-001.
    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::Storage)
        .expect("a shape allocated in the forward pass must size in the backward pass");
    let in_n = crate::wgpu::backend::num_elements(&grad_out.shape) as u32;

    let params = dispatch::prepare_shape_params(
        1, // paste
        in_n,
        out_shape,
        &grad_out.shape,
        start,
    );
    dispatch::dispatch_shape(&grad_out.buffer, &out_buf, &params);
    WgpuStorage::new(out_buf, out_shape.to_vec())
}
