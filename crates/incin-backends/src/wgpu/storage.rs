use alloc::sync::Arc;
use core::ops::Deref;
use incin_core::exec::{Alignment, TensorMeta};
use incin_core::prelude::{DTypeId, DeviceId, Error, OperationKind, Result, ShapeBuf};
use wgpu::util::DeviceExt;

use crate::wgpu::device::{get_device_state, try_get_device_state};

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
        dtype: impl Into<incin_core::tensor::dtype::DTypeDescriptor>,
        elements: usize,
        operation: OperationKind,
    ) -> Result<Arc<Self>> {
        Ok(Self::new_zeros(crate::bytes::byte_len(
            dtype.into(),
            elements,
            operation,
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

    pub(crate) fn try_from_slice<T: bytemuck::Pod>(data: &[T]) -> Result<Arc<Self>> {
        let state = try_get_device_state()?;
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
        Ok(Arc::new(Self {
            buffer,
            size: bytes.len(),
        }))
    }

    pub(crate) fn to_vec<T: bytemuck::Pod>(&self) -> Result<Vec<T>> {
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
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        state.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|error| {
                Error::Backend(incin_core::error::BackendError::Execution {
                    operation: OperationKind::Storage,
                    message: alloc::format!("WGPU map callback was lost: {error}").into(),
                })
            })?
            .map_err(|error| {
                Error::Backend(incin_core::prelude::BackendError::Execution {
                    operation: OperationKind::Storage,
                    message: alloc::format!("WGPU buffer mapping failed: {error}").into(),
                })
            })?;

        let data = slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(result)
    }
}

/// Re-exported from `incin_core::exec::tape` since `GRD-003`: one identity
/// counter serves the whole workspace.
pub use incin_core::exec::TensorId;

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
    pub(crate) fn with_fresh_autograd_identity(mut self) -> Self {
        self.id = incin_core::exec::TensorId::next();
        self
    }

    pub(crate) fn try_new(buffer: Arc<WgpuBuffer>, shape: Vec<usize>) -> Result<Self> {
        let capacity = buffer
            .size
            .checked_div(DTypeId::F32.encoding().bytes_per_block())
            .ok_or_else(|| Error::Msg("WGPU element size must be nonzero".into()))?;
        if !buffer
            .size
            .is_multiple_of(DTypeId::F32.encoding().bytes_per_block())
        {
            return Err(Error::Msg(format!(
                "WGPU buffer byte size {} is not a whole number of f32 elements",
                buffer.size
            )));
        }
        let meta = TensorMeta::contiguous(
            shape.as_slice().into(),
            DTypeId::F32.descriptor(),
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
        let logical_shape = ShapeBuf::from_slice(&shape);
        let logical_elements = logical_shape.numel().ok_or_else(|| {
            Error::Shape(incin_core::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Storage,
                expression: "Q8_0 logical element count",
            })
        })?;
        let expected_bytes = DTypeId::Q8_0.size_bytes(logical_elements, OperationKind::Storage)?;
        if buffer.size != expected_bytes {
            return Err(Error::Msg(format!(
                "WGPU Q8_0 buffer has {} bytes, expected {expected_bytes}",
                buffer.size
            )));
        }
        let meta = TensorMeta::contiguous(
            shape.as_slice().into(),
            DTypeId::Q8_0.descriptor(),
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
) -> incin_core::error::Result<WgpuStorage> {
    use crate::wgpu::dispatch;
    let out_n = crate::wgpu::backend::num_elements(out_shape)?;
    let out_buf = WgpuBuffer::new_zeros_for(DTypeId::F32, out_n, OperationKind::Storage)?;
    let in_n = crate::wgpu::backend::checked_u32(
        crate::wgpu::backend::num_elements(&grad_out.shape)?,
        "WGPU scatter input element count",
    )?;

    let params = dispatch::prepare_shape_params(
        1, // paste
        in_n,
        out_shape,
        &grad_out.shape,
        start,
    )?;
    dispatch::dispatch_shape(&grad_out.buffer, &out_buf, &params);
    Ok(WgpuStorage::new(out_buf, out_shape.to_vec()))
}
