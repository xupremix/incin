use alloc::sync::Arc;
use core::ops::Deref;
use incin_core::error::{BackendError, Error, Result};
use incin_core::exec::{Alignment, TensorMeta};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::DeviceId;
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};
use wgpu::util::DeviceExt;

use crate::wgpu::device::{get_device_state, try_get_device_state};

/// Physical bytes each element of `dtype` occupies in a WGPU storage buffer.
///
/// Not always `dtype.size_bytes`: `Bool` is stored as one `f32` (0.0/1.0)
/// because WGSL storage buffers cannot hold `bool`, while the `TensorMeta`
/// still reports `Bool` so the API-boundary dtype contract stays honest.
/// The integer widths match their logical widths — a physical `u8`/`u32`/
/// `i64` buffer is exactly what an index operand round-trips through.
pub(crate) fn physical_element_bytes(dtype: DTypeDescriptor) -> Result<usize> {
    match dtype.builtin_id() {
        Some(DTypeId::Bool) => Ok(4),
        Some(DTypeId::U8) => Ok(1),
        Some(DTypeId::U32) => Ok(4),
        Some(DTypeId::I64) => Ok(8),
        Some(DTypeId::F32) => Ok(4),
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Wgpu",
            op: "physical_element_bytes",
        }),
    }
}

/// `elements * physical_element_bytes(dtype)`, checked against overflow.
///
/// Non-`bool` dtypes share their logical width with their physical width, so
/// they route through the one checked [`crate::bytes::byte_len`]; only `bool`
/// needs the widened `f32` packing.
pub(crate) fn physical_byte_len(dtype: DTypeDescriptor, elements: usize) -> Result<usize> {
    if dtype.builtin_id() == Some(DTypeId::Bool) {
        elements.checked_mul(4).ok_or_else(|| {
            Error::Msg(alloc::format!(
                "WGPU buffer byte length overflows usize: {elements} bool elements * 4 bytes"
            ))
        })
    } else {
        crate::bytes::byte_len(dtype, elements, OperationKind::Storage)
    }
}

/// Allocation alignment for a physical WGPU buffer of `dtype`.
///
/// `i64` wants 8; `u8` is byte-aligned; everything else (including
/// `Bool`-as-`f32` and `u32`/`f32`) aligns to 4.
pub(crate) fn physical_alignment(dtype: DTypeDescriptor) -> Alignment {
    match dtype.builtin_id() {
        Some(DTypeId::I64) => Alignment::of::<i64>(),
        Some(DTypeId::U8) => Alignment::BYTE,
        _ => Alignment::of::<f32>(),
    }
}

/// Raw GPU buffer.  All fields are intentionally private - layout and usage
/// flags are an implementation detail and must not be relied upon by callers.
pub(crate) struct WgpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) size: usize,
}

impl WgpuBuffer {
    /// Allocate a zeroed buffer of at least `size_bytes`, padded up to
    /// [`wgpu::COPY_BUFFER_ALIGNMENT`] so later `copy_buffer_to_buffer`
    /// calls (which reject non-4-byte lengths) always have a legal extent.
    /// `Self::size` keeps the *logical* byte length for capacity math.
    pub(crate) fn new_zeros(size_bytes: usize) -> Arc<Self> {
        let state = get_device_state();
        let padded = size_bytes
            .next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT as usize)
            .max(wgpu::COPY_BUFFER_ALIGNMENT as usize);
        let buffer = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WgpuBuffer"),
            size: padded as u64,
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

    /// Allocate a zeroed buffer sized for `elements` physical values of
    /// `dtype`.
    ///
    /// Uses [`physical_byte_len`], not the logical `size_bytes`: a `bool`
    /// allocation is `elements * 4` bytes of physical `f32`, not
    /// `elements * 1`. The multiplication is checked either way, so an
    /// overflowing element count is reported instead of wrapping into an
    /// undersized buffer that a shader would then write past.
    pub(crate) fn new_zeros_for(
        dtype: impl Into<DTypeDescriptor>,
        elements: usize,
        _operation: OperationKind,
    ) -> Result<Arc<Self>> {
        Ok(Self::new_zeros(physical_byte_len(dtype.into(), elements)?))
    }

    pub(crate) fn from_slice<T: bytemuck::Pod>(data: &[T]) -> Arc<Self> {
        let state = get_device_state();
        let bytes = bytemuck::cast_slice(data);
        // `create_buffer_init` already pads the allocation to
        // `COPY_BUFFER_ALIGNMENT`; `Self::size` records the logical length.
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
        // `copy_buffer_to_buffer` requires size % COPY_BUFFER_ALIGNMENT == 0.
        // A logical `u8` payload of 2 bytes is real on the host but illegal as
        // a copy length, so stage (and copy) the 4-byte-rounded extent, then
        // cast only the logical prefix back to `T`.
        const COPY_BUFFER_ALIGNMENT: u64 = wgpu::COPY_BUFFER_ALIGNMENT;
        let logical = self.size as u64;
        if logical == 0 {
            return Ok(Vec::new());
        }
        let copy_size = logical.next_multiple_of(COPY_BUFFER_ALIGNMENT);
        let staging = state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: copy_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ToVec"),
            });
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, copy_size);
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
                Error::Backend(BackendError::Execution {
                    operation: OperationKind::Storage,
                    message: alloc::format!("WGPU buffer mapping failed: {error}").into(),
                })
            })?;

        let data = slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data[..self.size]).to_vec();
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
        Self::try_new_with_dtype(buffer, shape, DTypeId::F32.descriptor())
    }

    /// Wrap `buffer` as contiguous storage of `dtype` for `shape`.
    ///
    /// Capacity is derived from the *physical* element width (see
    /// [`physical_element_bytes`]): a `bool` buffer of `N * 4` bytes has
    /// logical capacity `N`, an `i64` buffer of `N * 8` bytes has capacity
    /// `N`, and so on. A buffer whose size is not a whole number of physical
    /// elements is refused rather than truncated.
    pub(crate) fn try_new_with_dtype(
        buffer: Arc<WgpuBuffer>,
        shape: Vec<usize>,
        dtype: DTypeDescriptor,
    ) -> Result<Self> {
        let width = physical_element_bytes(dtype)?;
        if width == 0 || !buffer.size.is_multiple_of(width) {
            return Err(Error::Msg(alloc::format!(
                "WGPU buffer byte size {} is not a whole number of {dtype:?} elements \
                 (physical width {width})",
                buffer.size
            )));
        }
        let capacity = buffer.size / width;
        let meta = TensorMeta::contiguous(
            shape.as_slice().into(),
            dtype,
            DeviceId::wgpu(0),
            physical_alignment(dtype),
            capacity,
        )
        .map_err(|error| Error::Msg(format!("invalid WGPU storage metadata: {error}")))?;
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

    /// Like [`new`](Self::new), but for a non-`f32` physical dtype.
    ///
    /// # Panics
    ///
    /// If the buffer size is not a whole number of `dtype` elements or the
    /// metadata is otherwise invalid — the same fail-loud contract as
    /// [`new`](Self::new), for backend-created allocations.
    pub(crate) fn new_with_dtype(
        buffer: Arc<WgpuBuffer>,
        shape: Vec<usize>,
        dtype: DTypeDescriptor,
    ) -> Self {
        Self::try_new_with_dtype(buffer, shape, dtype)
            .expect("backend-created contiguous WGPU storage must match its allocation")
    }

    pub fn metadata(&self) -> &TensorMeta {
        &self.meta
    }
}
