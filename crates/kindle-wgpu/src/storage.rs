use alloc::sync::Arc;
use wgpu::util::DeviceExt;

use crate::device::get_device_state;

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

/// Storage type used by `WgpuBackend` as `Backend::Storage<K>`.
/// The internal buffer and shape are private to prevent construction of
/// invalid states from outside this crate.
#[derive(Clone)]
pub struct WgpuStorage {
    pub(crate) buffer: Arc<WgpuBuffer>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl WgpuStorage {
    pub(crate) fn new(buffer: Arc<WgpuBuffer>, shape: Vec<usize>) -> Self {
        // Strides are computed lazily / contiguous-assumed for now.
        let ndim = shape.len();
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        Self {
            buffer,
            shape,
            strides,
        }
    }
}
