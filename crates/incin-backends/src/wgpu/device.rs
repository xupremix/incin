use alloc::sync::Arc;
use incin_core::error::{BackendError, Error, Result};
use incin_core::shapes::error::OperationKind;
use wgpu::{Adapter, Backends, Device, Instance, InstanceDescriptor, Queue, RequestAdapterOptions};

pub(crate) struct WgpuDeviceState {
    // Keep both owners alive for the lifetime of the device and queue. WGPU
    // does not expose a read path for either after initialization.
    pub(crate) _instance: Instance,
    pub(crate) _adapter: Adapter,
    pub(crate) device: Device,
    pub(crate) queue: Queue,
}

/// `WGPU_STATE`.
static WGPU_STATE: std::sync::OnceLock<Arc<WgpuDeviceState>> = std::sync::OnceLock::new();

pub(crate) fn get_device_state() -> Arc<WgpuDeviceState> {
    WGPU_STATE
        .get()
        .expect("WGPU state is initialized before an internal buffer is used")
        .clone()
}

pub(crate) fn try_get_device_state() -> Result<Arc<WgpuDeviceState>> {
    if let Some(state) = WGPU_STATE.get() {
        return Ok(state.clone());
    }
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| {
        Error::Backend(BackendError::Execution {
            operation: OperationKind::Storage,
            message: "no suitable WGPU adapter is available".into(),
        })
    })?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Incin WgpuDevice"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .map_err(|error| {
        Error::Backend(BackendError::Execution {
            operation: OperationKind::Storage,
            message: alloc::format!("WGPU device creation failed: {error}").into(),
        })
    })?;
    let state = Arc::new(WgpuDeviceState {
        _instance: instance,
        _adapter: adapter,
        device,
        queue,
    });
    let _ = WGPU_STATE.set(state);
    Ok(WGPU_STATE
        .get()
        .expect("the WGPU state was just initialized")
        .clone())
}
