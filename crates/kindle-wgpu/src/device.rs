use alloc::sync::Arc;
use wgpu::{Instance, Adapter, Device, Queue, InstanceDescriptor, Backends, RequestAdapterOptions};

pub(crate) struct WgpuDeviceState {
    pub(crate) instance: Instance,
    pub(crate) adapter: Adapter,
    pub(crate) device: Device,
    pub(crate) queue: Queue,
}

static WGPU_STATE: std::sync::OnceLock<Arc<WgpuDeviceState>> = std::sync::OnceLock::new();

pub(crate) fn get_device_state() -> Arc<WgpuDeviceState> {
    WGPU_STATE.get_or_init(|| {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).expect("No suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Kindle WgpuDevice"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )).expect("Failed to create WGPU device");

        Arc::new(WgpuDeviceState { instance, adapter, device, queue })
    }).clone()
}
