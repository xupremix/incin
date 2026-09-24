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
    /// A snapshot of the adapter's limits, taken at device creation (#91).
    ///
    /// Workgroup size, storage-buffer binding counts and binding sizes all
    /// differ by adapter (lavapipe's software limits are not a 680M's), so
    /// the reduction and scan kernels cannot use a constant launch
    /// configuration. Every dispatch in `super::dispatch` sizes itself from
    /// this snapshot rather than from `WG_SIZE`-style constants.
    pub(crate) limits: wgpu::Limits,
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
    // Fail-closed at init (#91): every shader in `shaders/` takes at most
    // five storage bindings (`select.wgsl`'s mask/a/b/out/params), so an
    // adapter that cannot bind five storage buffers per shader stage could
    // never run a dispatch. Refusing here names the adapter limit instead
    // of failing later inside an unrelated kernel launch.
    let limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 5 {
        return Err(Error::Backend(BackendError::Execution {
            operation: OperationKind::Storage,
            message: alloc::format!(
                "WGPU adapter supports only {} storage buffers per shader stage; \
                 this backend needs 5 (select.wgsl)",
                limits.max_storage_buffers_per_shader_stage,
            )
            .into(),
        }));
    }
    let state = Arc::new(WgpuDeviceState {
        _instance: instance,
        _adapter: adapter,
        device,
        queue,
        limits,
    });
    let _ = WGPU_STATE.set(state);
    Ok(WGPU_STATE
        .get()
        .expect("the WGPU state was just initialized")
        .clone())
}

/// The adapter limits snapshot every dispatch sizes itself from (#91).
///
/// Cloned out of the device state so dispatch helpers can read the limits
/// without holding the state lock across a launch.
pub(crate) fn adapter_limits() -> wgpu::Limits {
    get_device_state().limits.clone()
}

/// The 1-D workgroup size every compute dispatch uses: the shader constant
/// (`256`, matching each `@workgroup_size` declaration) clamped down to
/// what the adapter reports in `max_compute_workgroup_size_x`.
///
/// Clamping only ever goes down — a smaller workgroup is always legal — so
/// this cannot over-commit an adapter the way the old constant could on a
/// device with a narrower maximum.
pub(crate) fn workgroup_size_x() -> u32 {
    adapter_limits().max_compute_workgroup_size_x.clamp(1, 256)
}

/// Refuse a 1-D dispatch whose workgroup count exceeds the adapter's
/// `max_compute_workgroups_per_dimension` (#91).
///
/// Every `dispatch_*` helper routes its count through here before touching
/// the queue, so an oversized launch is a named refusal rather than a
/// driver validation error (or worse, silently dropped work).
pub(crate) fn check_workgroups_1d(workgroups: u32) -> Result<()> {
    let max = adapter_limits().max_compute_workgroups_per_dimension;
    if workgroups > max {
        return Err(Error::Backend(BackendError::Execution {
            operation: OperationKind::Storage,
            message: alloc::format!(
                "WGPU dispatch needs {workgroups} workgroups but the adapter allows \
                 only {max} per dimension; split the tensor or run it on the CPU"
            )
            .into(),
        }));
    }
    Ok(())
}

/// Refuse a storage-buffer allocation larger than the adapter's
/// `max_storage_buffer_binding_size` (#91: commonly 128 MiB).
///
/// Large tensors need chunked dispatch; until that exists, attempting one
/// is a named refusal at allocation time rather than a bind-group failure
/// inside an unrelated kernel.
pub(crate) fn check_buffer_bytes(size_bytes: u64) -> Result<()> {
    check_buffer_bytes_against(&adapter_limits(), size_bytes)
}

pub(crate) fn check_buffer_bytes_against(limits: &wgpu::Limits, size_bytes: u64) -> Result<()> {
    let max = u64::from(limits.max_storage_buffer_binding_size);
    if size_bytes > max {
        return Err(Error::Backend(BackendError::Execution {
            operation: OperationKind::Storage,
            message: alloc::format!(
                "WGPU buffer of {size_bytes} bytes exceeds the adapter's \
                 max_storage_buffer_binding_size of {max} bytes; chunked \
                 dispatch is not implemented, so this tensor is refused"
            )
            .into(),
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The launch configuration answers from the adapter, not from a
    /// constant (#91): the workgroup size is clamped into `1..=256` and at
    /// or below the adapter maximum, and both refusals fire on synthetic
    /// inputs without allocating anything.
    ///
    /// Needs a real adapter — same explicit-request contract as the
    /// integration suites, so a missing one fails rather than skips.
    #[test]
    fn launch_config_is_clamped_to_adapter_limits() {
        let state = try_get_device_state().expect("a WGPU adapter must exist");
        let max_ws = state.limits.max_compute_workgroup_size_x;
        let ws = workgroup_size_x();
        assert!(
            (1..=256).contains(&ws),
            "workgroup size {ws} escaped the 1..=256 clamp"
        );
        assert!(
            ws <= max_ws,
            "workgroup size {ws} exceeds the adapter maximum {max_ws}"
        );

        check_workgroups_1d(1).expect("one workgroup must always launch");
        let too_many = state
            .limits
            .max_compute_workgroups_per_dimension
            .saturating_add(1);
        // `saturating_add` guards the (absurd) adapter that reports
        // `u32::MAX`; on every real adapter this is max + 1 and refused.
        if too_many > state.limits.max_compute_workgroups_per_dimension {
            check_workgroups_1d(too_many).expect_err("past-the-maximum workgroups must be refused");
        }

        check_buffer_bytes(4).expect("a tiny buffer must be admitted");
        check_buffer_bytes(u64::from(state.limits.max_storage_buffer_binding_size))
            .expect("exactly-the-maximum must be admitted");
        check_buffer_bytes(
            u64::from(state.limits.max_storage_buffer_binding_size).saturating_add(1),
        )
        .expect_err("past-the-maximum bytes must be refused");
    }
}
