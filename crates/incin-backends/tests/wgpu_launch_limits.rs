//! Limit-aware launch configuration, observed from outside (#91).
//!
//! The workgroup size every dispatch uses comes from the adapter's
//! `max_compute_workgroup_size_x` rather than a constant, workgroup counts
//! are refused past `max_compute_workgroups_per_dimension`, and buffers
//! past `max_storage_buffer_binding_size` are refused at allocation time.
//! None of that is observable from the values a kernel writes, so this
//! suite observes the refusal side instead: a shape whose buffer no adapter
//! could bind is a named error, and the backend keeps working afterwards.
//!
//! Requires a WGPU adapter:
//! `cargo test -p incin-backends --features wgpu --test wgpu_launch_limits`.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::HostInterop;
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;

fn require_wgpu() {
    assert!(
        <TestBackend as HostInterop>::from_bytes::<f32>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .is_ok(),
        "no WGPU adapter, but the `wgpu` feature is enabled"
    );
}

/// Four billion `f32` elements is sixteen gigabytes: no adapter binds that,
/// so the refusal fires on the shape alone without allocating anything.
const OVERSIZE_ELEMENTS: usize = 1_000_000_000;

#[test]
fn an_unbindable_upload_is_refused_by_name() {
    require_wgpu();
    let error = match <TestBackend as HostInterop>::from_bytes::<f32>(
        &[0u8; 4],
        &[OVERSIZE_ELEMENTS],
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    ) {
        Ok(_) => panic!("a 16 GiB upload must be refused, not attempted"),
        Err(error) => error,
    };
    let text = format!("{error:?}");
    assert!(
        text.contains("max_storage_buffer_binding_size"),
        "the refusal must name the adapter limit it hit, got: {text}"
    );
}

#[test]
fn a_bool_oversize_counts_physical_not_logical_bytes() {
    require_wgpu();
    // A `bool` upload expands to `f32` on the device, so one billion
    // logical bool bytes is four physical gigabytes — still unbindable.
    let error = match <TestBackend as HostInterop>::from_bytes::<bool>(
        &[0u8; 4],
        &[OVERSIZE_ELEMENTS],
        DTypeId::Bool.descriptor(),
        &DeviceId::wgpu(0),
    ) {
        Ok(_) => panic!("a 4 GiB-physical bool upload must be refused, not attempted"),
        Err(error) => error,
    };
    let text = format!("{error:?}");
    assert!(
        text.contains("max_storage_buffer_binding_size"),
        "the bool refusal must name the same adapter limit, got: {text}"
    );
}

#[test]
fn small_uploads_still_succeed_after_a_refusal() {
    require_wgpu();
    let values: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let storage = <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        &[2, 2],
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("ordinary uploads must keep working after a refusal");
    assert_eq!(storage.shape.to_vec(), vec![2, 2]);
}
