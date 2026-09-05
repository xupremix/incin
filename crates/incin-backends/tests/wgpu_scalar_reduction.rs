//! WGPU scalar-reduction backward: the twin of
//! `l1_loss_trains_through_scalar_reduction_on_cuda` (#121).
//!
//! A mean reduction seeds the walk with a scalar gradient the next recipe
//! needs at full width. WGPU has no `L1Loss` executor, so this drives the
//! same shape through the supported composition (sub, abs, `mean_all`):
//! forward must be 2/3 and the pred gradient `[0, -1/3, -1/3]` (sign(0) is
//! 0). Before the `unbroadcast` tail, backward died in `iteration_plan`
//! with `expected [], got [3]`.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{
    AutogradBackend, HostInterop, HostReadback, StorageBackend, op,
};
use incin_core::exec::catalog::NoAttributes;
use incin_core::exec::{ExecutionContext, TapeStorage, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

/// Aborts unless a WGPU adapter is present (same contract as the other WGPU
/// suites: compiling with the feature is an explicit request for the
/// backend, so a missing adapter fails rather than skipping green).
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

fn upload(values: &[f32], shape: &[usize]) -> TestStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading the operand must succeed")
}

fn read(storage: &TestStorage) -> Vec<f64> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading a contiguous f32 buffer back must succeed")
}

#[test]
fn l1_mean_trains_through_scalar_reduction_on_wgpu() {
    require_wgpu();
    let context = ExecutionContext::new(TestBackend::default());
    let pred = upload(&[1.0, 0.0, -1.0], &[3]);
    let targ = upload(&[1.0, 1.0, 0.0], &[3]);
    let pred_id = TapeStorage::id(&pred);

    // diff = pred - targ = [0, -1, -1]
    let diff = {
        let inputs = [
            TensorHandle::from_storage::<TestBackend, f32, _>(&pred),
            TensorHandle::from_storage::<TestBackend, f32, _>(&targ),
        ];
        incin_core::exec::dispatch::execute::<op::Sub, _>(&context, NoAttributes, &inputs)
            .expect("sub executes on WGPU")
    };
    // abs(diff) = [0, 1, 1]
    let abs = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(&diff)];
        incin_core::exec::dispatch::execute::<op::Abs, _>(&context, NoAttributes, &inputs)
            .expect("abs executes on WGPU")
    };
    // mean(abs) = 2/3
    let loss = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(&abs)];
        incin_core::exec::dispatch::execute::<op::MeanAll, _>(&context, NoAttributes, &inputs)
            .expect("mean_all executes on WGPU")
    };
    let forward = read(&loss);
    assert_eq!(forward.len(), 1);
    assert!(
        (forward[0] - 2.0 / 3.0).abs() < 1e-6,
        "forward must be 2/3, got {}",
        forward[0]
    );

    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs on WGPU");
    let grad = grads.get(pred_id).expect("pred has a gradient");
    let values = read(grad);
    assert_eq!(values.len(), 3);
    assert!(
        values[0].abs() < 1e-5,
        "sign(0) must be 0, got {}",
        values[0]
    );
    for (i, value) in values.iter().enumerate().skip(1) {
        assert!(
            (value + 1.0 / 3.0).abs() < 1e-5,
            "grad[{i}] should be -1/3, got {value}"
        );
    }
}
