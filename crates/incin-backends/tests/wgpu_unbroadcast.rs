//! WGPU unbroadcast scalar-gradient hole: the twin of CUDA's #121 pinning
//! tests (`l1_loss_trains_through_scalar_reduction_on_cuda` and the two
//! `unbroadcast_*` unit tests).
//!
//! Broadcast forward expands a tensor; backward must UNbroadcast
//! (reduce-sum over broadcast dims) to recover the input's shape. The hole:
//! when the gradient flowing back is scalar-shaped (0-d) or the input was
//! scalar, the reduction was skipped or mis-shaped and the wrong gradient
//! (or a shape refusal) resulted. On CUDA this surfaced as
//! `Shape mismatch during 'iteration_plan': expected [], got [3]` for an
//! l1-mean backward, fixed by materializing a broadcast of the reduced
//! seed after a compatibility check; WGPU's `unbroadcast` tail
//! (`wgpu/tape.rs`) carries the same semantics via `broadcast_storage`.
//!
//! These tests pin the WGPU half end to end through public dispatch:
//! a scalar input broadcast forward, then backward with a scalar cotangent
//! and with a non-scalar cotangent, asserting exact gradient values and
//! shapes — plus the issue's l1-mean repro through `op::L1Loss` itself.
//!
//! Requires a WGPU adapter for the runtime half:
//! `cargo test -p incin-backends --no-default-features --features incin-backends/wgpu --test wgpu_unbroadcast`.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{
    AutogradBackend, HostInterop, HostReadback, StorageBackend, op,
};
use incin_core::exec::catalog::{LossAttributes, LossReduction, ShapeAttributes};
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

fn broadcast_as(
    context: &ExecutionContext<TestBackend>,
    input: &TestStorage,
    shape: &[usize],
) -> TestStorage {
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(input)];
    incin_core::exec::dispatch::execute::<op::BroadcastAs, _>(
        context,
        ShapeAttributes {
            shape: shape.to_vec(),
        },
        &inputs,
    )
    .expect("broadcast executes on WGPU")
}

#[test]
fn broadcast_scalar_forward_backward_with_scalar_cotangent() {
    // Scalar input broadcast forward, then a full-width consumer (mean)
    // whose backward seeds the walk with a scalar: the broadcast entry's
    // `unbroadcast` must reduce the [3] cotangent back to the scalar input
    // shape. d(mean(broadcast(2.0))) / d(input) = 1, exactly.
    require_wgpu();
    let context = ExecutionContext::new(TestBackend::default());
    let input = upload(&[2.0], &[]);
    let input_id = TapeStorage::id(&input);

    let wide = broadcast_as(&context, &input, &[3]);
    assert_eq!(wide.shape.dims(), &[3]);
    assert_eq!(read(&wide), vec![2.0, 2.0, 2.0]);

    let loss = {
        let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(&wide)];
        incin_core::exec::dispatch::execute::<op::MeanAll, _>(
            &context,
            incin_core::exec::catalog::NoAttributes,
            &inputs,
        )
        .expect("mean_all executes on WGPU")
    };
    assert_eq!(read(&loss), vec![2.0]);

    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&loss).expect("backward runs on WGPU");
    let grad = grads.get(input_id).expect("scalar input has a gradient");
    assert_eq!(
        grad.shape.dims(),
        &[] as &[usize],
        "the gradient must recover the scalar input shape"
    );
    let values = read(grad);
    assert_eq!(values.len(), 1);
    assert!(
        (values[0] - 1.0).abs() < 1e-5,
        "d(mean(broadcast(2.0)))/d(input) must be 1, got {}",
        values[0]
    );
}

#[test]
fn broadcast_scalar_forward_backward_with_nonscalar_cotangent() {
    // Same broadcast forward, but the walk is seeded explicitly with a
    // non-scalar cotangent: `unbroadcast` must sum-reduce it onto the
    // scalar input (1 + 2 + 4 = 7), not hand the [3] seed on.
    require_wgpu();
    let context = ExecutionContext::new(TestBackend::default());
    let input = upload(&[2.0], &[]);
    let input_id = TapeStorage::id(&input);

    let wide = broadcast_as(&context, &input, &[3]);
    assert_eq!(wide.shape.dims(), &[3]);

    let seed = upload(&[1.0, 2.0, 4.0], &[3]);
    let grads = <TestBackend as AutogradBackend>::backward_with::<f32>(&wide, &seed)
        .expect("backward with an explicit cotangent runs on WGPU");
    let grad = grads.get(input_id).expect("scalar input has a gradient");
    assert_eq!(
        grad.shape.dims(),
        &[] as &[usize],
        "the gradient must recover the scalar input shape"
    );
    let values = read(grad);
    assert_eq!(values.len(), 1);
    assert!(
        (values[0] - 7.0).abs() < 1e-5,
        "the non-scalar cotangent must reduce-sum to 7, got {}",
        values[0]
    );
}

#[test]
fn l1_loss_dispatch_trains_through_scalar_reduction_on_wgpu() {
    // The issue's repro through `op::L1Loss` itself rather than the manual
    // sub/abs/mean composition `wgpu_scalar_reduction` drives: pred
    // [1, 0, -1] against targ [1, 1, 0] under Mean. Forward is 2/3; the
    // mean seeds the walk with a scalar the abs/sub recipes need at full
    // width, and the pred gradient is [0, -1/3, -1/3] (sign(0) is 0).
    require_wgpu();
    let context = ExecutionContext::new(TestBackend::default());
    let pred = upload(&[1.0, 0.0, -1.0], &[3]);
    let targ = upload(&[1.0, 1.0, 0.0], &[3]);
    let pred_id = TapeStorage::id(&pred);

    let loss = {
        let inputs = [
            TensorHandle::from_storage::<TestBackend, f32, _>(&pred),
            TensorHandle::from_storage::<TestBackend, f32, _>(&targ),
        ];
        incin_core::exec::dispatch::execute::<op::L1Loss, _>(
            &context,
            LossAttributes {
                reduction: LossReduction::Mean,
            },
            &inputs,
        )
        .expect("l1 executes on WGPU")
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
    assert_eq!(grad.shape.dims(), &[3]);
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
