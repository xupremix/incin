//! WGPU product-reduction backward: `prod_all` and `prod_dim` advertise
//! `training = true` but recorded no tape entry, so a model reaching them
//! trained nothing downstream while the capability row promised training.
//!
//! Each case checks the forward against the closed-form product, that the
//! operation left a tape entry behind, and that a full backward walk yields
//! the zero-aware product rule the CPU reference proves
//! (`cpu::ops::reduce::helpers::{prod_all_grad, prod_dim_grad}`): no zeros
//! means `g * prod / x[i]`; exactly one zero means `g * (product of the
//! rest)` at the zero position and zero elsewhere.
//!
//! The one-zero cases are deliberate: a naive `grad * prod / x` backward
//! divides by zero there and reports `inf`/`NaN` instead of the finite
//! gradient the rule requires, so they are the only inputs that distinguish
//! the two.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{
    AutogradBackend, Execute, HostInterop, HostReadback, StorageBackend, op,
};
use incin_core::exec::catalog::{AxisAttributes, NoAttributes};
use incin_core::exec::{CanonicalOperation, ExecutionContext, TapeStorage, TensorHandle};
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

/// Runs one catalog operation through canonical dispatch and returns its
/// result plus the number of tape entries it added.
fn run<O, A>(attributes: A, inputs: &[TestStorage]) -> (TestStorage, usize)
where
    O: CanonicalOperation<Attributes = A>,
    O::Attributes: incin_core::exec::catalog::AttributeContract,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::default());
    let handles: Vec<TensorHandle> = inputs
        .iter()
        .map(TensorHandle::from_storage::<TestBackend, f32, _>)
        .collect();
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &handles)
        .expect("an advertised operation must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

fn assert_close(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-4, "{what}[{i}] should be {e}, got {a}");
    }
}

#[test]
fn prod_all_records_its_zero_aware_backward_on_wgpu() {
    require_wgpu();
    // No zeros: prod = 24, grad = [12, 8, 6].
    let input = upload(&[2.0, 3.0, 4.0], &[3]);
    let input_id = TapeStorage::id(&input);
    let (out, recorded) = run::<op::ProdAll, _>(NoAttributes, &[input]);
    assert_close(&read(&out), &[24.0], "prod_all forward");
    assert!(
        recorded >= 1,
        "prod_all advertises `training = true`, so it must record a tape entry"
    );
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&out).expect("backward runs on WGPU");
    let grad = grads.get(input_id).expect("input has a gradient");
    assert_close(&read(grad), &[12.0, 8.0, 6.0], "prod_all backward");

    // Exactly one zero: prod = 0, grad = [0, 8, 0].
    let input = upload(&[2.0, 0.0, 4.0], &[3]);
    let input_id = TapeStorage::id(&input);
    let (out, _) = run::<op::ProdAll, _>(NoAttributes, &[input]);
    assert_close(&read(&out), &[0.0], "prod_all forward with a zero");
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&out).expect("backward runs on WGPU");
    let grad = grads.get(input_id).expect("input has a gradient");
    assert_close(
        &read(grad),
        &[0.0, 8.0, 0.0],
        "prod_all backward with a zero",
    );
}

#[test]
fn prod_dim_records_its_zero_aware_backward_on_wgpu() {
    require_wgpu();
    // [[1, 2, 3], [4, 0, 6]] along axis 1: [6, 0]. Row 0 has no zeros, so
    // its gradient is `prod / x`; row 1 has exactly one zero, so only the
    // zero position receives the product of the rest.
    let input = upload(&[1.0, 2.0, 3.0, 4.0, 0.0, 6.0], &[2, 3]);
    let input_id = TapeStorage::id(&input);
    let (out, recorded) = run::<op::ProdDim, _>(AxisAttributes { axis: 1 }, &[input]);
    assert_close(&read(&out), &[6.0, 0.0], "prod_dim forward");
    assert!(
        recorded >= 1,
        "prod_dim advertises `training = true`, so it must record a tape entry"
    );
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&out).expect("backward runs on WGPU");
    let grad = grads.get(input_id).expect("input has a gradient");
    assert_close(
        &read(grad),
        &[6.0, 3.0, 2.0, 0.0, 24.0, 0.0],
        "prod_dim backward",
    );
}
