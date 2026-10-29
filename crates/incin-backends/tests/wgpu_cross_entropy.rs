//! `cross_entropy_loss` on real WGPU hardware, against the same hand-computed
//! NLL the CPU and CUDA twins use (issue #91 Tier-2: the last unadvertised
//! loss row, closed now that the integer class-target `gather` has a path).
//!
//! Fail-closed contract: the capability row claims `training = true`, so this
//! file checks both halves — forward values against the stable `log_softmax`
//! recipe, and a full backward walk that lands `dL/dlogits = (softmax -
//! onehot) / batch` on the logits. The gather's scatter-based backward is
//! what carries the gradient; a raw untaped gather would produce the right
//! forward and silently drop the gradient.
//!
//! Requires a WGPU adapter:
//! `cargo test -p incin-backends --features wgpu --test wgpu_cross_entropy`.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{
    AutogradBackend, HostInterop, HostReadback, StorageBackend, TapeStorage, op,
};
use incin_core::exec::catalog::{LossAttributes, LossReduction};
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;
type IntStorage = <TestBackend as StorageBackend>::Storage<i64>;

fn require_wgpu() {
    assert!(
        <TestBackend as HostInterop>::from_bytes::<f32>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .is_ok(),
        "no WGPU adapter, but the `wgpu` feature is enabled -- that is an explicit request for this backend. Skipping here would report `ok` for a test that ran nothing."
    );
}

fn upload_f32(values: &[f32], shape: &[usize]) -> TestStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading f32 values must succeed")
}

fn upload_i64(values: &[i64], shape: &[usize]) -> IntStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<i64>(
        &bytes,
        shape,
        DTypeId::I64.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading i64 targets must succeed")
}

fn read_f32(storage: &TestStorage) -> Vec<f64> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading an f32 buffer back must succeed")
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: got {a}, expected {e} (tol {tol})"
        );
    }
}

fn run_cross_entropy(
    logits: &TestStorage,
    target: &IntStorage,
    reduction: LossReduction,
) -> (TestStorage, usize) {
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(logits),
        TensorHandle::from_storage::<TestBackend, i64, _>(target),
    ];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::CrossEntropyLoss, _>(
        &context,
        LossAttributes { reduction },
        &handles,
    )
    .expect("cross_entropy_loss is advertised and must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

fn sum_loss(out: &TestStorage) -> TestStorage {
    let ctx = ExecutionContext::new(TestBackend::default());
    let handle = TensorHandle::from_storage::<TestBackend, f32, _>(out);
    incin_core::exec::dispatch::execute::<op::SumAll, _>(
        &ctx,
        incin_core::exec::catalog::NoAttributes,
        &[handle],
    )
    .expect("sum_all must run to seed the backward")
}

/// The CUDA/CPU fixture: logits `[2, 3]`, targets `[0, 2]`. Mean reduction
/// over two rows of three-class log-softmax NLL.
#[test]
fn cross_entropy_mean_matches_the_hand_computed_nll_and_trains() {
    require_wgpu();
    let logits = upload_f32(&[2.0, 1.0, 0.5, 0.5, 1.5, 0.0], &[2, 3]);
    let targets = upload_i64(&[0, 2], &[2]);
    let logits_id = TapeStorage::id(&logits);

    let (out, recorded) = run_cross_entropy(&logits, &targets, LossReduction::Mean);
    assert!(
        recorded >= 1,
        "cross_entropy_loss advertises training = true and must record"
    );

    // Host reference: -(log p(row0, class0) + log p(row1, class2)) / 2.
    let fwd = read_f32(&out);
    assert_close(&fwd, &[1.2144], 1e-3, "cross_entropy mean");

    // Walk the tape: dL/dlogits = (softmax - onehot) / batch, computed by
    // hand from the same logits the CUDA twin uses.
    let loss = sum_loss(&out);
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&loss)
        .expect("cross_entropy backward must run");
    let grad = grads
        .get(logits_id)
        .expect("logits must receive a gradient");
    let values = read_f32(grad);
    let expected = [-0.1857, 0.1156, 0.0701, 0.1156, 0.3142, -0.4299];
    assert_eq!(values.len(), expected.len(), "logits grad length");
    for (i, (got, want)) in values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "logits grad[{i}]: got {got}, want {want}"
        );
    }
}

#[test]
fn cross_entropy_sum_is_batch_times_mean_and_none_is_per_sample() {
    require_wgpu();
    let logits = upload_f32(&[2.0, 1.0, 0.5, 0.5, 1.5, 0.0], &[2, 3]);
    let targets = upload_i64(&[0, 2], &[2]);

    let (mean_out, _) = run_cross_entropy(&logits, &targets, LossReduction::Mean);
    let (sum_out, _) = run_cross_entropy(&logits, &targets, LossReduction::Sum);
    let (none_out, _) = run_cross_entropy(&logits, &targets, LossReduction::None);

    let mean = read_f32(&mean_out)[0];
    let sum = read_f32(&sum_out)[0];
    assert!(
        (sum - mean * 2.0).abs() < 1e-5,
        "sum should equal batch * mean: got {sum} vs {mean} * 2"
    );

    // `None` produces the per-sample NLL vector of length batch.
    let per_sample = read_f32(&none_out);
    assert_eq!(per_sample.len(), 2, "reduction=None yields [batch]");
    assert!(
        (per_sample[0] + per_sample[1] - sum).abs() < 1e-5,
        "per-sample NLL must sum to the Sum reduction"
    );
}

#[test]
fn cross_entropy_rejects_out_of_range_targets() {
    require_wgpu();
    let logits = upload_f32(&[2.0, 1.0, 0.5, 0.5, 1.5, 0.0], &[2, 3]);
    // Class index 3 is out of range for a 3-class row.
    let bad_targets = upload_i64(&[0, 3], &[2]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&logits),
        TensorHandle::from_storage::<TestBackend, i64, _>(&bad_targets),
    ];
    let result = incin_core::exec::dispatch::execute::<op::CrossEntropyLoss, _>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &handles,
    );
    assert!(
        result.is_err(),
        "out-of-range class target must fail closed, not wrap or clamp"
    );
}
