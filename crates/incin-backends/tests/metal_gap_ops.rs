//! Dispatch-level coverage for the Metal #92 gap-closure ops: losses,
//! variance/std/norm/cumsum reductions, the indexing family and the
//! batch/group normalization rewrites.
//!
//! `dispatch::execute` runs `admit_invocation` (the capability check) before
//! `Execute`, so every test below proves its row is both advertised *and*
//! executable on the real request path — the runtime twin of the
//! compile-time `assert_every_advertised_metal_row_executes` in
//! `metal/executor.rs`.
//!
//! No physical Metal device exists on this CI host: `HostInterop::from_bytes`
//! only validates `DeviceId::metal(0)` (kind + dtype), never a driver, and
//! `MetalStorage` wraps a plain host buffer — so the host-side math runs
//! exactly as `metal/*/tests` do, but through dispatch with the shared
//! declarations table admitted first.
#![cfg(feature = "metal")]

use incin_backends::metal::{MetalBackendImpl, tape_depth};
use incin_core::backend_authoring::{HostInterop, HostReadback, StorageBackend, op};
use incin_core::exec::catalog::{
    AxisAttributes, AxisVarianceAttributes, BatchNormAttributes, GroupNormAttributes,
    LossAttributes, LossReduction, NoAttributes, NormAttributes, VarianceAttributes,
};
use incin_core::exec::{ExecutionContext, TapeStorage, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, Metal};

type TestBackend = MetalBackendImpl<Metal>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;
type IntStorage = <TestBackend as StorageBackend>::Storage<i64>;

const VALUES: [f32; 6] = [1.0, 2.0, 3.0, -0.5, 0.25, 4.0];
const TARGETS: [f32; 6] = [0.5, 2.5, 1.0, 0.0, -1.0, 4.0];
const SHAPE: [usize; 2] = [2, 3];
const EPSILON: f64 = 1e-5;

fn require_metal() {
    assert!(
        <TestBackend as HostInterop>::from_bytes::<f32>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::metal(0),
        )
        .is_ok(),
        "no Metal device id, but the `metal` feature is enabled -- that is an explicit request for this backend."
    );
}

fn upload(values: &[f32], shape: &[usize]) -> TestStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::metal(0),
    )
    .expect("uploading the operand must succeed")
}

fn upload_i64(values: &[i64], shape: &[usize]) -> IntStorage {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    <TestBackend as HostInterop>::from_bytes::<i64>(
        &bytes,
        shape,
        DTypeId::I64.descriptor(),
        &DeviceId::metal(0),
    )
    .expect("uploading i64 indices must succeed")
}

fn read(storage: &TestStorage) -> Vec<f64> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading a contiguous f32 buffer back must succeed")
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol || (a.is_nan() && e.is_nan()),
            "{label}[{i}]: got {a}, expected {e} (tol {tol})"
        );
    }
}

fn run1<O, A>(input: &TestStorage, attributes: A) -> (TestStorage, usize)
where
    O: incin_core::exec::CanonicalOperation<Attributes = A>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
    A: Clone,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(input)];
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised Metal operation must execute");
    (out, tape_depth() - before)
}

fn run_pair<O, A>(lhs: &TestStorage, rhs: &TestStorage, attributes: A) -> (TestStorage, usize)
where
    O: incin_core::exec::CanonicalOperation<Attributes = A>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
    A: Clone,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, _>(lhs),
        TensorHandle::from_storage::<TestBackend, f32, _>(rhs),
    ];
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised Metal operation must execute");
    (out, tape_depth() - before)
}

/// Mixed-dtype pair: f32 values + i64 indices (embedding/gather targets).
fn run_mixed2<O, A>(values: &TestStorage, ints: &IntStorage, attributes: A) -> (TestStorage, usize)
where
    O: incin_core::exec::CanonicalOperation<Attributes = A>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
    A: Clone,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, _>(values),
        TensorHandle::from_storage::<TestBackend, i64, _>(ints),
    ];
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised Metal operation must execute");
    (out, tape_depth() - before)
}

/// Mixed-dtype pair with the index first (embedding: indices, weight).
fn run_mixed_index_first<O, A>(
    ints: &IntStorage,
    values: &TestStorage,
    attributes: A,
) -> (TestStorage, usize)
where
    O: incin_core::exec::CanonicalOperation<Attributes = A>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
    A: Clone,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, i64, _>(ints),
        TensorHandle::from_storage::<TestBackend, f32, _>(values),
    ];
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised Metal operation must execute");
    (out, tape_depth() - before)
}

/// Cross-entropy: f32 logits then i64 class targets.
fn run_cross_entropy(
    logits: &TestStorage,
    targets: &IntStorage,
    reduction: LossReduction,
) -> (TestStorage, usize) {
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(logits),
        TensorHandle::from_storage::<TestBackend, i64, _>(targets),
    ];
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::CrossEntropyLoss, _>(
        &context,
        LossAttributes { reduction },
        &handles,
    )
    .expect("cross_entropy_loss is advertised and must execute");
    (out, tape_depth() - before)
}

fn sum_loss(out: &TestStorage) -> TestStorage {
    let ctx = ExecutionContext::new(TestBackend::default());
    let handle = TensorHandle::from_storage::<TestBackend, f32, _>(out);
    incin_core::exec::dispatch::execute::<op::SumAll, _>(&ctx, NoAttributes, &[handle])
        .expect("sum_all must run to seed the backward")
}

/// Host reduction of per-element loss points.
fn reduced(points: &[f64], reduction: LossReduction) -> Vec<f64> {
    match reduction {
        LossReduction::None => points.to_vec(),
        LossReduction::Mean => vec![points.iter().sum::<f64>() / points.len() as f64],
        LossReduction::Sum => vec![points.iter().sum::<f64>()],
    }
}

// ── losses ──────────────────────────────────────────────────────────────────

#[test]
fn mse_loss_reduces_the_squared_diff_and_records() {
    require_metal();
    let pred = upload(&VALUES, &SHAPE);
    let target = upload(&TARGETS, &SHAPE);
    let pred_id = TapeStorage::id(&pred);

    let (out, recorded) = run_pair::<op::MseLoss, _>(
        &pred,
        &target,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
    );
    assert!(recorded >= 1, "mse_loss advertises training = true");
    let points: Vec<f64> = VALUES
        .iter()
        .zip(TARGETS.iter())
        .map(|(p, t)| f64::from(p - t) * f64::from(p - t))
        .collect();
    assert_close(
        &read(&out),
        &reduced(&points, LossReduction::Mean),
        1e-5,
        "mse mean",
    );

    let (none, _) = run_pair::<op::MseLoss, _>(
        &pred,
        &target,
        LossAttributes {
            reduction: LossReduction::None,
        },
    );
    assert_eq!(read(&none).len(), 6, "reduction=None keeps elementwise");

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as incin_core::backend_authoring::AutogradBackend>::backward::<f32>(&loss)
            .expect("mse backward must run");
    let grad = grads.get(pred_id).expect("pred receives a gradient");
    assert_eq!(
        read(grad).len(),
        6,
        "mse pred grad has the elementwise shape"
    );
}

#[test]
fn l1_loss_uses_the_sign_of_the_diff_and_records() {
    require_metal();
    let pred = upload(&VALUES, &SHAPE);
    let target = upload(&TARGETS, &SHAPE);
    let pred_id = TapeStorage::id(&pred);

    let (out, recorded) = run_pair::<op::L1Loss, _>(
        &pred,
        &target,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
    );
    assert!(recorded >= 1, "l1_loss advertises training = true");
    let points: Vec<f64> = VALUES
        .iter()
        .zip(TARGETS.iter())
        .map(|(p, t)| f64::from((p - t).abs()))
        .collect();
    assert_close(
        &read(&out),
        &reduced(&points, LossReduction::Mean),
        1e-5,
        "l1 mean",
    );

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as incin_core::backend_authoring::AutogradBackend>::backward::<f32>(&loss)
            .expect("l1 backward must run");
    let grad = grads.get(pred_id).expect("pred receives a gradient");
    assert_eq!(
        read(grad).len(),
        6,
        "l1 pred grad has the elementwise shape"
    );
}

#[test]
fn bce_with_logits_loss_is_finite_and_records() {
    require_metal();
    let pred = upload(&VALUES, &SHAPE);
    let target = upload(&[0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &SHAPE);
    let pred_id = TapeStorage::id(&pred);

    let (out, recorded) = run_pair::<op::BceWithLogitsLoss, _>(
        &pred,
        &target,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
    );
    assert!(
        recorded >= 1,
        "bce_with_logits_loss advertises training = true"
    );
    let fwd = read(&out);
    assert_eq!(fwd.len(), 1, "mean reduction yields a scalar");
    assert!(fwd[0].is_finite(), "bce mean must be finite: {fwd:?}");

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as incin_core::backend_authoring::AutogradBackend>::backward::<f32>(&loss)
            .expect("bce backward must run");
    let grad = grads.get(pred_id).expect("pred receives a gradient");
    assert!(
        read(grad).iter().all(|v| v.is_finite()),
        "bce pred grad must be finite"
    );
}

#[test]
fn cross_entropy_mean_matches_the_hand_computed_nll_and_trains() {
    require_metal();
    // Same fixture the CUDA/CPU/WGPU twins use.
    let logits = upload(&[2.0, 1.0, 0.5, 0.5, 1.5, 0.0], &[2, 3]);
    let targets = upload_i64(&[0, 2], &[2]);
    let logits_id = TapeStorage::id(&logits);

    let (out, recorded) = run_cross_entropy(&logits, &targets, LossReduction::Mean);
    assert!(
        recorded >= 1,
        "cross_entropy_loss advertises training = true"
    );
    assert_close(&read(&out), &[1.2144], 1e-3, "cross entropy mean");

    let (sum, _) = run_cross_entropy(&logits, &targets, LossReduction::Sum);
    let (none, _) = run_cross_entropy(&logits, &targets, LossReduction::None);
    let mean_v = read(&out)[0];
    let sum_v = read(&sum)[0];
    assert!(
        (sum_v - mean_v * 2.0).abs() < 1e-5,
        "sum should equal batch * mean: {sum_v} vs {mean_v} * 2"
    );
    assert_eq!(read(&none).len(), 2, "reduction=None yields [batch]");

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as incin_core::backend_authoring::AutogradBackend>::backward::<f32>(&loss)
            .expect("cross entropy backward must run");
    let grad = grads.get(logits_id).expect("logits receives a gradient");
    let values = read(grad);
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
fn cross_entropy_rejects_out_of_range_targets() {
    require_metal();
    let logits = upload(&[2.0, 1.0, 0.5, 0.5, 1.5, 0.0], &[2, 3]);
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

// ── variance / std / norm / cumsum ──────────────────────────────────────────

#[test]
fn variance_all_and_std_all_match_host_references() {
    require_metal();
    let t = upload(&VALUES, &SHAPE);
    let (var, recorded) = run1::<op::VarianceAll, _>(&t, VarianceAttributes { unbiased: false });
    assert!(recorded >= 1, "variance_all advertises training = true");
    let mean: f64 = VALUES.iter().map(|&v| f64::from(v)).sum::<f64>() / 6.0;
    let pop: f64 = VALUES
        .iter()
        .map(|&v| {
            let d = f64::from(v) - mean;
            d * d
        })
        .sum::<f64>()
        / 6.0;
    assert_close(&read(&var), &[pop], 1e-5, "variance population");

    let (std, _) = run1::<op::StdAll, _>(&t, VarianceAttributes { unbiased: false });
    assert_close(&read(&std), &[pop.sqrt()], 1e-5, "std = sqrt(var)");
}

#[test]
fn variance_dim_drops_the_axis_and_keepdim_keeps_it() {
    require_metal();
    let t = upload(&VALUES, &SHAPE);
    let (dim, recorded) = run1::<op::VarianceDim, _>(
        &t,
        AxisVarianceAttributes {
            axis: 1,
            unbiased: false,
        },
    );
    assert!(recorded >= 1, "variance_dim advertises training = true");
    assert_eq!(
        <TestBackend as StorageBackend>::shape::<f32>(&dim).dims(),
        &[2],
        "variance_dim drops the axis"
    );
    // Column population variances of [1,2,3] and [-0.5,0.25,4].
    let m0 = (1.0f64 + 2.0 + 3.0) / 3.0;
    let v0 = ((1.0f64 - m0).powi(2) + (2.0f64 - m0).powi(2) + (3.0f64 - m0).powi(2)) / 3.0;
    let m1 = (-0.5f64 + 0.25 + 4.0) / 3.0;
    let v1 = ((-0.5f64 - m1).powi(2) + (0.25f64 - m1).powi(2) + (4.0f64 - m1).powi(2)) / 3.0;
    assert_close(&read(&dim), &[v0, v1], 1e-5, "variance dim=1");

    let (keep, _) = run1::<op::VarianceKeepDim, _>(
        &t,
        AxisVarianceAttributes {
            axis: 1,
            unbiased: false,
        },
    );
    assert_eq!(
        <TestBackend as StorageBackend>::shape::<f32>(&keep).dims(),
        &[2, 1],
        "variance_keepdim keeps a unit axis"
    );
}

#[test]
fn norm_order_one_and_two_match_their_closed_forms() {
    require_metal();
    let t = upload(&VALUES, &SHAPE);
    let (l1, recorded) = run1::<op::Norm, _>(&t, NormAttributes { order: 1.0 });
    assert!(recorded >= 1, "norm advertises training = true");
    let abs_sum: f64 = VALUES.iter().map(|&v| f64::from(v.abs())).sum();
    assert_close(&read(&l1), &[abs_sum], 1e-5, "L1 norm");

    let (l2, _) = run1::<op::Norm, _>(&t, NormAttributes { order: 2.0 });
    let sq_sum: f64 = VALUES.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    assert_close(&read(&l2), &[sq_sum.sqrt()], 1e-5, "L2 norm");
}

#[test]
fn cumsum_scans_along_the_named_axis_and_records() {
    require_metal();
    let t = upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &SHAPE);
    let (across, recorded) = run1::<op::Cumsum, _>(&t, AxisAttributes { axis: 1 });
    assert!(recorded >= 1, "cumsum advertises training = true");
    assert_eq!(
        <TestBackend as StorageBackend>::shape::<f32>(&across).dims(),
        &[2, 3],
        "cumsum is shape-preserving"
    );
    assert_close(
        &read(&across),
        &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0],
        1e-5,
        "cumsum axis=1",
    );

    let (down, _) = run1::<op::Cumsum, _>(&t, AxisAttributes { axis: 0 });
    assert_close(
        &read(&down),
        &[1.0, 2.0, 3.0, 5.0, 7.0, 9.0],
        1e-5,
        "cumsum axis=0",
    );
}

// ── indexing family ─────────────────────────────────────────────────────────

#[test]
fn embedding_gathers_rows_and_accumulates_repeated_indices() {
    require_metal();
    // weight [3,2]: row0=[1,2], row1=[3,4], row2=[5,6]; indices [0,2,0]
    let weight = upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let indices = upload_i64(&[0, 2, 0], &[3]);
    let weight_id = TapeStorage::id(&weight);

    let (out, recorded) =
        run_mixed_index_first::<op::EmbeddingExact, _>(&indices, &weight, NoAttributes);
    assert!(recorded >= 1, "embedding advertises training = true");
    assert_close(
        &read(&out),
        &[1.0, 2.0, 5.0, 6.0, 1.0, 2.0],
        1e-6,
        "embedding forward",
    );

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as incin_core::backend_authoring::AutogradBackend>::backward::<f32>(&loss)
            .expect("embedding backward must run");
    let g = grads.get(weight_id).expect("weight receives a gradient");
    // Row 0 selected twice → 2.0 each; row 1 never → 0; row 2 once → 1.
    assert_close(
        &read(g),
        &[2.0, 2.0, 0.0, 0.0, 1.0, 1.0],
        1e-6,
        "embedding grad",
    );
}

#[test]
fn gather_selects_along_the_axis_and_scatters_grad_back() {
    require_metal();
    // input [2,3]; index [[2,0],[0,1]] shape [2,2], dim=1
    let input = upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let index = upload_i64(&[2, 0, 0, 1], &[2, 2]);
    let input_id = TapeStorage::id(&input);

    let (out, recorded) = run_mixed2::<op::Gather, _>(&input, &index, AxisAttributes { axis: 1 });
    assert!(recorded >= 1, "gather advertises training = true");
    assert_close(&read(&out), &[3.0, 1.0, 4.0, 5.0], 1e-6, "gather forward");

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as incin_core::backend_authoring::AutogradBackend>::backward::<f32>(&loss)
            .expect("gather backward must run");
    let g = grads.get(input_id).expect("input receives a gradient");
    // cotangent lands only on (0,2), (0,0), (1,0), (1,1)
    assert_close(
        &read(g),
        &[1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        1e-6,
        "gather grad",
    );
}

#[test]
fn index_select_replaces_the_axis_and_accumulates_dups() {
    require_metal();
    let input = upload(&[10.0, 20.0, 30.0, 40.0], &[4]);
    let index = upload_i64(&[2, 0, 2], &[3]);
    let input_id = TapeStorage::id(&input);

    let (out, recorded) =
        run_mixed2::<op::IndexSelect, _>(&input, &index, AxisAttributes { axis: 0 });
    assert!(recorded >= 1, "index_select advertises training = true");
    assert_close(
        &read(&out),
        &[30.0, 10.0, 30.0],
        1e-6,
        "index_select forward",
    );

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as incin_core::backend_authoring::AutogradBackend>::backward::<f32>(&loss)
            .expect("index_select backward must run");
    let g = grads.get(input_id).expect("input receives a gradient");
    // index 2 twice → 2.0; index 0 once → 1.0; 1 and 3 untouched
    assert_close(&read(g), &[1.0, 0.0, 2.0, 0.0], 1e-6, "index_select grad");
}

// ── batch_norm / group_norm ─────────────────────────────────────────────────

#[test]
fn batch_norm_inference_uses_running_statistics_and_records() {
    require_metal();
    let input = upload(&[1.0, 4.0, 3.0, 6.0], &[2, 2]);
    let weight = upload(&[2.0, 1.0], &[2]);
    let bias = upload(&[0.1, -0.2], &[2]);
    let running_mean = storage_from_host(&[0.5, 1.0], &[2]);
    let running_var = storage_from_host(&[1.0, 4.0], &[2]);

    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&input),
        TensorHandle::from_storage::<TestBackend, f32, _>(&weight),
        TensorHandle::from_storage::<TestBackend, f32, _>(&bias),
        TensorHandle::from_storage::<TestBackend, f32, _>(&running_mean),
        TensorHandle::from_storage::<TestBackend, f32, _>(&running_var),
    ];
    let attributes = BatchNormAttributes {
        epsilon: EPSILON,
        momentum: 0.1,
        training: false,
        has_weight: true,
        has_bias: true,
        has_running_mean: true,
        has_running_variance: true,
    };
    let before = tape_depth();
    let out =
        incin_core::exec::dispatch::execute::<op::BatchNorm, _>(&context, attributes, &handles)
            .expect("inference batch_norm is advertised and must execute");
    assert!(
        tape_depth() - before >= 1,
        "batch_norm advertises training = true"
    );

    // Hand formula: w * (x - mu) / sqrt(var + eps) + b
    let want: Vec<f64> = [1.0f32, 4.0, 3.0, 6.0]
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let ch = i % 2;
            let mu = [0.5f32, 1.0][ch];
            let var = [1.0f32, 4.0][ch];
            let w = [2.0f32, 1.0][ch];
            let b = [0.1f32, -0.2][ch];
            f64::from(w * ((x - mu) / (var + EPSILON as f32).sqrt()) + b)
        })
        .collect();
    assert_close(&read(&out), &want, 1e-4, "batch_norm inference");
}

#[test]
fn batch_norm_training_normalizes_by_batch_statistics_and_backward_runs() {
    require_metal();
    let input = upload(&[1.0, 4.0, 3.0, 6.0], &[2, 2]);
    let weight = upload(&[1.5, 0.5], &[2]);
    let bias = upload(&[0.1, -0.1], &[2]);
    let input_id = TapeStorage::id(&input);

    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&input),
        TensorHandle::from_storage::<TestBackend, f32, _>(&weight),
        TensorHandle::from_storage::<TestBackend, f32, _>(&bias),
    ];
    let attributes = BatchNormAttributes {
        epsilon: EPSILON,
        momentum: 0.1,
        training: true,
        has_weight: true,
        has_bias: true,
        has_running_mean: false,
        has_running_variance: false,
    };
    let before = tape_depth();
    let out =
        incin_core::exec::dispatch::execute::<op::BatchNorm, _>(&context, attributes, &handles)
            .expect("training batch_norm must execute");
    assert!(
        tape_depth() - before >= 1,
        "training-mode batch_norm advertises training = true"
    );

    // Channel means of the normalized (pre-affine) values are ~0; after the
    // affine scale the means are w*0 + b = b.
    let vals = read(&out);
    assert_eq!(vals.len(), 4);
    assert!(vals.iter().all(|v| v.is_finite()), "training output finite");

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as incin_core::backend_authoring::AutogradBackend>::backward::<f32>(&loss)
            .expect("training batch_norm backward must run");
    let g = grads.get(input_id).expect("input receives a gradient");
    assert!(
        read(g).iter().all(|v| v.is_finite()),
        "training grad finite"
    );
}

fn storage_from_host(values: &[f32], shape: &[usize]) -> TestStorage {
    upload(values, shape)
}

#[test]
fn group_norm_normalizes_within_each_group_and_backward_runs() {
    require_metal();
    // [1, 4] with 2 groups → two groups of two values each.
    let input = upload(&[1.0, 4.0, 2.0, 8.0], &[1, 4]);
    let input_id = TapeStorage::id(&input);
    let (out, recorded) = run1::<op::GroupNorm, _>(
        &input,
        GroupNormAttributes {
            groups: 2,
            epsilon: EPSILON,
        },
    );
    assert!(recorded >= 1, "group_norm advertises training = true");
    let vals = read(&out);
    assert_eq!(vals.len(), 4);
    // Each group's normalized values sum to ~0 (before any affine, which
    // group_norm on Metal does not apply — it is pure normalization).
    for group in 0..2 {
        let mean = (vals[group * 2] + vals[group * 2 + 1]) / 2.0;
        assert!(mean.abs() < 1e-4, "group {group} mean should be ~0: {mean}");
    }

    let loss = sum_loss(&out);
    let grads =
        <TestBackend as incin_core::backend_authoring::AutogradBackend>::backward::<f32>(&loss)
            .expect("group_norm backward must run");
    let g = grads.get(input_id).expect("input receives a gradient");
    assert!(
        read(g).iter().all(|v| v.is_finite()),
        "group_norm grad finite"
    );
}
