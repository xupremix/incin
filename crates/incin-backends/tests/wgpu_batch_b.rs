//! Batch-B WGPU gap closure on real hardware: structural windows and joins,
//! normalization rewrites, composed losses/moments, and the matmul family.
//!
//! Every operation advertised in the Batch-B capability update must prove
//! two things this file checks against host references computed with the
//! same formulas CPU uses:
//!
//! - forward values (structural ops exactly; statistical ops within a
//!   documented f32 tolerance);
//! - at least one tape entry for every row that claims `training = true`,
//!   because a composed forward that returns the right numbers but records
//!   nothing would silently drop the gradient.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{HostInterop, HostReadback, StorageBackend, op};
use incin_core::exec::catalog::{
    AxisAttributes, AxisVarianceAttributes, BatchNormAttributes, DiagonalAttributes,
    LayerNormAttributes, LinearAttributes, LossAttributes, LossReduction, NarrowAttributes,
    NoAttributes, NormAttributes, SliceAttributes, VarianceAttributes,
};
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

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
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised Batch-B operation must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

fn run2<O, A>(lhs: &TestStorage, rhs: &TestStorage, attributes: A) -> (TestStorage, usize)
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
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised two-input Batch-B operation must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

fn run_many<O, A>(inputs_storage: &[&TestStorage], attributes: A) -> (TestStorage, usize)
where
    O: incin_core::exec::CanonicalOperation<Attributes = A>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
    A: Clone,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs: Vec<_> = inputs_storage
        .iter()
        .map(|s| TensorHandle::from_storage::<TestBackend, f32, _>(*s))
        .collect();
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised multi-input Batch-B operation must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

// --- structural: narrow / slice / concat / stack / tril / triu ------------

/// A 3x4 row-major field so narrow/slice windows are unmistakable.
const FIELD: [f32; 12] = [
    0.0, 1.0, 2.0, 3.0, //
    10.0, 11.0, 12.0, 13.0, //
    20.0, 21.0, 22.0, 23.0,
];

#[test]
fn narrow_takes_the_requested_window_and_records() {
    require_wgpu();
    let input = upload(&FIELD, &[3, 4]);
    let (out, recorded) = run1::<op::Narrow, _>(
        &input,
        NarrowAttributes {
            axis: 1,
            start: 1,
            length: 2,
        },
    );
    assert_close(
        &read(&out),
        &[1.0, 2.0, 11.0, 12.0, 21.0, 22.0],
        0.0,
        "narrow",
    );
    assert!(recorded >= 1, "narrow advertises training = true");
}

#[test]
fn slice_exact_takes_a_per_axis_window() {
    require_wgpu();
    let input = upload(&FIELD, &[3, 4]);
    let (out, recorded) = run1::<op::SliceExact, _>(
        &input,
        SliceAttributes {
            ranges: vec![(1, 3), (0, 2)],
        },
    );
    assert_close(&read(&out), &[10.0, 11.0, 20.0, 21.0], 0.0, "slice");
    assert!(recorded >= 1, "slice advertises training = true");
}

#[test]
fn concat_joins_along_axis_zero() {
    require_wgpu();
    let a = upload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = upload(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let (out, recorded) = run_many::<op::ConcatExact, _>(&[&a, &b], AxisAttributes { axis: 0 });
    assert_close(
        &read(&out),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        0.0,
        "concat axis 0",
    );
    assert!(recorded >= 1, "concat advertises training = true");
}

#[test]
fn concat_joins_along_axis_one_without_reordering_rows() {
    require_wgpu();
    let a = upload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = upload(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let (out, _) = run_many::<op::ConcatExact, _>(&[&a, &b], AxisAttributes { axis: 1 });
    assert_close(
        &read(&out),
        &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0],
        0.0,
        "concat axis 1",
    );
}

#[test]
fn stack_inserts_a_new_leading_axis() {
    require_wgpu();
    let a = upload(&[1.0, 2.0], &[2]);
    let b = upload(&[3.0, 4.0], &[2]);
    let (out, recorded) = run_many::<op::StackExact, _>(&[&a, &b], AxisAttributes { axis: 0 });
    assert_close(&read(&out), &[1.0, 2.0, 3.0, 4.0], 0.0, "stack");
    assert_eq!(
        out.shape.to_vec(),
        vec![2, 2],
        "stack axis 0 of two rank-1 operands is [2, 2]"
    );
    assert!(recorded >= 1, "stack advertises training = true");
}

#[test]
fn tril_and_triu_mask_the_correct_half() {
    require_wgpu();
    let m = upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let (lower, recorded_lo) = run1::<op::Tril, _>(&m, DiagonalAttributes { offset: 0 });
    assert_close(
        &read(&lower),
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0],
        0.0,
        "tril",
    );
    assert!(recorded_lo >= 1, "tril advertises training = true");

    let (upper, recorded_up) = run1::<op::Triu, _>(&m, DiagonalAttributes { offset: 0 });
    assert_close(
        &read(&upper),
        &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0],
        0.0,
        "triu",
    );
    assert!(recorded_up >= 1, "triu advertises training = true");
}

// --- normalization rewrites: layer_norm / group_norm / batch_norm ---------

/// Two rows of four, deliberately non-zero-mean so the centering step is
/// exercised rather than a no-op.
const NORM_IN: [f32; 8] = [1.0, 2.0, 3.0, 6.0, -4.0, 0.0, 2.0, 4.0];

fn layer_norm_reference(values: &[f32], rows: usize, cols: usize, eps: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; values.len()];
    for r in 0..rows {
        let row = &values[r * cols..(r + 1) * cols];
        let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / cols as f64;
        let var = row
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum::<f64>()
            / cols as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for (c, &x) in row.iter().enumerate() {
            out[r * cols + c] = (f64::from(x) - mean) * inv;
        }
    }
    out
}

#[test]
fn layer_norm_matches_the_host_reference_and_records() {
    require_wgpu();
    let input = upload(&NORM_IN, &[2, 4]);
    let weight = upload(&[1.0f32; 4], &[4]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&input),
        TensorHandle::from_storage::<TestBackend, f32, _>(&weight),
    ];
    let attributes = LayerNormAttributes {
        normalized_shape: vec![4],
        epsilon: 1e-5,
        has_bias: false,
    };
    let before = incin_backends::wgpu::tape_depth();
    let out =
        incin_core::exec::dispatch::execute::<op::LayerNorm, _>(&context, attributes, &handles)
            .expect("layer_norm is advertised and must execute");
    let recorded = incin_backends::wgpu::tape_depth() - before;
    assert_close(
        &read(&out),
        &layer_norm_reference(&NORM_IN, 2, 4, 1e-5),
        1e-5,
        "layer_norm",
    );
    assert!(recorded >= 1, "layer_norm advertises training = true");
}

#[test]
fn group_norm_matches_a_host_rewrite_over_groups() {
    require_wgpu();
    // One sample, C=4 channels, spatial 2x2 -> group into two groups of 2
    // channels. Each group's 8 values form one normalized row. GroupNorm is
    // a single-input op (affine weight is not a catalog operand).
    let input = upload(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[1, 4, 2, 2],
    );
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [TensorHandle::from_storage::<TestBackend, f32, _>(&input)];
    let attributes = incin_core::exec::catalog::GroupNormAttributes {
        groups: 2,
        epsilon: 1e-5,
    };
    let before = incin_backends::wgpu::tape_depth();
    let out =
        incin_core::exec::dispatch::execute::<op::GroupNorm, _>(&context, attributes, &handles)
            .expect("group_norm is advertised and must execute");
    let recorded = incin_backends::wgpu::tape_depth() - before;
    let actual = read(&out);
    // Host reference: for each group of 2 channels, the 2*2*2 = 8 values
    // form one normalized row.
    let values: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let mut expected = vec![0.0f64; 16];
    let eps = 1e-5;
    for g in 0..2usize {
        let mut row = Vec::new();
        for c in g * 2..(g + 1) * 2 {
            for s in 0..4 {
                row.push(values[c * 4 + s]);
            }
        }
        let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / row.len() as f64;
        let var = row
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum::<f64>()
            / row.len() as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, &x) in row.iter().enumerate() {
            let c = g * 2 + i / 4;
            let s = i % 4;
            expected[c * 4 + s] = (f64::from(x) - mean) * inv;
        }
    }
    assert_close(&actual, &expected, 1e-4, "group_norm");
    assert!(recorded >= 1, "group_norm advertises training = true");
}

#[test]
fn batch_norm_inference_uses_running_statistics_and_refuses_training() {
    require_wgpu();
    let input = upload(&NORM_IN, &[2, 4]);
    let running_mean = upload(&[0.0f32; 4], &[4]);
    let running_var = upload(&[1.0f32; 4], &[4]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&input),
        TensorHandle::from_storage::<TestBackend, f32, _>(&running_mean),
        TensorHandle::from_storage::<TestBackend, f32, _>(&running_var),
    ];
    // Inference mode: identity affine, running mean 0 / var 1 => identity.
    let inference = BatchNormAttributes {
        epsilon: 1e-5,
        momentum: 0.1,
        training: false,
        has_weight: false,
        has_bias: false,
        has_running_mean: true,
        has_running_variance: true,
    };
    let out =
        incin_core::exec::dispatch::execute::<op::BatchNorm, _>(&context, inference, &handles)
            .expect("inference-mode batch_norm is advertised and must execute");
    // weight/bias absent => Candle defaults of ones/zeros => x itself when
    // mean=0 and var=1: (x - 0) / sqrt(1 + 1e-5) * 1 + 0.
    let expected: Vec<f64> = NORM_IN
        .iter()
        .map(|&x| f64::from(x) / f64::from((1.0f32 + 1e-5).sqrt()))
        .collect();
    assert_close(&read(&out), &expected, 1e-4, "batch_norm inference");

    // Training mode must be refused by name — there is no batch-statistics
    // kernel on this backend, and silently answering with the inference
    // formula is the failure the executor was written to prevent.
    let training = BatchNormAttributes {
        epsilon: 1e-5,
        momentum: 0.1,
        training: true,
        has_weight: false,
        has_bias: false,
        has_running_mean: true,
        has_running_variance: true,
    };
    let err =
        match incin_core::exec::dispatch::execute::<op::BatchNorm, _>(&context, training, &handles)
        {
            Ok(_) => panic!("training-mode batch_norm must be refused on WGPU"),
            Err(err) => err,
        };
    let message = format!("{err}");
    assert!(
        message.to_lowercase().contains("train") || message.to_lowercase().contains("batch"),
        "the refusal must name the training/batch-statistics gap: {message}"
    );
}

// --- composed reductions: var / std / norm / mse / l1 ---------------------

#[test]
fn variance_and_std_all_match_the_host_reference() {
    require_wgpu();
    let input = upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
    let (var, recorded_var) =
        run1::<op::VarianceAll, _>(&input, VarianceAttributes { unbiased: false });
    // population variance of 1..=6 = 17.5/6 = 35/12
    assert_close(&read(&var), &[35.0 / 12.0], 1e-5, "var_all biased");
    assert!(recorded_var >= 1, "var_all advertises training = true");

    let (std, _) = run1::<op::StdAll, _>(&input, VarianceAttributes { unbiased: false });
    assert_close(
        &read(&std),
        &[(35.0f64 / 12.0).sqrt()],
        1e-5,
        "std_all biased",
    );

    let (var_u, _) = run1::<op::VarianceAll, _>(&input, VarianceAttributes { unbiased: true });
    // sample variance of 1..=6 = 17.5/5 = 3.5
    assert_close(&read(&var_u), &[3.5], 1e-5, "var_all unbiased");
}

#[test]
fn variance_and_std_keepdim_preserve_the_reduced_axis() {
    require_wgpu();
    let input = upload(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let (var, _) = run1::<op::VarianceKeepDim, _>(
        &input,
        AxisVarianceAttributes {
            axis: 1,
            unbiased: false,
        },
    );
    // row 0: 1,2,3 mean 2 var 2/3; row 1: 4,5,6 mean 5 var 2/3
    assert_close(&read(&var), &[2.0 / 3.0, 2.0 / 3.0], 1e-5, "var_keepdim");
    assert_eq!(
        var.shape.to_vec(),
        vec![2, 1],
        "keepdim must leave the reduced axis in place"
    );

    let (std, _) = run1::<op::StdKeepDim, _>(
        &input,
        AxisVarianceAttributes {
            axis: 1,
            unbiased: false,
        },
    );
    assert_close(
        &read(&std),
        &[(2.0f64 / 3.0).sqrt(), (2.0f64 / 3.0).sqrt()],
        1e-5,
        "std_keepdim",
    );
}

#[test]
fn norm_matches_the_host_l2_reference() {
    require_wgpu();
    let input = upload(&[3.0, 4.0], &[2]);
    let (out, recorded) = run1::<op::Norm, _>(&input, NormAttributes { order: 2.0 });
    assert_close(&read(&out), &[5.0], 1e-5, "norm order 2");
    assert!(recorded >= 1, "norm advertises training = true");
}

#[test]
fn mse_and_l1_losses_match_the_host_reference() {
    require_wgpu();
    let pred = upload(&[1.0, 2.0, 3.0], &[3]);
    let target = upload(&[1.0, 0.0, 0.0], &[3]);
    let mean = LossAttributes {
        reduction: LossReduction::Mean,
    };
    let (mse, recorded_mse) = run2::<op::MseLoss, _>(&pred, &target, mean.clone());
    // errors: 0, 4, 9 -> mean 13/3
    assert_close(&read(&mse), &[13.0 / 3.0], 1e-5, "mse_loss mean");
    assert!(recorded_mse >= 1, "mse_loss advertises training = true");

    let (l1, recorded_l1) = run2::<op::L1Loss, _>(&pred, &target, mean);
    // |0| + |2| + |3| = 5, mean 5/3
    assert_close(&read(&l1), &[5.0 / 3.0], 1e-5, "l1_loss mean");
    assert!(recorded_l1 >= 1, "l1_loss advertises training = true");

    let sum = LossAttributes {
        reduction: LossReduction::Sum,
    };
    let (mse_sum, _) = run2::<op::MseLoss, _>(&pred, &target, sum);
    assert_close(&read(&mse_sum), &[13.0], 1e-5, "mse_loss sum");
}

// --- matmul family: bmm / addmm / dot / linear ----------------------------

#[test]
fn bmm_matches_a_host_batched_matmul() {
    require_wgpu();
    // BatchedMatMul requires rank >= 3; one batch of 2x2.
    let a = upload(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let b = upload(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 2]);
    let (out, recorded) = run2::<op::BatchedMatMul, _>(&a, &b, NoAttributes);
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    assert_close(&read(&out), &[19.0, 22.0, 43.0, 50.0], 1e-5, "bmm");
    assert!(recorded >= 1, "bmm advertises training = true");
}

#[test]
fn addmm_applies_alpha_beta_and_the_matmul() {
    require_wgpu();
    let mat = upload(&[10.0, 10.0, 10.0, 10.0], &[2, 2]);
    let a = upload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = upload(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&mat),
        TensorHandle::from_storage::<TestBackend, f32, _>(&a),
        TensorHandle::from_storage::<TestBackend, f32, _>(&b),
    ];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::Addmm, _>(
        &context,
        incin_core::exec::catalog::AddmmAttributes {
            alpha: 2.0,
            beta: 0.5,
        },
        &handles,
    )
    .expect("addmm is advertised and must execute");
    let recorded = incin_backends::wgpu::tape_depth() - before;
    // product = [[19,22],[43,50]]; 2*product = [[38,44],[86,100]]
    // 0.5*mat = [[5,5],[5,5]]; sum = [[43,49],[91,105]]
    assert_close(&read(&out), &[43.0, 49.0, 91.0, 105.0], 1e-4, "addmm");
    assert!(recorded >= 1, "addmm advertises training = true");
}

#[test]
fn dot_is_the_elementwise_product_then_an_all_sum() {
    require_wgpu();
    let a = upload(&[1.0, 2.0, 3.0], &[3]);
    let b = upload(&[4.0, 5.0, 6.0], &[3]);
    let (out, recorded) = run2::<op::Dot, _>(&a, &b, NoAttributes);
    assert_close(&read(&out), &[32.0], 1e-5, "dot");
    assert!(recorded >= 1, "dot advertises training = true");
}

#[test]
fn linear_without_bias_is_a_matrix_product() {
    require_wgpu();
    let input = upload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let weight = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&input),
        TensorHandle::from_storage::<TestBackend, f32, _>(&weight),
    ];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::Linear, _>(
        &context,
        LinearAttributes { has_bias: false },
        &handles,
    )
    .expect("linear without bias is advertised and must execute");
    let recorded = incin_backends::wgpu::tape_depth() - before;
    // identity weight: input @ I^T = input
    assert_close(&read(&out), &[1.0, 2.0, 3.0, 4.0], 1e-5, "linear identity");
    assert!(recorded >= 1, "linear advertises training = true");
}
