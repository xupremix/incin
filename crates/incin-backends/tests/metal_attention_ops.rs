//! Dispatch-level coverage for the Metal #92 attention-block ops.
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
    AttentionAttributes, AxisAttributes, DiagonalAttributes, DropoutAttributes, EpsilonAttributes,
    LayerNormAttributes, LinearAttributes, NarrowAttributes, SliceAttributes, TransposeAttributes,
};
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, Metal};

type TestBackend = MetalBackendImpl<Metal>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

/// Two rows of four. The second row is deliberately wide (`-8.0` next to
/// `8.0`): the input where the unstable `exp(x) / sum(exp(x))` spelling
/// overflows and the stable one does not.
const VALUES: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -8.0, 0.0, 8.0, 0.0];

/// A 3x4 row-major field so narrow/slice windows are unmistakable.
const FIELD: [f32; 12] = [
    0.0, 1.0, 2.0, 3.0, //
    10.0, 11.0, 12.0, 13.0, //
    20.0, 21.0, 22.0, 23.0,
];

/// Two rows of four, deliberately non-zero-mean so centering is exercised.
const NORM_IN: [f32; 8] = [1.0, 2.0, 3.0, 6.0, -4.0, 0.0, 2.0, 4.0];

/// 2x3, so a transpose is observable (a square one hides a row/column swap).
const RECT: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

const EPSILON: f64 = 1e-5;

/// Aborts unless the Metal device id can be admitted. Mirrors the WGPU
/// suites' `require_wgpu`: failing (not skipping) is right because these
/// suites are `#![cfg(feature = "metal")]` — compiling them is an explicit
/// request for the backend, and a silent skip would report `ok` for a test
/// that ran nothing.
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
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::metal(0),
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

/// One-input dispatch: runs the op and reports the tape entries it added.
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

/// Variadic dispatch for `concat`/`stack`.
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
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised multi-input Metal operation must execute");
    (out, tape_depth() - before)
}

/// Fixed-arity dispatch for two- and three-input ops (`linear`, `attention`).
fn run_n<O, A>(inputs_storage: &[&TestStorage], attributes: A) -> (TestStorage, usize)
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
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised fixed-arity Metal operation must execute");
    (out, tape_depth() - before)
}

/// Stable f64 softmax over `axis` of a row-major `[rows, cols]` input.
/// Axis 0 strided by `cols`, axis 1 contiguous — a helper that only handled
/// inner segments would silently check axis 1 twice.
fn ref_softmax(values: &[f32], rows: usize, cols: usize, axis: usize) -> Vec<f64> {
    let minor = if axis == 0 { rows } else { cols };
    let mut out = vec![0.0f64; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            let (base, stride, index) = if axis == 0 {
                (c, cols, r)
            } else {
                (r * cols, 1, c)
            };
            let mut max = f32::NEG_INFINITY;
            for k in 0..minor {
                max = max.max(values[base + k * stride]);
            }
            let mut total = 0.0f64;
            let mut exp = 0.0f64;
            for k in 0..minor {
                let e = f64::from(values[base + k * stride] - max).exp();
                total += e;
                if k == index {
                    exp = e;
                }
            }
            out[r * cols + c] = exp / total;
        }
    }
    out
}

/// `(x - mean) / sqrt(var + eps)` per row, plus an optional per-column bias.
fn layer_norm_reference(
    values: &[f32],
    rows: usize,
    cols: usize,
    bias: Option<&[f32]>,
) -> Vec<f64> {
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
        let inv = 1.0 / (var + EPSILON).sqrt();
        for (c, &x) in row.iter().enumerate() {
            let mut y = (f64::from(x) - mean) * inv;
            if let Some(b) = bias {
                y += f64::from(b[c]);
            }
            out[r * cols + c] = y;
        }
    }
    out
}

// ── softmax / log_softmax ──────────────────────────────────────────────────

#[test]
fn softmax_matches_the_stable_reference_and_records() {
    require_metal();
    let input = upload(&VALUES, &[2, 4]);
    let (out, recorded) = run1::<op::Softmax, _>(&input, AxisAttributes { axis: 1 });
    let got = read(&out);
    assert_close(&got, &ref_softmax(&VALUES, 2, 4, 1), 1e-5, "softmax axis 1");
    for row in got.chunks(4) {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "row sums to {sum}");
    }
    assert!(
        recorded >= 2,
        "softmax advertises training = true (got {recorded} entries)"
    );
}

#[test]
fn softmax_axis_zero_is_honoured_and_a_wide_row_stays_finite() {
    require_metal();
    let input = upload(&VALUES, &[4, 2]);
    let (out, _) = run1::<op::Softmax, _>(&input, AxisAttributes { axis: 0 });
    assert_close(
        &read(&out),
        &ref_softmax(&VALUES, 4, 2, 0),
        1e-5,
        "softmax axis 0",
    );

    const WIDE: [f32; 4] = [-100.0, 0.0, 100.0, 50.0];
    let wide = upload(&WIDE, &[1, 4]);
    let (out2, _) = run1::<op::Softmax, _>(&wide, AxisAttributes { axis: 1 });
    let got2 = read(&out2);
    assert!(
        got2.iter().all(|v| v.is_finite()),
        "the unstable spelling overflows this row; the stable one must not"
    );
    assert_close(&got2, &ref_softmax(&WIDE, 1, 4, 1), 1e-6, "wide softmax");
}

#[test]
fn log_softmax_matches_the_formula_and_records() {
    require_metal();
    let input = upload(&VALUES, &[2, 4]);
    let (out, recorded) = run1::<op::LogSoftmax, _>(&input, AxisAttributes { axis: 1 });
    let want: Vec<f64> = ref_softmax(&VALUES, 2, 4, 1)
        .iter()
        .map(|p| p.ln())
        .collect();
    assert_close(&read(&out), &want, 1e-5, "log_softmax");
    assert!(
        recorded >= 2,
        "log_softmax advertises training = true (got {recorded} entries)"
    );
}

// ── layer_norm / rms_norm ──────────────────────────────────────────────────

#[test]
fn layer_norm_without_bias_matches_the_reference_and_records() {
    require_metal();
    let input = upload(&NORM_IN, &[2, 4]);
    let weight = upload(&[1.0f32; 4], &[4]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&input),
        TensorHandle::from_storage::<TestBackend, f32, _>(&weight),
    ];
    let attributes = LayerNormAttributes {
        normalized_shape: vec![4],
        epsilon: EPSILON,
        has_bias: false,
    };
    let before = tape_depth();
    let out =
        incin_core::exec::dispatch::execute::<op::LayerNorm, _>(&context, attributes, &handles)
            .expect("layer_norm is advertised and must execute");
    let recorded = tape_depth() - before;
    assert_close(
        &read(&out),
        &layer_norm_reference(&NORM_IN, 2, 4, None),
        1e-5,
        "layer_norm",
    );
    assert!(recorded >= 1, "layer_norm advertises training = true");
}

#[test]
fn layer_norm_with_bias_applies_the_affine_shift_and_records() {
    require_metal();
    let bias_values = [0.5f32, -1.0, 0.25, 2.0];
    let input = upload(&NORM_IN, &[2, 4]);
    let weight = upload(&[1.0f32; 4], &[4]);
    let bias = upload(&bias_values, &[4]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&input),
        TensorHandle::from_storage::<TestBackend, f32, _>(&weight),
        TensorHandle::from_storage::<TestBackend, f32, _>(&bias),
    ];
    let attributes = LayerNormAttributes {
        normalized_shape: vec![4],
        epsilon: EPSILON,
        has_bias: true,
    };
    let before = tape_depth();
    let out =
        incin_core::exec::dispatch::execute::<op::LayerNorm, _>(&context, attributes, &handles)
            .expect("layer_norm with bias is advertised and must execute");
    let recorded = tape_depth() - before;
    assert_close(
        &read(&out),
        &layer_norm_reference(&NORM_IN, 2, 4, Some(&bias_values)),
        1e-5,
        "layer_norm with bias",
    );
    assert!(recorded >= 1, "layer_norm advertises training = true");
}

#[test]
fn rms_norm_matches_the_reference_and_records() {
    require_metal();
    const RVALUES: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, -2.0, 3.0];
    const WEIGHT: [f32; 4] = [1.0, 0.5, 2.0, 1.5];

    let input = upload(&RVALUES, &[2, 4]);
    let weight = upload(&WEIGHT, &[4]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&input),
        TensorHandle::from_storage::<TestBackend, f32, _>(&weight),
    ];
    let before = tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::RmsNorm, _>(
        &context,
        EpsilonAttributes { epsilon: EPSILON },
        &handles,
    )
    .expect("rms_norm is advertised and must execute");
    let recorded = tape_depth() - before;

    let mut want = vec![0.0f64; RVALUES.len()];
    for row in 0..2 {
        let slice = &RVALUES[row * 4..(row + 1) * 4];
        let mean_square: f64 = slice
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            / 4.0;
        let scale = (mean_square + EPSILON).sqrt();
        for (column, value) in slice.iter().enumerate() {
            want[row * 4 + column] = (f64::from(*value) / scale) * f64::from(WEIGHT[column]);
        }
    }
    assert_close(&read(&out), &want, 1e-5, "rms_norm");
    assert!(recorded >= 1, "rms_norm advertises training = true");
}

#[test]
fn rms_norm_an_all_zero_row_stays_finite() {
    require_metal();
    const ZEROS: [f32; 8] = [0.0; 8];
    let input = upload(&ZEROS, &[2, 4]);
    let weight = upload(&[1.0f32; 4], &[4]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&input),
        TensorHandle::from_storage::<TestBackend, f32, _>(&weight),
    ];
    let out = incin_core::exec::dispatch::execute::<op::RmsNorm, _>(
        &context,
        EpsilonAttributes { epsilon: EPSILON },
        &handles,
    )
    .expect("rms_norm is advertised and must execute");
    let got = read(&out);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "sqrt(0) without the epsilon guard would divide by zero: {got:?}"
    );
    assert_close(&got, &[0.0; 8], 0.0, "zero rms_norm");
}

// ── transpose / narrow / slice ─────────────────────────────────────────────

#[test]
fn transpose_reorders_a_rectangular_tensor_and_round_trips() {
    require_metal();
    let input = upload(&RECT, &[2, 3]);
    let (out, recorded) = run1::<op::TransposeExact, _>(
        &input,
        TransposeAttributes {
            first: 0,
            second: 1,
        },
    );
    // [[1,2,3],[4,5,6]] transposed is [[1,4],[2,5],[3,6]].
    assert_close(
        &read(&out),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        0.0,
        "transpose",
    );
    assert!(recorded >= 1, "transpose advertises training = true");

    let (back, _) = run1::<op::TransposeExact, _>(
        &out,
        TransposeAttributes {
            first: 1,
            second: 0,
        },
    );
    assert_close(&read(&back), &read(&input), 0.0, "transpose twice");
}

#[test]
fn narrow_takes_the_requested_window_and_records() {
    require_metal();
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

    let (along, _) = run1::<op::Narrow, _>(
        &input,
        NarrowAttributes {
            axis: 0,
            start: 1,
            length: 2,
        },
    );
    assert_close(
        &read(&along),
        &[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0],
        0.0,
        "narrow axis 0",
    );
}

#[test]
fn slice_exact_takes_a_per_axis_window() {
    require_metal();
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

// ── concat / stack ─────────────────────────────────────────────────────────

#[test]
fn concat_joins_along_axis_zero() {
    require_metal();
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
    require_metal();
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
    require_metal();
    let a = upload(&[1.0, 2.0], &[2]);
    let b = upload(&[3.0, 4.0], &[2]);
    let (out, recorded) = run_many::<op::StackExact, _>(&[&a, &b], AxisAttributes { axis: 0 });
    assert_close(&read(&out), &[1.0, 2.0, 3.0, 4.0], 0.0, "stack");
    assert_eq!(
        TestBackend::shape::<f32>(&out).dims(),
        &[2, 2],
        "stack axis 0 of two rank-1 operands is [2, 2]"
    );
    assert!(recorded >= 1, "stack advertises training = true");
}

// ── squeeze / unsqueeze ────────────────────────────────────────────────────

#[test]
fn unsqueeze_adds_a_unit_axis_and_records() {
    require_metal();
    let input = upload(&RECT, &[2, 3]);
    let (out, recorded) = run1::<op::UnsqueezeExact, _>(&input, AxisAttributes { axis: 2 });
    assert_eq!(
        TestBackend::shape::<f32>(&out).dims(),
        &[2, 3, 1],
        "unsqueeze at rank inserts a trailing unit axis"
    );
    assert_close(&read(&out), &read(&input), 0.0, "unsqueeze values");
    assert!(recorded >= 1, "unsqueeze advertises training = true");
}

#[test]
fn squeeze_drops_a_unit_axis_and_round_trips() {
    require_metal();
    let input = upload(&RECT, &[1, 2, 3]);
    let (squeezed, recorded) = run1::<op::SqueezeExact, _>(&input, AxisAttributes { axis: 0 });
    assert_eq!(
        TestBackend::shape::<f32>(&squeezed).dims(),
        &[2, 3],
        "squeeze drops the requested unit axis"
    );
    assert_close(&read(&squeezed), &read(&input), 0.0, "squeeze values");
    assert!(recorded >= 1, "squeeze advertises training = true");

    let (back, _) = run1::<op::UnsqueezeExact, _>(&squeezed, AxisAttributes { axis: 0 });
    assert_close(&read(&back), &read(&input), 0.0, "unsqueeze after squeeze");
}

// ── tril / triu ────────────────────────────────────────────────────────────

#[test]
fn tril_and_triu_mask_the_correct_half_and_record() {
    require_metal();
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

    // The offset shifts the kept diagonal: `diag >= 1` above, `<= 1` below.
    let (shifted, _) = run1::<op::Triu, _>(&m, DiagonalAttributes { offset: 1 });
    assert_close(
        &read(&shifted),
        &[0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0],
        0.0,
        "triu offset 1",
    );
}

// ── dropout ────────────────────────────────────────────────────────────────

/// Two rows of three, mixed signs, for the keep-mask walk.
const DROPOUT_IN: [f32; 6] = [1.0, -2.0, 3.0, -4.0, 5.0, 6.0];

#[test]
fn dropout_inference_is_identity_and_training_zeroes_or_scales() {
    require_metal();
    let input = upload(&DROPOUT_IN, &[2, 3]);
    // training=false: the identity path must still execute through dispatch.
    let (out, recorded_off) = run1::<op::Dropout, _>(
        &input,
        DropoutAttributes {
            probability: 0.5,
            training: false,
        },
    );
    assert_close(
        &read(&out),
        &DROPOUT_IN.map(f64::from),
        0.0,
        "dropout inference",
    );
    assert_eq!(
        recorded_off, 0,
        "the identity path pushes no tape entry of its own"
    );

    // training=true with p in (0,1): a scaled keep-mask — each element is
    // either 0 or x / (1 - p), and the mul/mul_scalar chain records.
    let (train, recorded_tr) = run1::<op::Dropout, _>(
        &input,
        DropoutAttributes {
            probability: 0.5,
            training: true,
        },
    );
    assert!(
        recorded_tr >= 1,
        "dropout training advertises training = true"
    );
    let actual = read(&train);
    assert_eq!(
        actual.len(),
        DROPOUT_IN.len(),
        "dropout preserves the element count"
    );
    for (i, a) in actual.iter().enumerate() {
        let x = f64::from(DROPOUT_IN[i]);
        let scaled = x / 0.5;
        assert!(
            *a == 0.0 || (a - scaled).abs() <= 1e-5,
            "dropout training[{i}]: got {a}, expected 0 or {scaled}"
        );
    }
}

// ── linear ─────────────────────────────────────────────────────────────────

#[test]
fn linear_without_bias_is_a_matrix_product_and_records() {
    require_metal();
    let input = upload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let weight = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let (out, recorded) =
        run_n::<op::Linear, _>(&[&input, &weight], LinearAttributes { has_bias: false });
    // identity weight: input @ I^T = input
    assert_close(&read(&out), &[1.0, 2.0, 3.0, 4.0], 1e-5, "linear identity");
    assert!(recorded >= 1, "linear advertises training = true");
}

#[test]
fn linear_with_bias_adds_the_per_column_shift_and_records() {
    require_metal();
    let input = upload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let weight = upload(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let bias = upload(&[0.5, -0.5], &[2]);
    let (out, recorded) = run_n::<op::Linear, _>(
        &[&input, &weight, &bias],
        LinearAttributes { has_bias: true },
    );
    assert_close(&read(&out), &[1.5, 1.5, 3.5, 3.5], 1e-5, "linear with bias");
    assert!(recorded >= 1, "linear with bias advertises training = true");
}

// ── scaled_dot_product_attention ───────────────────────────────────────────

#[test]
fn scaled_dot_product_attention_without_mask_records() {
    require_metal();
    // Single head, two keys: q/k identity, v a plain block — the same
    // fixture WGPU's Batch-C attention test pins.
    let q = upload(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
    let k = upload(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
    let v = upload(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let (out, recorded) = run_n::<op::ScaledDotProductAttention, _>(
        &[&q, &k, &v],
        AttentionAttributes {
            scale: None,
            has_mask: false,
        },
    );
    assert_eq!(
        TestBackend::shape::<f32>(&out).dims(),
        &[1, 2, 2],
        "attention keeps the query shape"
    );
    // scores = q @ k^T / sqrt(d_k); with q = k = I this is I / sqrt(2).
    // Host softmax of [s, 0] with s = 1/sqrt(2): [e^s, 1] / (e^s + 1).
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let p_keep = s.exp() / (s.exp() + 1.0);
    let p_other = 1.0 / (s.exp() + 1.0);
    // row0 = p_keep * [1,2] + p_other * [3,4]; row1 swaps the weights.
    let expected = [
        p_keep * 1.0 + p_other * 3.0,
        p_keep * 2.0 + p_other * 4.0,
        p_other * 1.0 + p_keep * 3.0,
        p_other * 2.0 + p_keep * 4.0,
    ];
    assert_close(&read(&out), &expected, 1e-4, "attention");
    assert!(recorded >= 1, "attention advertises training = true");
}

#[test]
fn scaled_dot_product_attention_with_a_zero_mask_equals_the_plain_chain() {
    require_metal();
    let q = upload(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
    let k = upload(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
    let v = upload(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let mask = upload(&[0.0; 4], &[1, 2, 2]);
    let (plain, _) = run_n::<op::ScaledDotProductAttention, _>(
        &[&q, &k, &v],
        AttentionAttributes {
            scale: None,
            has_mask: false,
        },
    );
    let (masked, recorded) = run_n::<op::ScaledDotProductAttention, _>(
        &[&q, &k, &v, &mask],
        AttentionAttributes {
            scale: None,
            has_mask: true,
        },
    );
    assert_close(
        &read(&masked),
        &read(&plain),
        1e-5,
        "attention with a zero additive mask",
    );
    assert!(recorded >= 1, "masked attention advertises training = true");
}
