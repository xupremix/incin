//! Batch-C WGPU gap closure on real hardware: log-softmax/logsumexp/cumsum,
//! instance/dropout/bce, outer/SDPA, and the structural repeat/pad/chunk/split
//! family.
//!
//! Every operation advertised in the Batch-C capability update must prove:
//!
//! - forward values against a host reference computed with the same formulas
//!   CPU uses (f32 tolerance for statistical ops, exact for structural ones);
//! - at least one tape entry for every row that claims `training = true`,
//!   because a forward that returns the right numbers but records nothing
//!   would silently drop the gradient.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{HostInterop, HostReadback, StorageBackend, op};
use incin_core::exec::catalog::{
    AttentionAttributes, AxisAttributes, ChunkAttributes, DropoutAttributes, EpsilonAttributes,
    LossAttributes, LossReduction, NoAttributes, PadAttributes, RepeatAttributes,
    RepeatInterleaveAttributes, ShapeAttributes, SplitAttributes,
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
        .expect("an advertised Batch-C operation must execute");
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
        .expect("an advertised two-input Batch-C operation must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

// --- elementwise / reduction: log_softmax / logsumexp / cumsum -------------

/// Two-by-three rows with distinct magnitudes so the max-shift is non-trivial.
const LS_IN: [f32; 6] = [1.0, 2.0, 3.0, -1.0, 0.0, 1.0];

fn host_log_softmax(values: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        let row = &values[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row
            .iter()
            .map(|&x| (f64::from(x) - f64::from(max)).exp())
            .sum();
        for (c, &x) in row.iter().enumerate() {
            out[r * cols + c] = (f64::from(x) - f64::from(max)).exp().ln() - sum.ln();
        }
    }
    out
}

fn host_logsumexp_dim(values: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &values[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row
            .iter()
            .map(|&x| (f64::from(x) - f64::from(max)).exp())
            .sum();
        out.push(f64::from(max) + sum.ln());
    }
    out
}

fn host_logsumexp_keepdim(values: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    host_logsumexp_dim(values, rows, cols)
        .into_iter()
        .flat_map(|v| [v])
        .collect()
}

fn host_cumsum(values: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        let mut acc = 0.0f64;
        for c in 0..cols {
            acc += f64::from(values[r * cols + c]);
            out[r * cols + c] = acc;
        }
    }
    out
}

#[test]
fn log_softmax_matches_the_host_reference_and_records() {
    require_wgpu();
    let input = upload(&LS_IN, &[2, 3]);
    let (out, recorded) = run1::<op::LogSoftmax, _>(&input, AxisAttributes { axis: 1 });
    assert_close(
        &read(&out),
        &host_log_softmax(&LS_IN, 2, 3),
        1e-5,
        "log_softmax",
    );
    assert!(recorded >= 1, "log_softmax advertises training = true");
}

#[test]
fn logsumexp_dim_and_keepdim_match_the_host_reference() {
    require_wgpu();
    let input = upload(&LS_IN, &[2, 3]);
    let (dim, recorded) = run1::<op::LogSumExpDim, _>(&input, AxisAttributes { axis: 1 });
    assert_close(
        &read(&dim),
        &host_logsumexp_dim(&LS_IN, 2, 3),
        1e-5,
        "logsumexp dim",
    );
    assert!(recorded >= 1, "logsumexp dim advertises training = true");

    let (keep, _) = run1::<op::LogSumExpKeepDim, _>(&input, AxisAttributes { axis: 1 });
    assert_close(
        &read(&keep),
        &host_logsumexp_keepdim(&LS_IN, 2, 3),
        1e-5,
        "logsumexp keepdim",
    );
    assert_eq!(
        keep.shape.to_vec(),
        vec![2, 1],
        "logsumexp keepdim must leave the reduced axis in place"
    );
}

#[test]
fn cumsum_accumulates_along_the_requested_axis_and_records() {
    require_wgpu();
    let input = upload(&LS_IN, &[2, 3]);
    let (out, recorded) = run1::<op::Cumsum, _>(&input, AxisAttributes { axis: 1 });
    assert_close(&read(&out), &host_cumsum(&LS_IN, 2, 3), 1e-5, "cumsum");
    assert!(recorded >= 1, "cumsum advertises training = true");
}

// --- nn: instance_norm / dropout / bce_with_logits --------------------------

/// N-channel input (1, 4, 2, 2) so per-instance stats are well defined.
const IN_IN: [f32; 16] = [
    1.0, 2.0, 3.0, 4.0, //
    5.0, 6.0, 7.0, 8.0, //
    9.0, 10.0, 11.0, 12.0, //
    13.0, 14.0, 15.0, 16.0,
];

/// Instance norm is `group_norm` with one group per channel: each
/// `(sample, channel)` vector is normalized alone over its spatial extent.
fn host_instance_norm(values: &[f32], shape: &[usize], eps: f64) -> Vec<f64> {
    let (n, c, hw) = (shape[0], shape[1], shape[2..].iter().product::<usize>());
    let mut out = vec![0.0; values.len()];
    for bi in 0..n {
        for ci in 0..c {
            let start = (bi * c + ci) * hw;
            let group = &values[start..start + hw];
            let mean = group.iter().map(|&x| f64::from(x)).sum::<f64>() / hw as f64;
            let var = group
                .iter()
                .map(|&x| {
                    let d = f64::from(x) - mean;
                    d * d
                })
                .sum::<f64>()
                / hw as f64;
            let inv = 1.0 / (var + eps).sqrt();
            for (j, &x) in group.iter().enumerate() {
                out[start + j] = (f64::from(x) - mean) * inv;
            }
        }
    }
    out
}

#[test]
fn instance_norm_matches_the_host_reference_and_records() {
    require_wgpu();
    let input = upload(&IN_IN, &[1, 4, 2, 2]);
    let (out, recorded) = run1::<op::InstanceNorm, _>(&input, EpsilonAttributes { epsilon: 1e-5 });
    assert_close(
        &read(&out),
        &host_instance_norm(&IN_IN, &[1, 4, 2, 2], 1e-5),
        1e-4,
        "instance_norm",
    );
    assert!(recorded >= 1, "instance_norm advertises training = true");
}

#[test]
fn dropout_inference_is_identity_and_training_records() {
    require_wgpu();
    let input = upload(&LS_IN, &[2, 3]);
    // training=false: the clone-links-identity path returns the operand
    // itself (same tensor id), so a gradient arriving there needs no entry
    // of its own — matching CPU, CUDA and Metal's eval-mode dropout.
    let (out, recorded_id) = run1::<op::Dropout, _>(
        &input,
        DropoutAttributes {
            probability: 0.5,
            training: false,
        },
    );
    assert_close(
        &read(&out),
        LS_IN
            .iter()
            .map(|&x| f64::from(x))
            .collect::<Vec<_>>()
            .as_slice(),
        0.0,
        "dropout inference",
    );
    assert_eq!(
        incin_core::exec::TapeStorage::id(&out),
        incin_core::exec::TapeStorage::id(&input),
        "eval-mode dropout is the clone-links-identity: same tensor id, no \
         tape entry of its own"
    );
    let _ = recorded_id;

    // training=true with p in (0,1): the output is a scaled keep-mask; the
    // magnitude of each kept element is x / (1-p), dropped elements are 0.
    // The mul / mul_scalar composition is what records the tape entry.
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
        LS_IN.len(),
        "dropout preserves the element count"
    );
    for (i, a) in actual.iter().enumerate() {
        let x = f64::from(LS_IN[i]);
        let scaled = x / 0.5;
        assert!(
            *a == 0.0 || (a - scaled).abs() <= 1e-5,
            "dropout training[{i}]: got {a}, expected 0 or {scaled}"
        );
    }
}

#[test]
fn bce_with_logits_matches_the_host_reference_and_records() {
    require_wgpu();
    let pred = upload(&[2.0, -2.0, 0.0], &[3]);
    let target = upload(&[1.0, 0.0, 1.0], &[3]);
    let mean = LossAttributes {
        reduction: LossReduction::Mean,
    };
    let (out, recorded) = run2::<op::BceWithLogitsLoss, _>(&pred, &target, mean);
    // Host: max(x,0) - x*y + log(1 + exp(-|x|)), then mean.
    // (2,1): 0 + log(1+e^-2) = 0.126928
    // (-2,0): 0 + log(1+e^-2) = 0.126928
    // (0,1): 0 + log 2 = 0.693147
    // mean ≈ 0.315668
    assert_close(
        &read(&out),
        &[0.3156677484512329],
        1e-5,
        "bce_with_logits mean",
    );
    assert!(recorded >= 1, "bce_with_logits advertises training = true");
}

// --- linalg: outer ---------------------------------------------------------

#[test]
fn outer_is_the_rank_one_product_and_records() {
    require_wgpu();
    let lhs = upload(&[1.0, 2.0], &[2]);
    let rhs = upload(&[3.0, 4.0, 5.0], &[3]);
    let (out, recorded) = run2::<op::Outer, _>(&lhs, &rhs, NoAttributes);
    assert_close(&read(&out), &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0], 1e-5, "outer");
    assert_eq!(out.shape.to_vec(), vec![2, 3], "outer shape is lhs x rhs");
    assert!(recorded >= 1, "outer advertises training = true");
}

// --- nn: scaled_dot_product_attention ---------------------------------------

#[test]
fn scaled_dot_product_attention_without_mask_records() {
    require_wgpu();
    // Single head, two keys: q/k/v all [1, 2, 2].
    let q = upload(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
    let k = upload(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
    let v = upload(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let context = ExecutionContext::new(TestBackend::default());
    let handles = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&q),
        TensorHandle::from_storage::<TestBackend, f32, _>(&k),
        TensorHandle::from_storage::<TestBackend, f32, _>(&v),
    ];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::ScaledDotProductAttention, _>(
        &context,
        AttentionAttributes {
            scale: None,
            has_mask: false,
        },
        &handles,
    )
    .expect("attention without mask is advertised and must execute");
    let recorded = incin_backends::wgpu::tape_depth() - before;
    // scores = q @ k^T / sqrt(d_k); with q = k = I this is I / sqrt(2).
    // softmax over the last axis, then @ v. Output keeps the query shape.
    let actual = read(&out);
    assert_eq!(
        out.shape.to_vec(),
        vec![1, 2, 2],
        "attention keeps the query shape"
    );
    assert_eq!(
        actual.len(),
        4,
        "attention returns one value per query element"
    );
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
    assert_close(&actual, &expected, 1e-4, "attention");
    assert!(recorded >= 1, "attention advertises training = true");
}

// --- structural: broadcast_left / pad / repeat / repeat_interleave ----------

#[test]
fn broadcast_left_prepends_dimensions_and_records() {
    require_wgpu();
    let input = upload(&[1.0, 2.0], &[2]);
    let (out, recorded) =
        run1::<op::BroadcastLeft, _>(&input, ShapeAttributes { shape: vec![1, 2] });
    assert_close(&read(&out), &[1.0, 2.0], 0.0, "broadcast_left");
    assert_eq!(
        out.shape.to_vec(),
        vec![1, 2],
        "broadcast_left prepends the prefix"
    );
    assert!(recorded >= 1, "broadcast_left advertises training = true");
}

#[test]
fn pad_fills_the_window_with_the_constant_and_records() {
    require_wgpu();
    let input = upload(&LS_IN, &[2, 3]);
    let (out, recorded) = run1::<op::Pad, _>(
        &input,
        PadAttributes {
            padding: vec![(1, 1), (0, 1)],
            value: 0.0,
        },
    );
    // shape: (2+2, 3+1) = (4, 4)
    let expected = [
        0.0, 0.0, 0.0, 0.0, //
        1.0, 2.0, 3.0, 0.0, //
        -1.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_close(&read(&out), &expected, 0.0, "pad");
    assert_eq!(out.shape.to_vec(), vec![4, 4], "pad shape");
    assert!(recorded >= 1, "pad advertises training = true");
}

#[test]
fn repeat_tiles_along_every_axis_and_records() {
    require_wgpu();
    let input = upload(&[1.0, 2.0], &[2]);
    let (out, recorded) = run1::<op::Repeat, _>(&input, RepeatAttributes { repeats: vec![2] });
    assert_close(&read(&out), &[1.0, 2.0, 1.0, 2.0], 0.0, "repeat");
    assert_eq!(out.shape.to_vec(), vec![4], "repeat shape");
    assert!(recorded >= 1, "repeat advertises training = true");
}

#[test]
fn repeat_interleave_repeats_each_element_and_records() {
    require_wgpu();
    let input = upload(&[1.0, 2.0], &[2]);
    let (out, recorded) = run1::<op::RepeatInterleave, _>(
        &input,
        RepeatInterleaveAttributes {
            repeats: 2,
            axis: 0,
        },
    );
    assert_close(&read(&out), &[1.0, 1.0, 2.0, 2.0], 0.0, "repeat_interleave");
    assert_eq!(out.shape.to_vec(), vec![4], "repeat_interleave shape");
    assert!(
        recorded >= 1,
        "repeat_interleave advertises training = true"
    );
}

// --- structural multi-output: chunk / split ---------------------------------

#[test]
fn chunk_splits_into_consecutive_pieces_and_records() {
    require_wgpu();
    let input = upload(&LS_IN, &[2, 3]);
    let outs = [TensorHandle::from_storage::<TestBackend, f32, _>(&input)];
    let context = ExecutionContext::new(TestBackend::default());
    let handles = outs;
    let before = incin_backends::wgpu::tape_depth();
    let pieces = incin_core::exec::dispatch::execute::<op::Chunk, _>(
        &context,
        ChunkAttributes { chunks: 3, axis: 1 },
        &handles,
    )
    .expect("chunk is advertised and must execute");
    let recorded = incin_backends::wgpu::tape_depth() - before;
    assert_eq!(pieces.len(), 3, "chunk produces three pieces");
    assert_close(&read(&pieces[0]), &[1.0, -1.0], 0.0, "chunk 0");
    assert_close(&read(&pieces[1]), &[2.0, 0.0], 0.0, "chunk 1");
    assert_close(&read(&pieces[2]), &[3.0, 1.0], 0.0, "chunk 2");
    assert!(recorded >= 1, "chunk advertises training = true");
}

#[test]
fn split_slices_by_split_size_and_records() {
    require_wgpu();
    let input = upload(&LS_IN, &[2, 3]);
    let outs = [TensorHandle::from_storage::<TestBackend, f32, _>(&input)];
    let context = ExecutionContext::new(TestBackend::default());
    let handles = outs;
    let before = incin_backends::wgpu::tape_depth();
    let pieces = incin_core::exec::dispatch::execute::<op::Split, _>(
        &context,
        SplitAttributes {
            split_size: 2,
            axis: 1,
        },
        &handles,
    )
    .expect("split is advertised and must execute");
    let recorded = incin_backends::wgpu::tape_depth() - before;
    // axis extent 3, split_size 2 -> consecutive pieces of size 2 and 1.
    assert_eq!(pieces.len(), 2, "split produces two pieces");
    assert_close(&read(&pieces[0]), &[1.0, 2.0, -1.0, 0.0], 0.0, "split 0");
    assert_close(&read(&pieces[1]), &[3.0, 1.0], 0.0, "split 1");
    assert!(recorded >= 1, "split advertises training = true");
}
