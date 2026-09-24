//! Issue #106: the six CUDA gap ops' hand fixtures, refusals, and tape
//! connectivity — `conv1d`, `conv_transpose2d`, `adaptive_avg_pool2d`,
//! `to_dtype`, `quantize`, `dequantize`.
//!
//! Requires a GPU:
//! `cargo test -p incin-backends --features cuda --test cuda_conv_spatial -- --ignored`.
#![cfg(feature = "cuda")]

use half::f16;
use incin_backends::cuda::{
    CudaBackendImpl, tape_depth,
    testing::{download_bytes, download_f32, require_cuda, upload_f32_shaped},
};
use incin_core::backend_authoring::{AutogradBackend, StorageBackend};
use incin_core::exec::catalog::{
    AdaptivePool2dAttributes, Conv1dAttributes, ConvTranspose2dAttributes, DTypeAttributes,
    QuantizationAttributes,
};
use incin_core::exec::{ExecutionContext, TapeStorage, TensorHandle, dispatch, op};
use incin_core::prelude::{CudaN, DTypeId, Local};
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

fn context() -> ExecutionContext<TestBackend> {
    ExecutionContext::new(TestBackend::new())
}

fn handle(storage: &TestStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<TestBackend, f32, Local>(storage)
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// conv1d
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA hardware"]
fn conv1d_forward_hand_computed_no_padding() {
    require_cuda();
    let ctx = context();
    let act = upload_f32_shaped(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
    let w = upload_f32_shaped(&[1, 1, 2], &[10.0, 1.0]);
    let before = tape_depth();
    let out = dispatch::execute::<op::Conv1dExact, _>(
        &ctx,
        Conv1dAttributes {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            has_bias: false,
        },
        &[handle(&act), handle(&w)],
    )
    .expect("conv1d executes");
    assert_eq!(
        tape_depth() - before,
        1,
        "conv1d records one tape entry (via conv2d composition)"
    );
    assert_eq!(out.shape.to_vec(), vec![1, 1, 3]);
    assert_close(
        &download_f32(&out),
        &[12.0, 23.0, 34.0],
        1e-5,
        "conv1d forward",
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv1d_backward_overlapping_windows_accumulates_grad_input() {
    require_cuda();
    let ctx = context();
    let act = upload_f32_shaped(&[1, 1, 3], &[1.0, 2.0, 3.0]);
    let w = upload_f32_shaped(&[1, 1, 2], &[1.0, 1.0]);
    let act_id = TapeStorage::id(&act);
    let out = dispatch::execute::<op::Conv1dExact, _>(
        &ctx,
        Conv1dAttributes {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            has_bias: false,
        },
        &[handle(&act), handle(&w)],
    )
    .expect("conv1d executes");
    let grads =
        <TestBackend as AutogradBackend>::backward::<f32>(&out).expect("conv1d backward on CUDA");
    let g = grads.get(act_id).expect("conv1d input receives a gradient");
    assert_eq!(g.shape.to_vec(), vec![1, 1, 3]);
    // window0 covers input[0],input[1]; window1 covers input[1],input[2].
    // sum-loss: grad_input = [w[0], w[1]+w[0], w[1]] = [1, 2, 1].
    assert_close(&download_f32(g), &[1.0, 2.0, 1.0], 1e-5, "conv1d grad");
}

// ---------------------------------------------------------------------------
// conv_transpose2d
// ---------------------------------------------------------------------------

fn transpose_attrs(
    stride: [usize; 2],
    padding: [usize; 2],
    output_padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> ConvTranspose2dAttributes {
    ConvTranspose2dAttributes {
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        has_bias: false,
    }
}

fn run_transpose(
    ctx: &ExecutionContext<TestBackend>,
    act: &TestStorage,
    w: &TestStorage,
    attrs: ConvTranspose2dAttributes,
) -> Result<TestStorage, CanonicalErrorWrapper> {
    dispatch::execute::<op::ConvTranspose2d, _>(ctx, attrs, &[handle(act), handle(w)])
        .map_err(CanonicalErrorWrapper)
}

/// Wraps `CanonicalError` so `?`/`expect_err` can use it without leaking the
/// private enum into every test signature.
struct CanonicalErrorWrapper(incin_core::exec::CanonicalError);

impl std::fmt::Display for CanonicalErrorWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::fmt::Debug for CanonicalErrorWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv_transpose2d_forward_hand_computed_basic() {
    require_cuda();
    let ctx = context();
    let act = upload_f32_shaped(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let w = upload_f32_shaped(&[1, 1, 2, 2], &[1.0; 4]);
    let before = tape_depth();
    let out = run_transpose(
        &ctx,
        &act,
        &w,
        transpose_attrs([1, 1], [0, 0], [0, 0], [1, 1], 1),
    )
    .expect("conv_transpose2d executes");
    assert!(
        tape_depth() - before >= 1,
        "conv_transpose2d must record at least the col2im tape entry"
    );
    assert_eq!(out.shape.to_vec(), vec![1, 1, 3, 3]);
    assert_close(
        &download_f32(&out),
        &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0],
        1e-5,
        "conv_transpose2d basic",
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv_transpose2d_forward_stride_upsamples() {
    require_cuda();
    let ctx = context();
    let act = upload_f32_shaped(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let w = upload_f32_shaped(&[1, 1, 2, 2], &[1.0; 4]);
    let out = run_transpose(
        &ctx,
        &act,
        &w,
        transpose_attrs([2, 2], [0, 0], [0, 0], [1, 1], 1),
    )
    .expect("stride-2 conv_transpose2d executes");
    assert_eq!(out.shape.to_vec(), vec![1, 1, 4, 4]);
    assert_close(
        &download_f32(&out),
        &[
            1.0, 1.0, 2.0, 2.0, //
            1.0, 1.0, 2.0, 2.0, //
            3.0, 3.0, 4.0, 4.0, //
            3.0, 3.0, 4.0, 4.0,
        ],
        1e-5,
        "conv_transpose2d stride-2",
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv_transpose2d_output_padding_appends_trailing_zeros_only() {
    require_cuda();
    let ctx = context();
    let act = upload_f32_shaped(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let w = upload_f32_shaped(&[1, 1, 2, 2], &[1.0; 4]);
    let out = run_transpose(
        &ctx,
        &act,
        &w,
        transpose_attrs([2, 2], [0, 0], [1, 1], [1, 1], 1),
    )
    .expect("output_padding conv_transpose2d executes");
    assert_eq!(out.shape.to_vec(), vec![1, 1, 5, 5]);
    let vals = download_f32(&out);
    // Natural [0..4, 0..4] region matches the no-output-padding result;
    // the trailing row and column are exactly zero.
    let natural = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    for row in 0..4 {
        for col in 0..4 {
            let got = vals[row * 5 + col];
            let want = natural[row * 4 + col];
            assert!(
                (got - want).abs() <= 1e-5,
                "leading [{row},{col}]: got {got}, want {want}"
            );
        }
    }
    for col in 0..5 {
        assert_eq!(vals[4 * 5 + col], 0.0, "trailing row [{col}] must be 0");
    }
    for row in 0..5 {
        assert_eq!(vals[row * 5 + 4], 0.0, "trailing col [{row}] must be 0");
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv_transpose2d_backward_reaches_input_and_weight() {
    require_cuda();
    let ctx = context();
    let act = upload_f32_shaped(&[1, 1, 2, 2], &[0.1, 0.2, 0.3, 0.4]);
    let w = upload_f32_shaped(&[1, 1, 2, 2], &[0.5, 0.6, 0.7, 0.8]);
    let (act_id, w_id) = (TapeStorage::id(&act), TapeStorage::id(&w));
    let out = run_transpose(
        &ctx,
        &act,
        &w,
        transpose_attrs([1, 1], [0, 0], [0, 0], [1, 1], 1),
    )
    .expect("conv_transpose2d executes");
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&out)
        .expect("conv_transpose2d backward on CUDA");
    let ga = grads
        .get(act_id)
        .expect("conv_transpose2d input receives a gradient");
    let gw = grads
        .get(w_id)
        .expect("conv_transpose2d weight receives a gradient");
    assert_eq!(ga.shape.to_vec(), vec![1, 1, 2, 2]);
    assert_eq!(gw.shape.to_vec(), vec![1, 1, 2, 2]);
    // Forward scatters the input through a ones kernel into a 3x3; with a
    // ones-seed on the 3x3, d(sum)/d(w[tap]) = sum of every input that
    // lands under that tap. Every tap (out is 3x3, k=2) sees all four
    // inputs exactly once, so grad_weight = [1; 4].
    let ga_vals = download_f32(ga);
    let gw_vals = download_f32(gw);
    assert!(
        ga_vals.iter().all(|v| v.is_finite()),
        "grad_input must be finite: {ga_vals:?}"
    );
    assert!(
        gw_vals.iter().all(|v| v.is_finite()),
        "grad_weight must be finite: {gw_vals:?}"
    );
    assert_close(&gw_vals, &[1.0; 4], 1e-5, "conv_transpose2d grad_weight");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv_transpose2d_refuses_anisotropic_stride() {
    require_cuda();
    let ctx = context();
    let act = upload_f32_shaped(&[1, 1, 2, 2], &[1.0; 4]);
    let w = upload_f32_shaped(&[1, 1, 2, 2], &[1.0; 4]);
    let err = run_transpose(
        &ctx,
        &act,
        &w,
        transpose_attrs([2, 1], [0, 0], [0, 0], [1, 1], 1),
    )
    .expect_err("anisotropic stride must be refused");
    let message = format!("{err}");
    assert!(
        message.contains("strides differ per axis"),
        "refusal must name the anisotropic stride: {message}"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv_transpose2d_refuses_groups_other_than_one() {
    require_cuda();
    let ctx = context();
    // groups=2, weight [1,1,2,2]: inference sees Cin match and Cout=weight[1]*2.
    let act = upload_f32_shaped(&[1, 1, 2, 2], &[1.0; 4]);
    let w = upload_f32_shaped(&[1, 1, 2, 2], &[1.0; 4]);
    let err = run_transpose(
        &ctx,
        &act,
        &w,
        transpose_attrs([1, 1], [0, 0], [0, 0], [1, 1], 2),
    )
    .expect_err("groups != 1 must be refused");
    let message = format!("{err}");
    assert!(
        message.contains("only groups == 1 is supported on CudaBackendImpl"),
        "refusal must name the groups limitation and the backend: {message}"
    );
    assert!(
        message.contains("groups=2"),
        "refusal must report the offending groups value: {message}"
    );
}

// ---------------------------------------------------------------------------
// adaptive_avg_pool2d
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA hardware"]
fn adaptive_avg_pool2d_evenly_dividing_matches_hand_fixture() {
    require_cuda();
    let ctx = context();
    let values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = upload_f32_shaped(&[1, 1, 4, 4], &values);
    let before = tape_depth();
    let out = dispatch::execute::<op::AdaptiveAvgPool2dExact, _>(
        &ctx,
        AdaptivePool2dAttributes { output: [2, 2] },
        &[handle(&input)],
    )
    .expect("adaptive_avg_pool2d executes");
    assert_eq!(
        tape_depth() - before,
        1,
        "adaptive_avg_pool2d must record its tape entry"
    );
    assert_eq!(out.shape.to_vec(), vec![1, 1, 2, 2]);
    assert_close(
        &download_f32(&out),
        &[3.5, 5.5, 11.5, 13.5],
        1e-5,
        "adaptive 4x4→2x2",
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn adaptive_avg_pool2d_non_evenly_dividing_produces_variable_windows() {
    require_cuda();
    let ctx = context();
    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input = upload_f32_shaped(&[1, 1, 5, 1], &values);
    let out = dispatch::execute::<op::AdaptiveAvgPool2dExact, _>(
        &ctx,
        AdaptivePool2dAttributes { output: [3, 1] },
        &[handle(&input)],
    )
    .expect("adaptive 5→3 executes");
    assert_eq!(out.shape.to_vec(), vec![1, 1, 3, 1]);
    // Windows [0,2), [1,4), [3,5) → means 1.5, 3.0, 4.5.
    assert_close(&download_f32(&out), &[1.5, 3.0, 4.5], 1e-5, "adaptive 5→3");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn adaptive_avg_pool2d_backward_spreads_sum_seed_evenly() {
    require_cuda();
    let ctx = context();
    let values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = upload_f32_shaped(&[1, 1, 4, 4], &values);
    let input_id = TapeStorage::id(&input);
    let out = dispatch::execute::<op::AdaptiveAvgPool2dExact, _>(
        &ctx,
        AdaptivePool2dAttributes { output: [2, 2] },
        &[handle(&input)],
    )
    .expect("adaptive_avg_pool2d executes");
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&out)
        .expect("adaptive_avg_pool2d backward on CUDA");
    let g = grads
        .get(input_id)
        .expect("adaptive input receives a gradient");
    assert_eq!(g.shape.to_vec(), vec![1, 1, 4, 4]);
    // Even 4→2: each input feeds exactly one 2×2 window → 1/4 under a
    // ones-seed (sum-loss) backward.
    let vals = download_f32(g);
    assert!(
        vals.iter().all(|v| (v - 0.25).abs() <= 1e-5),
        "each element must get 0.25, got {vals:?}"
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn adaptive_avg_pool2d_unbatched_rank3_backward_reaches_input() {
    require_cuda();
    let ctx = context();
    let values: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let input = upload_f32_shaped(&[1, 4, 4], &values);
    let input_id = TapeStorage::id(&input);
    let out = dispatch::execute::<op::AdaptiveAvgPool2dExact, _>(
        &ctx,
        AdaptivePool2dAttributes { output: [2, 2] },
        &[handle(&input)],
    )
    .expect("rank-3 adaptive_avg_pool2d executes");
    assert_eq!(out.shape.to_vec(), vec![1, 2, 2]);
    let grads = <TestBackend as AutogradBackend>::backward::<f32>(&out)
        .expect("rank-3 adaptive backward on CUDA");
    let g = grads
        .get(input_id)
        .expect("rank-3 adaptive input receives a gradient");
    assert_eq!(g.shape.to_vec(), vec![1, 4, 4]);
    let vals = download_f32(g);
    assert!(
        vals.iter().all(|v| (v - 0.25).abs() <= 1e-5),
        "rank-3 grad must be 0.25 everywhere, got {vals:?}"
    );
}

// ---------------------------------------------------------------------------
// to_dtype
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA hardware"]
fn to_dtype_f32_to_f64_round_trips_values() {
    require_cuda();
    let ctx = context();
    let input = upload_f32_shaped(&[4], &[1.0, -2.5, 3.75, 0.0]);
    let out = dispatch::execute::<op::ToDType, _>(
        &ctx,
        DTypeAttributes {
            dtype: DTypeId::F64.descriptor(),
        },
        &[handle(&input)],
    )
    .expect("f32→f64 to_dtype executes");
    assert_eq!(out.dtype(), DTypeId::F64.descriptor());
    let bytes = download_bytes(&out);
    let vals: &[f64] = bytemuck::cast_slice(&bytes);
    assert_eq!(vals, &[1.0, -2.5, 3.75, 0.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn to_dtype_f32_to_f16_narrows_within_half_precision() {
    require_cuda();
    let ctx = context();
    let input = upload_f32_shaped(&[3], &[1.0, -2.5, 0.5]);
    let out = dispatch::execute::<op::ToDType, _>(
        &ctx,
        DTypeAttributes {
            dtype: DTypeId::F16.descriptor(),
        },
        &[handle(&input)],
    )
    .expect("f32→f16 to_dtype executes");
    assert_eq!(out.dtype(), DTypeId::F16.descriptor());
    let bytes = download_bytes(&out);
    let bits: &[u16] = bytemuck::cast_slice(&bytes);
    let got: Vec<f32> = bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
    assert_close(&got, &[1.0, -2.5, 0.5], 1e-3, "f32→f16");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn to_dtype_refuses_q8_0_target_by_name() {
    require_cuda();
    let ctx = context();
    let input = upload_f32_shaped(&[2], &[1.0, 2.0]);
    let error = dispatch::execute::<op::ToDType, _>(
        &ctx,
        DTypeAttributes {
            dtype: DTypeId::Q8_0.descriptor(),
        },
        &[handle(&input)],
    )
    .expect_err("Q8_0 is not a to_dtype target");
    let message = format!("{error}");
    assert!(
        message.contains("to_dtype"),
        "the refusal must name the operation: {message}"
    );
}

// ---------------------------------------------------------------------------
// quantize / dequantize
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires CUDA hardware"]
fn quantize_dequantize_round_trips_within_block_tolerance() {
    require_cuda();
    let ctx = context();
    let values: Vec<f32> = (0..32).map(|index| index as f32 - 8.0).collect();
    let input = upload_f32_shaped(&[32], &values);

    let blocks = dispatch::execute::<op::Quantize, _>(
        &ctx,
        QuantizationAttributes {
            dtype: DTypeId::Q8_0.descriptor(),
        },
        &[handle(&input)],
    )
    .expect("quantize executes");
    assert_eq!(blocks.dtype(), DTypeId::Q8_0.descriptor());

    let restored = dispatch::execute::<op::Dequantize, _>(
        &ctx,
        QuantizationAttributes {
            dtype: DTypeId::F32.descriptor(),
        },
        &[handle(&blocks)],
    )
    .expect("dequantize executes");
    assert_eq!(restored.dtype(), DTypeId::F32.descriptor());
    assert_eq!(restored.shape.to_vec(), vec![32]);

    let largest = values
        .iter()
        .fold(0.0f32, |seen, &value| seen.max(value.abs()));
    let tolerance = largest / 127.0 / 2.0 + 1e-6;
    let got = download_f32(&restored);
    for (index, &original) in values.iter().enumerate() {
        let difference = (got[index] - original).abs();
        assert!(
            difference <= tolerance,
            "element {index} moved by {difference}, more than the {tolerance} \
             the block scale allows"
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn quantize_refuses_non_q8_0_target_by_name() {
    require_cuda();
    let ctx = context();
    let input = upload_f32_shaped(&[32], &[0.0; 32]);
    let error = dispatch::execute::<op::Quantize, _>(
        &ctx,
        QuantizationAttributes {
            dtype: DTypeId::F16.descriptor(),
        },
        &[handle(&input)],
    )
    .expect_err("f16 is not a quantized representation this backend produces");
    let message = format!("{error}");
    assert!(
        message.contains("quantize"),
        "the refusal must name the operation: {message}"
    );
}
