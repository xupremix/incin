//! BatchNorm2d module tests (backend-exercising).
//!
//! These live in integration tests rather than in-crate unit tests:
//! naming a downstream backend impl (e.g. `CpuBackendImpl<Cpu>`) inside
//! `src/` resolves the test build of `incin_core` against the normal
//! build the backend was compiled against, so the trait never resolves.
//! As a separate crate target both builds coincide and the impl applies.
//!
//! The contract under test is the one that was missing: before
//! `is_training` and `TrainMode`, `forward` passed `training: false` on
//! every call, so the layer normalized by `running_mean = 0` and
//! `running_var = 1` and was a fixed per-channel affine. BatchNorm1d has
//! always had the two modes; these pin that 2d now does too.

use incin_backends::cpu::CpuBackendImpl;
use incin_core::nn::batch_norm::BatchNormShape;
use incin_core::nn::{BatchNorm2d, Module, Param, TrainMode, Trainable};
use incin_core::shapes::{DimCons, Nil};
use incin_core::tensor::base::Tensor;
use incin_core::tensor::device::Cpu;
use typenum::consts::{U1, U2, U3};

type B = CpuBackendImpl<Cpu>;
type C2 = DimCons<U2, Nil>;
/// `[2, 2, 3, 2]`: 24 elements, two channels over a 3x2 spatial plane.
type N2332 = DimCons<U2, DimCons<U2, DimCons<U3, DimCons<U2, Nil>>>>;
/// `[1, 2, 1, 2]`: the smallest rank-4 tensor with two channels.
type N1212 = DimCons<U1, DimCons<U2, DimCons<U1, DimCons<U2, Nil>>>>;
type BN = BatchNorm2d<C2, B>;

fn module() -> BN {
    BN::build((1e-5f32, 0.1f32)).unwrap()
}

#[track_caller]
fn assert_close(got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol,
            "element {i}: got {g:.6}, expected {e:.6}"
        );
    }
}

/// 1..=24, laid out for `[2, 2, 3, 2]`. Row-major, flat index `i` decodes
/// to `(b, c, y, x)` with `c = (i / 6) % 2`, so:
///
/// - channel 0 holds `1..=6` and `13..=18`, mean 9.5;
/// - channel 1 holds `7..=12` and `19..=24`, mean 15.5;
/// - both have the deviations +-(3.5, 4.5, 5.5, 6.5, 7.5, 8.5), so a
///   population variance of `2 * (12.25 + 20.25 + 30.25 + 42.25 + 56.25
///   + 72.25) / 12 = 467 / 12`.
///
/// The two channels differ only by a constant shift, which is what makes
/// them a test of per-channel (not per-tensor) normalization.
fn ramp() -> Vec<f32> {
    (0..24).map(|i| i as f32 + 1.0).collect()
}

const CHANNEL_MEANS: [f32; 2] = [9.5, 15.5];
const CHANNEL_VAR: f32 = 467.0 / 12.0;

fn standardized() -> Vec<f32> {
    ramp()
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let mean = CHANNEL_MEANS[(i / 6) % 2];
            (v - mean) / (CHANNEL_VAR + 1e-5).sqrt()
        })
        .collect()
}

fn forward_2332(m: &BN, values: &[f32]) -> Vec<f32> {
    m.forward(Tensor::<N2332, B>::from_slice(values, ()).unwrap())
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[test]
fn a_new_layer_starts_in_training_mode() {
    assert!(
        module().is_training,
        "a built layer normalizes by batch statistics"
    );
}

#[test]
fn training_forward_matches_hand_computation() {
    // Identity affine, so the output is the standardized input.
    let m = module();
    assert!(m.is_training);
    assert_close(&forward_2332(&m, &ramp()), &standardized(), 1e-4);
}

#[test]
fn each_channel_is_normalized_independently() {
    // A constant shift of the whole tensor leaves the output unchanged
    // only if the mean is taken per channel rather than once per tensor -
    // and the two channels here carry different means, so a per-tensor
    // mean would move them apart.
    let m = module();
    let shifted: Vec<f32> = ramp().iter().map(|v| v + 100.0).collect();
    assert_close(
        &forward_2332(&m, &ramp()),
        &forward_2332(&m, &shifted),
        1e-4,
    );
}

#[test]
fn eval_forward_uses_running_stats() {
    // Defaults (mean 0, var 1) with weight 2 and bias 1: out = x * 2 + 1.
    let mut m = module();
    type PShape = <C2 as BatchNormShape>::ParamShape;
    m.weight =
        Param::<PShape, B, f32, Trainable>::new_init((), incin_core::nn::init::constant(2.0))
            .unwrap();
    m.bias = Param::<PShape, B, f32, Trainable>::new_init((), incin_core::nn::init::constant(1.0))
        .unwrap();
    m.eval();
    assert!(!m.is_training, "eval() selects the running statistics");
    let x = Tensor::<N1212, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    let out = m.forward(x).unwrap().to_vec1::<f32>().unwrap();
    let scale = (1.0f32 + 1e-5).sqrt().recip();
    let expected: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .map(|v| v * scale * 2.0 + 1.0)
        .collect();
    assert_close(&out, &expected, 1e-5);
}

#[test]
fn train_and_eval_differ_on_the_same_input() {
    // The regression this file exists for: with `training: false`
    // hardcoded, these two were identical and every "batch norm" was a
    // fixed per-channel affine.
    let mut m = module();
    let train_out = forward_2332(&m, &ramp());
    m.eval();
    let eval_out = forward_2332(&m, &ramp());
    assert!(
        train_out
            .iter()
            .zip(eval_out.iter())
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "training and evaluation outputs must differ on a non-degenerate input"
    );
    m.train();
    assert!(m.is_training, "train() restores batch statistics");
    assert_close(&forward_2332(&m, &ramp()), &train_out, 1e-6);
}

#[test]
fn freeze_and_unfreeze_keep_the_mode() {
    let m = module().freeze();
    assert!(
        m.is_training,
        "freezing parameters must not change the mode"
    );
    let mut m = m.unfreeze();
    m.eval();
    assert!(
        !m.unfreeze().is_training,
        "unfreezing parameters must not change the mode either"
    );
}
