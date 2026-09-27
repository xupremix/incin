//! BatchNorm1d module tests (backend-exercising).
//!
//! These live in integration tests rather than in-crate unit tests:
//! naming a downstream backend impl (e.g. `CpuBackendImpl<Cpu>`) inside
//! `src/` resolves the test build of `incin_core` against the normal
//! build the backend was compiled against, so the trait never resolves.
//! As a separate crate target both builds coincide and the impl applies.

use incin_backends::cpu::CpuBackendImpl;
use incin_core::nn::batch_norm1d::BatchNorm1dShape;
use incin_core::nn::{BatchNorm1d, Module, Param, TrainMode, Trainable};
use incin_core::nn::{StatePath, collect_state};
use incin_core::shapes::{DimCons, Nil};
use incin_core::tensor::base::Tensor;
use incin_core::tensor::device::Cpu;
use typenum::consts::{U1, U2, U3, U4};

type B = CpuBackendImpl<Cpu>;
type C2 = DimCons<U2, Nil>;
type N223 = DimCons<U2, DimCons<U2, DimCons<U3, Nil>>>;
type N12 = DimCons<U1, DimCons<U2, Nil>>;
type N4 = DimCons<U4, Nil>;
type N1222 = DimCons<U1, DimCons<U2, DimCons<U2, DimCons<U2, Nil>>>>;
type BN = BatchNorm1d<C2, B>;

fn module(affine: bool, track: bool) -> BN {
    BN::build((1e-5f32, 0.1f32, affine, track)).unwrap()
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

#[test]
fn training_forward_matches_hand_computation() {
    // [2, 2, 3] row-major: flat index i decodes to (b, c, l) with
    // c = (i / 3) % 2. Channel 0 holds [1, 2, 3, 11, 12, 13]
    // (mean 7.0), channel 1 holds [4, 5, 6, 14, 15, 16] (mean 10.0);
    // both have squared deviations [36, 25, 16, 16, 25, 36],
    // variance 154/6. Identity affine.
    let m = module(true, true);
    assert!(m.is_training);
    let x = Tensor::<N223, B>::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        (),
    )
    .unwrap();
    let out = m.forward(x).unwrap().to_vec1::<f32>().unwrap();
    let expected: Vec<f32> = [
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]
    .iter()
    .enumerate()
    .map(|(i, &v)| {
        let mean = if (i / 3) % 2 == 0 { 7.0 } else { 10.0 };
        ((v - mean) / (154.0 / 6.0 + 1e-5f64).sqrt()) as f32
    })
    .collect();
    assert_close(&out, &expected, 1e-4);
}

#[test]
fn eval_forward_uses_running_stats() {
    // Defaults (mean 0, var 1) with weight 2 and bias 1:
    // out = x * 2 + 1 applied per element.
    let mut m = module(true, true);
    type PShape = <C2 as BatchNorm1dShape>::ParamShape;
    m.weight = Some(
        Param::<PShape, B, f32, Trainable>::new_init((), incin_core::nn::init::constant(2.0))
            .unwrap(),
    );
    m.bias = Some(
        Param::<PShape, B, f32, Trainable>::new_init((), incin_core::nn::init::constant(1.0))
            .unwrap(),
    );
    m.eval();
    assert!(!m.is_training);
    let x = Tensor::<N12, B>::from_slice(&[1.0f32, 2.0], ()).unwrap();
    let out = m.forward(x).unwrap().to_vec1::<f32>().unwrap();
    let eps = 1e-5f32;
    let expected = [
        (1.0 / (1.0 + eps).sqrt()) * 2.0 + 1.0,
        (2.0 / (1.0 + eps).sqrt()) * 2.0 + 1.0,
    ];
    assert_close(&out, &expected, 1e-5);
}

#[test]
fn train_and_eval_differ_with_default_stats() {
    // Batch statistics (train) vs running statistics (eval, mean 0/var 1)
    // disagree on any non-degenerate input.
    let mut m = module(true, true);
    let x = Tensor::<N223, B>::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        (),
    )
    .unwrap();
    let train_out = m.forward(x).unwrap().to_vec1::<f32>().unwrap();
    m.eval();
    let x = Tensor::<N223, B>::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        (),
    )
    .unwrap();
    let eval_out = m.forward(x).unwrap().to_vec1::<f32>().unwrap();
    assert!(
        train_out
            .iter()
            .zip(eval_out.iter())
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "train and eval outputs should differ"
    );
    m.train();
    assert!(m.is_training);
}

#[test]
fn untracked_eval_falls_back_to_batch_stats() {
    let mut m = module(false, false);
    let snap = collect_state::<B, _>(&m).unwrap();
    assert!(
        snap.is_empty(),
        "affine=false + untracked must register nothing"
    );
    let x = Tensor::<N223, B>::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        (),
    )
    .unwrap();
    let train_out = m.forward(x).unwrap().to_vec1::<f32>().unwrap();
    m.eval();
    let x = Tensor::<N223, B>::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        (),
    )
    .unwrap();
    let eval_out = m.forward(x).unwrap().to_vec1::<f32>().unwrap();
    assert_close(&train_out, &eval_out, 1e-6);
}

#[test]
fn state_dict_contains_params_and_buffers() {
    let m = module(true, true);
    let snap = collect_state::<B, _>(&m).unwrap();
    for key in ["weight", "bias", "running_mean", "running_var"] {
        assert!(
            snap.get(&StatePath::root().try_child(key).unwrap())
                .is_some(),
            "missing state key {key}"
        );
    }

    let m = module(false, true);
    let snap = collect_state::<B, _>(&m).unwrap();
    assert!(
        snap.get(&StatePath::root().try_child("weight").unwrap())
            .is_none()
    );
    assert!(
        snap.get(&StatePath::root().try_child("bias").unwrap())
            .is_none()
    );
    assert!(
        snap.get(&StatePath::root().try_child("running_mean").unwrap())
            .is_some()
    );
    assert!(
        snap.get(&StatePath::root().try_child("running_var").unwrap())
            .is_some()
    );
}

#[test]
fn ranks_outside_two_and_three_are_refused() {
    let m = module(true, true);
    let x = Tensor::<N4, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    assert!(m.forward(x).is_err());
    let x = Tensor::<N1222, B>::from_slice(&[1.0f32; 8], ()).unwrap();
    assert!(m.forward(x).is_err());
}
