//! Upsample module tests (backend-exercising).
//!
//! These live in integration tests rather than in-crate unit tests:
//! naming a downstream backend impl (e.g. `CpuBackendImpl<Cpu>`) inside
//! `src/` resolves the test build of `incin_core` against the normal
//! build the backend was compiled against, so the trait never resolves.
//! As a separate crate target both builds coincide and the impl applies.

use incin_backends::cpu::CpuBackendImpl;
use incin_core::error::Error;
use incin_core::nn::collect_state;
use incin_core::nn::{Module, Upsample, UpsampleMode};
use incin_core::shapes::{DimCons, Nil};
use incin_core::tensor::base::Tensor;
use incin_core::tensor::device::Cpu;
use typenum::consts::{U1, U2, U4};

type B = CpuBackendImpl<Cpu>;
type N1122 = DimCons<U1, DimCons<U1, DimCons<U2, DimCons<U2, Nil>>>>;
type N1222 = DimCons<U1, DimCons<U2, DimCons<U2, DimCons<U2, Nil>>>>;
type N112 = DimCons<U1, DimCons<U1, DimCons<U2, Nil>>>;
type N14 = DimCons<U1, DimCons<U4, Nil>>;

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
fn nearest_scale_two_matches_hand_computation() {
    let m = Upsample::nearest(2, 2).unwrap();
    let x = Tensor::<N1122, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    let out = m.forward(x).unwrap();
    assert_eq!(out.dims().as_ref(), &[1, 1, 4, 4]);
    assert_close(
        &out.to_vec1::<f32>().unwrap(),
        &[
            1.0, 1.0, 2.0, 2.0, //
            1.0, 1.0, 2.0, 2.0, //
            3.0, 3.0, 4.0, 4.0, //
            3.0, 3.0, 4.0, 4.0,
        ],
        1e-6,
    );
}

#[test]
fn size_form_matches_scale_form() {
    let scale = Upsample::nearest(2, 3).unwrap();
    let size = Upsample::nearest_size(4, 6).unwrap();
    let x = Tensor::<N1122, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    let a = scale.forward(x).unwrap().to_vec1::<f32>().unwrap();
    let x = Tensor::<N1122, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    let b = size.forward(x).unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(b.len(), 4 * 6); // [1, 1, 4, 6]
    assert_close(&a, &b, 1e-7);
}

#[test]
fn parity_with_repeat_interleave_composition() {
    let m = Upsample::nearest(2, 2).unwrap();
    let x =
        Tensor::<N1222, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], ()).unwrap();
    let got = m.forward(x).unwrap().to_vec1::<f32>().unwrap();

    let x =
        Tensor::<N1222, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], ()).unwrap();
    let expected = x
        .into_dyn()
        .repeat_interleave(2, -2)
        .unwrap()
        .repeat_interleave(2, -1)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_close(&got, &expected, 1e-7);
}

#[test]
fn rank_three_scales_width_only() {
    let m = Upsample::nearest(1, 3).unwrap();
    let x = Tensor::<N112, B>::from_slice(&[1.0f32, 2.0], ()).unwrap();
    let out = m.forward(x).unwrap();
    assert_eq!(out.dims().as_ref(), &[1, 1, 6]);
    assert_close(
        &out.to_vec1::<f32>().unwrap(),
        &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        1e-6,
    );
}

#[test]
fn bilinear_is_a_typed_refusal() {
    assert!(Upsample::new(UpsampleMode::Bilinear, 2, 2).is_err());
    assert!(Upsample::with_size(UpsampleMode::Bilinear, 4, 4).is_err());
    // A bilinear value built field-by-field still fails closed in forward.
    let m = Upsample {
        mode: UpsampleMode::Bilinear,
        scale_factor: Some([2, 2]),
        size: None,
    };
    let x = Tensor::<N1122, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    let err = m.forward(x).unwrap_err();
    assert!(matches!(err, Error::InvalidModuleState { .. }));
}

#[test]
fn bad_geometry_is_refused() {
    assert!(Upsample::nearest(0, 2).is_err());
    assert!(Upsample::nearest_size(4, 0).is_err());
    // Non-integer size multiples cannot be nearest-interpolated.
    let m = Upsample::nearest_size(3, 4).unwrap();
    let x = Tensor::<N1122, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    assert!(m.forward(x).is_err());
    // Downsampling is not an upsample.
    let m = Upsample::nearest_size(1, 1).unwrap();
    let x = Tensor::<N1122, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    assert!(m.forward(x).is_err());
    // Rank-2 inputs have no spatial axis.
    let m = Upsample::nearest(2, 2).unwrap();
    let x = Tensor::<N14, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    assert!(m.forward(x).is_err());
    // A height factor on a rank-3 input is refused, not dropped.
    let m = Upsample::nearest(2, 2).unwrap();
    let x = Tensor::<N112, B>::from_slice(&[1.0f32, 2.0], ()).unwrap();
    assert!(m.forward(x).is_err());
}

#[test]
fn stateless_layer_reports_empty_state() {
    let m = Upsample::nearest(2, 2).unwrap();
    let snap = collect_state::<B, _>(&m).unwrap();
    assert!(snap.is_empty());
}
