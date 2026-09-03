//! Distribution sampling on the real CPU backend.
//!
//! These tests previously ran against the shape-only stand-in backend, where
//! every sampler returned a shape and no values, so nothing about the
//! distributions themselves could be asserted. Against a real backend the
//! sampled values exist, so each test checks the support of the distribution
//! it names in addition to the shape it produces.

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::backend_authoring::{Backend, SupportsDType};
use incin_core::distributions::{
    Bernoulli, Distribution, DistributionExecutor, Exponential, Gumbel, Normal,
    TensorDistributionExt, Uniform,
};
use incin_core::prelude::*;
use incin_core::shapes::{Dyn, ShapeBuf};

type B = CpuBackendImpl;

/// Every sampler under test produces `f32`, and `HostReadback` is the only
/// way to see the values a shape-only backend never had.
fn values<S: Shape + DynShape, L: incin::shapes::Layout>(
    tensor: &Tensor<S, B, f32, incin::prelude::NoGrad, incin::dist::Local, L>,
) -> Vec<f64> {
    tensor
        .to_vec1::<f32>()
        .expect("cpu storage reads back to host")
        .into_iter()
        .map(f64::from)
        .collect()
}

#[test]
fn uniform_samples_land_inside_the_requested_half_open_range() {
    let u = Uniform::new(-5.0, 5.0);
    let t = Tensor::<Dyn, B>::sample(&u, vec![10, 10]).unwrap();
    assert_eq!(t.dims(), vec![10, 10]);

    let drawn = values(&t);
    assert_eq!(drawn.len(), 100);
    assert!(
        drawn.iter().all(|v| (-5.0..5.0).contains(v)),
        "uniform(-5, 5) produced a value outside its own support: {drawn:?}"
    );
}

#[test]
fn normal_samples_are_finite_and_centred_near_the_requested_mean() {
    let n = Normal::new(10.0, 2.0);
    let t = Tensor::<Dyn, B>::sample(&n, vec![64, 64]).unwrap();
    assert_eq!(t.dims(), vec![64, 64]);

    let drawn = values(&t);
    assert!(
        drawn.iter().all(|v| v.is_finite()),
        "normal produced NaN/inf"
    );

    // 4096 draws; the standard error of the mean is 2/64 = 0.031, so a
    // tolerance of 0.5 is over sixteen standard errors and cannot flake,
    // while still catching a sampler that ignores its mean entirely.
    let mean = drawn.iter().sum::<f64>() / drawn.len() as f64;
    assert!(
        (mean - 10.0).abs() < 0.5,
        "normal(10, 2) sample mean was {mean}, which is not near its mean"
    );
}

#[test]
fn bernoulli_samples_are_only_ever_zero_or_one() {
    let b_dist = Bernoulli::new(0.5);
    let t = Tensor::<Dyn, B>::sample(&b_dist, vec![100]).unwrap();
    assert_eq!(t.dims(), vec![100]);

    let drawn = values(&t);
    assert!(
        drawn.iter().all(|v| *v == 0.0 || *v == 1.0),
        "bernoulli produced a value that is neither 0 nor 1: {drawn:?}"
    );
}

#[test]
fn exponential_samples_are_non_negative() {
    let exp_dist = Exponential::new(1.5);
    let t = Tensor::<Dyn, B>::sample(&exp_dist, vec![20]).unwrap();
    assert_eq!(t.dims(), vec![20]);

    let drawn = values(&t);
    assert!(
        drawn.iter().all(|v| *v >= 0.0 && v.is_finite()),
        "exponential produced a negative or non-finite value: {drawn:?}"
    );
}

#[test]
fn gumbel_samples_are_finite() {
    let g_dist = Gumbel::new(0.0, 1.0);
    let t = Tensor::<Dyn, B>::sample(&g_dist, vec![5, 5]).unwrap();
    assert_eq!(t.dims(), vec![5, 5]);

    let drawn = values(&t);
    assert_eq!(drawn.len(), 25);
    assert!(
        drawn.iter().all(|v| v.is_finite()),
        "gumbel produced a non-finite value: {drawn:?}"
    );
}

#[test]
fn a_downstream_distribution_composes_catalog_operations() {
    // A distribution defined outside this crate's own set, written the way a
    // downstream author must write one. The orphan rule forbids a blanket
    // `impl<Bk> DistributionExecutor<ConstantAdd, f32> for Bk` from outside
    // the crate that owns the trait, so the executor is implemented for one
    // concrete backend - which is exactly the shape of the extension a
    // downstream crate can actually write. The offset is what proves the
    // composition ran: a uniform draw on [0, 1) shifted by 42 must land in
    // [42, 43), which no unshifted sampler could produce.
    struct ConstantAdd(f32);

    impl DistributionExecutor<ConstantAdd, f32> for B {
        fn sample_distribution<S: Shape + DynShape, G: RequiresGrad>(
            distribution: &ConstantAdd,
            shape: ShapeBuf,
            device: &<Self::Device as Device>::Field,
        ) -> Result<Tensor<S, Self, f32, G, incin::dist::Local, incin::shapes::RowMajor<S>>>
        where
            (S, f32, Self::Device, G): TensorArgs<S, f32, Self::Device, G>,
        {
            let base: Tensor<S, Self, f32, G, incin::dist::Local, incin::shapes::RowMajor<S>> =
                Uniform::new(0.0, 1.0).sample::<S, Self, G>(shape, device)?;
            base.add_scalar(distribution.0 as f64)
        }
    }

    impl Distribution<f32> for ConstantAdd {
        fn sample<
            S: Shape + DynShape,
            Bk: Backend + SupportsDType<f32> + DistributionExecutor<Self, f32>,
            G: RequiresGrad,
        >(
            &self,
            shape: ShapeBuf,
            device: &<Bk::Device as Device>::Field,
        ) -> Result<Tensor<S, Bk, f32, G, incin::dist::Local, incin::shapes::RowMajor<S>>>
        where
            (S, f32, Bk::Device, G): TensorArgs<S, f32, Bk::Device, G>,
        {
            Bk::sample_distribution::<S, G>(self, shape, device)
        }
    }

    let custom = ConstantAdd(42.0);
    let t = Tensor::<Dyn, B>::sample(&custom, vec![3, 3]).unwrap();
    assert_eq!(t.dims(), vec![3, 3]);

    let drawn = values(&t);
    assert!(
        drawn.iter().all(|v| (42.0..43.0).contains(v)),
        "the +42 offset did not reach the sampled tensor: {drawn:?}"
    );
}

#[test]
fn a_fully_static_shape_needs_no_runtime_arguments() {
    use typenum::{U2, U3};

    // When shape, device, dtype, and grad are all compile-time known, the
    // argument tuple collapses to `()`.
    let dist = Normal::new(0.0, 1.0);
    type Static23 =
        incin_core::shapes::DimCons<U2, incin_core::shapes::DimCons<U3, incin_core::shapes::Nil>>;

    let t = Tensor::<Static23, B>::sample(&dist, ()).unwrap();
    assert_eq!(t.dims().as_ref(), &[2, 3]);
    assert_eq!(values(&t).len(), 6);
}
