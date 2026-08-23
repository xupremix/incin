//! Integration coverage for `test_rms_norm_static` on the documented public surface.
#![allow(clippy::type_complexity)]

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;
use incin_macros::s;
use typenum::consts::{U1, U3, U4, U8, U16};

type D1<C> = DimCons<C, Nil>;
type D2<A, B> = DimCons<A, DimCons<B, Nil>>;
type D6<A, B, C, D, E, F> =
    DimCons<A, DimCons<B, DimCons<C, DimCons<D, DimCons<E, DimCons<F, Nil>>>>>>;

#[test]
fn test_rms_norm_static() {
    let _t1: Tensor<s![2, 3, 4], CpuBackendImpl> = Tensor::zeros(()).unwrap();
    let norm: RMSNorm<s![4], CpuBackendImpl> = RMSNorm::build(0.001).unwrap();

    let _ = norm;
}

#[test]
fn test_nn_layers_accept_structural_shapes() {
    type B = CpuBackendImpl;

    let _: Linear<D2<U3, U4>, B> = Linear::build(()).unwrap();
    let _: Conv1d<D6<U16, U3, U3, U1, U1, U1>, B> = Conv1d::build(()).unwrap();
    let _: Conv2d<D6<U16, U3, U3, U1, U1, U1>, B> = Conv2d::build(()).unwrap();
    let _: LayerNorm<D1<U4>, B> = LayerNorm::build(1e-5).unwrap();
    let _: RMSNorm<D1<U4>, B> = RMSNorm::build(1e-5).unwrap();
    let _: BatchNorm2d<D1<U4>, B> = BatchNorm2d::build((1e-5, 0.1)).unwrap();
    let _: Embedding<D2<U8, U4>, B> = Embedding::build(()).unwrap();
}

#[test]
fn test_dropout() {
    let mut dropout: Dropout = Dropout::new(0.5);
    // test properties
    assert!(dropout.is_training);
    dropout.is_training = false;
    assert!(!dropout.is_training);
}

#[test]
fn test_structural_reduction_static() {
    let t1: Tensor<s![2, 3, 4], CpuBackendImpl> = Tensor::zeros(()).unwrap();
    let _var_all = t1.var_all(true).unwrap();
    let _std_all = t1.std_all(false).unwrap();
}

#[test]
fn test_lr_schedulers() {
    let mut linear = LinearLR::new(1.0, 0.1, 10);
    assert_eq!(linear.get_lr(), 1.0);
    linear.step();
    assert_eq!(linear.get_lr(), 0.91);
    for _ in 0..10 {
        linear.step();
    }
    assert_eq!(linear.get_lr(), 0.1);
}
