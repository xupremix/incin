extern crate kindle_core as kindle;

use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_core::prelude::Cpu;
use kindle_macros::s;

#[test]
fn test_concat_static_success() {
    let t1: Tensor<s![2, 3], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![4, 3], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();

    let _out = t1.concat::<s![4, 3], typenum::U0>(&t2).unwrap();
    // Static shape is verified by compilation
}

#[test]
fn test_try_concat_dynamic() {
    let t1: Tensor<(usize, typenum::U3), DummyBackend<f32, Cpu>> = Tensor::zeros((2,)).unwrap();
    let t2: Tensor<(usize, typenum::U3), DummyBackend<f32, Cpu>> = Tensor::zeros((4,)).unwrap();

    let out = t1.try_concat(&t2, 0).unwrap();
    assert_eq!(out.shape_field().as_slice(), &[6, 3]);
}

#[test]
fn test_stack_static_success() {
    let t1: Tensor<s![2, 3], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![2, 3], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();

    let _out = t1.stack::<typenum::U1>(&t2).unwrap();
    // Static shape is verified by compilation
}

#[test]
fn test_try_stack_dynamic() {
    let t1: Tensor<Dyn, DummyBackend<f32, Cpu>> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<Dyn, DummyBackend<f32, Cpu>> = Tensor::zeros([2, 3]).unwrap();

    let out = t1.try_stack(&t2, 1).unwrap();
    assert_eq!(out.shape_field().as_slice(), &[2, 2, 3]);
}
