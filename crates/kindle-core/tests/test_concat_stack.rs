use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_macros::s;

#[test]
fn test_concat_static_success() {
    let t1: Tensor<s![U2, U3], DummyBackend> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![U4, U3], DummyBackend> = Tensor::zeros(()).unwrap();

    let _out = t1.concat::<s![U4, U3], U0>(&t2).unwrap();
    // Static shape is verified by compilation
}

#[test]
fn test_try_concat_dynamic() {
    let t1: Tensor<(usize, U3), DummyBackend> = Tensor::zeros((2,)).unwrap();
    let t2: Tensor<(usize, U3), DummyBackend> = Tensor::zeros((4,)).unwrap();

    let out = t1.try_concat(&t2, 0).unwrap();
    assert_eq!(out.shape_field().as_slice(), &[6, 3]);
}

#[test]
fn test_stack_static_success() {
    let t1: Tensor<s![U2, U3], DummyBackend> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![U2, U3], DummyBackend> = Tensor::zeros(()).unwrap();

    let _out = t1.stack::<U1>(&t2).unwrap();
    // Static shape is verified by compilation
}

#[test]
fn test_try_stack_dynamic() {
    let t1: Tensor<Dyn, DummyBackend> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<Dyn, DummyBackend> = Tensor::zeros([2, 3]).unwrap();

    let out = t1.try_stack(&t2, 1).unwrap();
    assert_eq!(out.shape_field().as_slice(), &[2, 2, 3]);
}
