extern crate incin_core as incin;

use incin_core::prelude::Cpu;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;
use incin_macros::s;

trait Same<T> {}
impl<T> Same<T> for T {}

fn assert_same<A, B>()
where
    A: Same<B>,
{
}

#[test]
/// Test concat static success.
fn test_concat_static_success() {
    let t1: Tensor<s![2, 3], DummyBackend<Cpu>> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![4, 3], DummyBackend<Cpu>> = Tensor::zeros(()).unwrap();

    let _out = t1.concat::<s![4, 3], Here>(&t2).unwrap();
    type Out = <s![2, 3] as ConcatShape<s![4, 3], Here>>::Output;
    type Expected =
        DimCons<<typenum::U2 as core::ops::Add<typenum::U4>>::Output, DimCons<typenum::U3, Nil>>;
    assert_same::<Out, Expected>();
}

#[test]
/// Test try concat dynamic.
fn test_try_concat_dynamic() {
    let t1: Tensor<s![dyn, 3], DummyBackend<Cpu>> = Tensor::zeros((2, ())).unwrap();
    let t2: Tensor<s![dyn, 3], DummyBackend<Cpu>> = Tensor::zeros((4, ())).unwrap();

    let out = t1.try_concat(&t2, 0).unwrap();
    assert_eq!(out.shape_buf().as_ref(), &[6, 3]);
}

#[test]
fn test_try_concat_rejects_mismatched_ranks() {
    let t1: Tensor<Dyn, DummyBackend<Cpu>> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<Dyn, DummyBackend<Cpu>> = Tensor::zeros([2]).unwrap();

    assert!(matches!(
        t1.try_concat(&t2, 0),
        Err(Error::Shape(ShapeError::RankMismatch {
            operation: OperationKind::Concat,
            expected: RankExpectation::Exactly(2),
            actual: 1,
        }))
    ));
}

#[test]
fn test_try_concat_slice_rejects_mismatched_ranks() {
    let t1: Tensor<Dyn, DummyBackend<Cpu>> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<Dyn, DummyBackend<Cpu>> = Tensor::zeros([2]).unwrap();

    assert!(matches!(
        Tensor::try_concat_slice(&[&t1, &t2], 0),
        Err(Error::Shape(ShapeError::RankMismatch {
            operation: OperationKind::Concat,
            expected: RankExpectation::Exactly(2),
            actual: 1,
        }))
    ));
}

#[test]
/// Test stack static success.
fn test_stack_static_success() {
    let t1: Tensor<s![2, 3], DummyBackend<Cpu>> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![2, 3], DummyBackend<Cpu>> = Tensor::zeros(()).unwrap();

    let _out = t1.stack::<Next<Here>>(&t2).unwrap();
    type Out = <s![2, 3] as StackShape<Next<Here>>>::Output;
    type Expected = DimCons<typenum::U2, DimCons<typenum::U2, DimCons<typenum::U3, Nil>>>;
    assert_same::<Out, Expected>();
}

#[test]
/// Test try stack dynamic.
fn test_try_stack_dynamic() {
    let t1: Tensor<Dyn, DummyBackend<Cpu>> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<Dyn, DummyBackend<Cpu>> = Tensor::zeros([2, 3]).unwrap();

    let out = t1.try_stack(&t2, 1).unwrap();
    assert_eq!(out.shape_buf().as_ref(), &[2, 2, 3]);
}
