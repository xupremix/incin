//! Matmul checks for the canonical structural shape representation.

extern crate incin_core as incin;

use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;
use incin_macros::s;

incin_core::dim!(Batch, Contract, Features, Time);

trait Same<T> {}
impl<T> Same<T> for T {}

fn assert_same<A, B>()
where
    A: Same<B>,
    B: Same<A>,
{
}

fn field<S: Shape>(dims: &[usize]) -> ShapeBuf {
    S::try_from_dims(dims).expect("test dimensions must match the shape type")
}

#[test]
fn arbitrary_batch_rank_retains_matrix_output_type() {
    type L = s![1; 8];
    type R = s![1; 8];
    type Out = <L as MatMulShape<R>>::Output;
    type Expected = s![1; 8];
    assert_same::<Out, Expected>();
}

#[test]
fn structural_matmul_checks_runtime_contraction() {
    type L = DimCons<usize, DimCons<usize, Nil>>;
    type R = DimCons<usize, DimCons<usize, Nil>>;
    let err = <L as MatMulShape<R>>::output_shape(&field::<L>(&[2, 3]), &field::<R>(&[4, 5]))
        .unwrap_err();
    assert_eq!(err.axis(), Some(Axis::Named("k")));
}

#[test]
fn structural_matmul_accepts_matching_runtime_contraction() {
    type L = s![2, dyn, 4];
    type R = s![2, dyn, 5];
    type Out = <L as MatMulShape<R>>::Output;
    type Expected = s![2, dyn, 5];
    assert_same::<Out, Expected>();
    let out = <L as MatMulShape<R>>::output_shape(&field::<L>(&[2, 3, 4]), &field::<R>(&[2, 4, 5]))
        .unwrap();
    assert_eq!(out.as_ref(), &[2, 3, 5]);
}

#[test]
fn static_matmul_output_type_preserves_batch_and_matrix_axes() {
    type L = s![2, 3, 4];
    type R = s![2, 4, 5];
    type Out = <L as MatMulShape<R>>::Output;
    type Expected = s![2, 3, 5];

    assert_same::<Out, Expected>();
}

#[test]
fn anonymous_and_named_equal_static_contractions_are_compatible() {
    type L = s![2, 64];
    type R = s![Contract: 64, 5];
    type Out = <L as MatMulShape<R>>::Output;
    type Expected = s![2, 5];

    assert_same::<Out, Expected>();
    let out =
        <L as MatMulShape<R>>::output_shape(&field::<L>(&[2, 64]), &field::<R>(&[64, 5])).unwrap();
    assert_eq!(out.as_ref(), &[2, 5]);
}

#[test]
fn matching_named_static_contractions_preserve_the_matrix_result() {
    type L = s![2, Contract: 64];
    type R = s![Contract: 64, 5];
    type Out = <L as MatMulShape<R>>::Output;
    type Expected = s![2, 5];

    assert_same::<Out, Expected>();
}

#[test]
fn named_static_and_runtime_contractions_use_numeric_validation() {
    type StaticLhs = s![3, Features: 64];
    type StaticRhs = s![Features: 64, 5];
    type RuntimeLhs = s![3, Features: dyn];
    type RuntimeRhs = s![Features: dyn, 5];
    type ReverseRuntimeLhs = s![3, Features: 64];
    type ReverseRuntimeRhs = s![Features: dyn, 5];
    type Expected = s![3, 5];

    assert_same::<<StaticLhs as MatMulShape<StaticRhs>>::Output, Expected>();
    assert_same::<<RuntimeLhs as MatMulShape<RuntimeRhs>>::Output, Expected>();
    assert_same::<<ReverseRuntimeLhs as MatMulShape<ReverseRuntimeRhs>>::Output, Expected>();

    let lhs = field::<RuntimeLhs>(&[3, 64]);
    let rhs = field::<RuntimeRhs>(&[64, 5]);
    assert_eq!(
        <RuntimeLhs as MatMulShape<RuntimeRhs>>::output_shape(&lhs, &rhs)
            .unwrap()
            .as_ref(),
        &[3, 5]
    );

    let mismatched_rhs = field::<RuntimeRhs>(&[63, 5]);
    assert!(matches!(
        <RuntimeLhs as MatMulShape<RuntimeRhs>>::output_shape(&lhs, &mismatched_rhs),
        Err(ShapeError::DimensionMismatch {
            operation: OperationKind::MatMul,
            axis: Axis::Named("k"),
            ..
        })
    ));
}

#[test]
fn named_mixed_contractions_preserve_the_known_output_extent() {
    type NamedRuntimeLhs = s![3, Features: dyn];
    type NamedStaticRhs = s![Features: 64, 5];
    type StaticNamedLhs = s![3, Features: 64];
    type NamedRuntimeRhs = s![Features: dyn, 5];
    type AnonymousStaticLhs = s![3, 64];

    assert_same::<<NamedRuntimeLhs as MatMulShape<NamedStaticRhs>>::Output, s![3, 5]>();
    assert_same::<<StaticNamedLhs as MatMulShape<NamedRuntimeRhs>>::Output, s![3, 5]>();
    assert_same::<<AnonymousStaticLhs as MatMulShape<NamedRuntimeRhs>>::Output, s![3, 5]>();

    let lhs = field::<NamedRuntimeLhs>(&[3, 64]);
    let rhs = field::<NamedStaticRhs>(&[64, 5]);
    assert_eq!(
        <NamedRuntimeLhs as MatMulShape<NamedStaticRhs>>::output_shape(&lhs, &rhs)
            .unwrap()
            .as_ref(),
        &[3, 5]
    );
}

#[test]
fn named_batch_broadcast_preserves_name_and_static_extent() {
    type L = s![Batch: 25, 3, Features: 64];
    type R = s![Batch: 1, Features: 64, 5];
    type Out = <L as MatMulShape<R>>::Output;
    type Expected = s![Batch: 25, 3, 5];

    assert_same::<Out, Expected>();
    let lhs = field::<L>(&[25, 3, 64]);
    let rhs = field::<R>(&[1, 64, 5]);
    assert_eq!(
        <L as MatMulShape<R>>::output_shape(&lhs, &rhs)
            .unwrap()
            .as_ref(),
        &[25, 3, 5]
    );
}

#[test]
fn bmm_preserves_named_static_output_type() {
    type B = DummyBackend<Cpu>;
    type L = s![Batch: 25, 3, Features: 64];
    type R = s![Batch: 1, Features: 64, 5];
    type Expected = s![Batch: 25, 3, 5];

    let lhs: Tensor<L, B> = Tensor::ones(()).unwrap();
    let rhs: Tensor<R, B> = Tensor::ones(()).unwrap();
    let output = lhs.bmm(&rhs).unwrap();

    assert_same::<_, Tensor<Expected, B>>();
    assert_eq!(output.shape_buf().as_ref(), &[25, 3, 5]);
}
