//! Matmul checks for the canonical structural shape representation.

extern crate incin_core as incin;

use incin_core::prelude::*;
use incin_macros::s;

incin_core::dim!(Batch, Contract);

trait Same<T> {}
impl<T> Same<T> for T {}

fn assert_same<A, B>()
where
    A: Same<B>,
{
}

fn field<S: Shape>(dims: &[usize]) -> ShapeBuf {
    S::from_dyn(dims).expect("test dimensions must match the shape type")
}

#[test]
fn arbitrary_batch_rank_retains_matrix_output_type() {
    type L = s![1; 8];
    type R = s![1; 8];
    type Out = <L as MatMulShape<R>>::Output;
    type E = BroadcastExtent<typenum::U1, typenum::U1>;
    type Expected = DimCons<
        E,
        DimCons<
            E,
            DimCons<
                E,
                DimCons<E, DimCons<E, DimCons<E, DimCons<typenum::U1, DimCons<typenum::U1, Nil>>>>>,
            >,
        >,
    >;
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
    type Expected = DimCons<
        BroadcastExtent<typenum::U2, typenum::U2>,
        DimCons<usize, DimCons<typenum::U5, Nil>>,
    >;
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
    type Expected = DimCons<
        BroadcastExtent<typenum::U2, typenum::U2>,
        DimCons<typenum::U3, DimCons<typenum::U5, Nil>>,
    >;

    trait Same<T> {}
    impl<T> Same<T> for T {}
    fn assert_same<A, B>()
    where
        A: Same<B>,
    {
    }
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
