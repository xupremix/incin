//! Arbitrary-rank checks for the canonical recursive shape algebra.

#![recursion_limit = "512"]

extern crate incin_core as incin;

use incin_core::prelude::*;
use incin_core::shapes::shape_ops::SwapAxes;
use incin_macros::s;

trait Same<T> {}
impl<T> Same<T> for T {}

fn assert_same<A, B>()
where
    A: Same<B>,
    B: Same<A>,
{
}

#[test]
fn structural_shapes_have_no_frontend_rank_ladder() {
    type R8 = s![1; 8];
    type R16 = s![1; 16];
    type R64 = s![1; 64];
    type R200 = s![1; 200];
    assert_eq!(<R8 as Shape>::RANK, Some(8));
    assert_eq!(<R16 as Shape>::RANK, Some(16));
    assert_eq!(<R64 as Shape>::RANK, Some(64));
    assert_eq!(<R200 as Shape>::RANK, Some(200));
}

#[test]
fn structural_static_numel_is_folded_without_generated_ranks() {
    type S = s![2, 3, 4, 5];
    assert_eq!(<S as Shape>::STATIC_NUMEL, Some(120));
}

#[test]
fn structural_broadcast_and_matmul_retain_output_types() {
    type L = s![1, 3, 1, 8];
    type R = s![2, 3, 4, 8];
    type B = <L as BroadcastShape<R>>::Output;
    type ExpectedB = s![2, 3, 4, 8];
    assert_same::<B, ExpectedB>();

    type M1 = s![2, 3, 4];
    type M2 = s![2, 4, 5];
    type MOut = <M1 as MatMulShape<M2>>::Output;
    type ExpectedM = s![2, 3, 5];
    assert_same::<MOut, ExpectedM>();
}

#[test]
fn high_rank_structural_operations_are_exercised_without_rank_ladders() {
    type R16 = s![1; 16];
    type R64 = s![1; 64];
    type R200 = s![1; 200];

    type R16Transpose = <R16 as SwapAxes<Here, Next<Here>>>::Output;
    type R64Reduced = <R64 as ReduceAt<Here>>::Output;

    assert_same::<R16Transpose, R16>();
    assert_same::<R64Reduced, s![1; 63]>();

    let dims16 = ShapeBuf::from_slice(&[1; 16]);
    let dims64 = ShapeBuf::from_slice(&[1; 64]);
    let dims200 = ShapeBuf::from_slice(&[1; 200]);
    let _ = <R16 as Shape>::validate_dims(dims16.as_ref());
    let _ = <R64 as Shape>::validate_dims(dims64.as_ref());
    let transposed = <R16 as SwapAxes<Here, Next<Here>>>::swap_shape(&dims16)
        .expect("rank-16 transpose validates its selectors");
    assert_eq!(transposed.as_ref(), &[1; 16]);
    let reduced = <R64 as ReduceAt<Here>>::reduce_shape(&dims64)
        .expect("rank-64 reduction validates its selector");
    assert_eq!(reduced.as_ref(), &[1; 63]);
    let broadcast = <R200 as BroadcastShape<R200>>::output_shape(&dims200, &dims200)
        .expect("identical rank-200 shapes broadcast");

    assert_eq!(broadcast.as_ref(), &[1; 200]);
}
