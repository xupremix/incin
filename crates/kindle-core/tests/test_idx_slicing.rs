extern crate kindle_core as kindle;

use kindle_core::prelude::*;
use kindle_macros::idx;

fn assert_shape_eq<S1: Shape, S2: Shape>() {}

#[test]
fn test_idx_slicing_compile_time() {
    type S = (Const<4>, Const<4>, Const<3>);
    
    // idx![1..3, .., 0..2]
    type IdxT = idx![1..3, .., 0..2];
    
    // The output shape should be (2, 4, 2)
    assert_shape_eq::<<IdxT as kindle_core::shapes::idx::SliceTarget<S>>::Output, (Const<2>, Const<4>, Const<2>)>();
}

#[test]
fn test_idx_slicing_full() {
    type S = (Const<4>, Const<4>, Const<3>);
    
    // idx![.., .., ..]
    type IdxT = idx![.., .., ..];
    
    // The output shape should be (4, 4, 3)
    assert_shape_eq::<<IdxT as kindle_core::shapes::idx::SliceTarget<S>>::Output, (Const<4>, Const<4>, Const<3>)>();
}
