use kindle_core::prelude::*;
use typenum::{U2, U4, U8};

fn assert_reshape_eq<S1: Shape, S2: Shape>() where S1: kindle_core::shapes::reshape::ReshapeShape<S2> {}

#[test]
fn test_advanced_reshaping_compile_time() {
    // Both sides are typenum Unsigned and have same ElementCount
    type S1 = (U2, U8);
    type S2 = (U4, U4);
    assert_reshape_eq::<S1, S2>();
}

#[test]
fn test_advanced_reshaping_different_rank() {
    type S1 = (U2, U2, U4);
    type S2 = (U4, U4);
    assert_reshape_eq::<S1, S2>();
}
