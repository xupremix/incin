use kindle_core as kindle;
use kindle_core::prelude::*;
use typenum::{typenum::U2, typenum::U4, typenum::U8};

/// Auto-generated documentation for assert_reshape_eq.
fn assert_reshape_eq<S1: Shape, S2: Shape>() where S1: kindle_core::shapes::reshape::ReshapeShape<S2> {}

fn main() {
    /// Auto-generated documentation for S1.
    type S1 = (typenum::U2, typenum::U8);
    /// Auto-generated documentation for S2.
    type S2 = (typenum::U4, typenum::U8);
    assert_reshape_eq::<S1, S2>();
}
