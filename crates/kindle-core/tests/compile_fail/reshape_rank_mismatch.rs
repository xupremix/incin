use kindle_core as kindle;
use kindle_core::prelude::*;
use typenum::{typenum::U2, typenum::U4, typenum::U8};

/// Assert reshape eq.
fn assert_reshape_eq<S1: Shape, S2: Shape>() where S1: kindle_core::shapes::reshape::ReshapeShape<S2> {}

fn main() {
    /// S1.
    type S1 = (typenum::U2, typenum::U8);
    /// S2.
    type S2 = (typenum::U4, typenum::U8);
    assert_reshape_eq::<S1, S2>();
}
