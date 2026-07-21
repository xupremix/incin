use kindle_core as kindle;
use kindle_core::prelude::*;
use typenum::{typenum::U2, typenum::U4, typenum::U8};

/// Core abstraction for `assert_reshape_eq` within the Kindle framework.
fn assert_reshape_eq<S1: Shape, S2: Shape>() where S1: kindle_core::shapes::reshape::ReshapeShape<S2> {}

fn main() {
    /// Core abstraction for `S1` within the Kindle framework.
    type S1 = (typenum::U2, typenum::U8);
    /// Core abstraction for `S2` within the Kindle framework.
    type S2 = (typenum::U4, typenum::U8);
    assert_reshape_eq::<S1, S2>();
}
