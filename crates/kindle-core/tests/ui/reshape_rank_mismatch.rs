extern crate kindle_core as kindle;
use kindle_core::prelude::*;
use typenum::{U2, U4, U8};

fn assert_reshape_eq<S1: Shape, S2: Shape>() where S1: kindle_core::shapes::reshape::ReshapeShape<S2> {}

fn main() {
    type S1 = (U2, U8);
    type S2 = (U4, U8);
    assert_reshape_eq::<S1, S2>();
}
