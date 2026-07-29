//! `#[module]` aggregates a struct's fields. There is nothing to aggregate on
//! a function, and the attribute must reject it rather than emit an impl for
//! a type that does not exist.
use ::incin::prelude::*;

#[module]
pub fn not_a_struct() {}

fn main() {}
