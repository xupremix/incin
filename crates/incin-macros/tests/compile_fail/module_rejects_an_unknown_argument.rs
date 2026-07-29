//! `#[module]`'s grammar is versioned and closed: an argument it does not know
//! is a typo, and expanding as though it were absent hides it.
use ::incin::prelude::*;

#[module(no_such_argument)]
pub struct Bad<B: Backend> {
    fc: Linear<s![8, 4], B>,
}

fn main() {}
