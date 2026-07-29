//! `s![dim; count]` needs its count, and the count must be an integer the
//! macro can evaluate at expansion time.
use ::incin::prelude::*;

type Bad = s![4; ];

fn main() {}
