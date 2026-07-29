//! `s!` parses a dimension as a literal, `dyn`, `_`, or a path. A string
//! literal is none of those, and the parser must say so at the token rather
//! than expand to something that fails later as a type error.
use ::incin::prelude::*;

type Bad = s!["two", 3];

fn main() {}
