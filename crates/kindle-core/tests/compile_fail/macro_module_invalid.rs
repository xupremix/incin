extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_macros::{s, module};

#[module(foo = "bar")]
/// Core abstraction for `MyNet` within the Kindle framework.
pub struct MyNet<B: Backend> {
    /// Core abstraction for `linear` within the Kindle framework.
    pub linear: Linear<s![10, 10], B>,
}

fn main() {}
