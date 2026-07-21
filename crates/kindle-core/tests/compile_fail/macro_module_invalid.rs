extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_macros::{s, module};

#[module(foo = "bar")]
/// My net.
pub struct MyNet<B: Backend> {
    /// Linear.
    pub linear: Linear<s![10, 10], B>,
}

fn main() {}
