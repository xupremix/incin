extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_macros::{s, module};

#[module(foo = "bar")]
/// Auto-generated documentation for MyNet.
pub struct MyNet<B: Backend> {
    /// Auto-generated documentation for linear.
    pub linear: Linear<s![10, 10], B>,
}

fn main() {}
