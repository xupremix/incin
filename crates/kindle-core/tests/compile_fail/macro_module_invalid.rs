use kindle_core::prelude::*;
use kindle_macros::{s, module};

#[module(foo = "bar")]
pub struct MyNet<B: Backend> {
    pub linear: Linear<s![10, 10], B>,
}

fn main() {}
