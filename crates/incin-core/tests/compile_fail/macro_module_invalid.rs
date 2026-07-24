extern crate incin_core as incin;
use incin_core as incin;
use incin_core::prelude::*;
use incin_macros::{s, module};

#[module(foo = "bar")]
/// My net.
pub struct MyNet<B: Backend> {
    /// Linear.
    pub linear: Linear<s![10, 10], B>,
}

fn main() {}
