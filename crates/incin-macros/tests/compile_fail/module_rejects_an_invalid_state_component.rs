//! State names are single durable path components, not dotted paths.
use ::incin::prelude::*;

#[module]
pub struct Bad<B: Backend> {
    #[state(name = "q_proj.weight")]
    fc: Linear<s![8, 4], B>,
}

fn main() {}
