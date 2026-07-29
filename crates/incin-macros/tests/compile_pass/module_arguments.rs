//! `#[module]` accepts its three argument forms, and the struct stays usable
//! as an ordinary type after each.
use ::incin::prelude::*;

#[module]
pub struct Plain<B: Backend> {
    fc: Linear<s![8, 4], B>,
}

#[module(no_stats)]
pub struct Quiet<B: Backend> {
    fc: Linear<s![8, 4], B>,
}

fn main() {
    let plain = Plain::<DefaultBackend> {
        fc: Linear::build(()).unwrap(),
    };
    // The point of the attribute: parameters are aggregated from the fields.
    assert!(!plain.parameters().is_empty());

    let quiet = Quiet::<DefaultBackend> {
        fc: Linear::build(()).unwrap(),
    };
    assert!(!quiet.parameters().is_empty());
}
