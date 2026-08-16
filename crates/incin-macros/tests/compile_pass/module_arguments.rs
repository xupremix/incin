//! `#[module]` accepts its supported argument forms, and the struct stays usable
//! as an ordinary type after each.
use ::incin::prelude::*;
use ::incin::{
    backend_authoring::{Backend, VariableBackend},
    optim::ParameterGroup,
};

#[module]
pub struct Plain<B: Backend + VariableBackend> {
    fc: Linear<s![8, 4], B>,
}

#[module(no_stats)]
pub struct Quiet<B: Backend + VariableBackend> {
    fc: Linear<s![8, 4], B>,
}

fn main() {
    let plain = Plain::<DefaultBackend> {
        fc: Linear::build(()).unwrap(),
    };
    // The point of the attribute: parameters are aggregated from the fields.
    assert!(!ParameterGroup::<DefaultBackend, f32>::from_module(&plain)
        .unwrap()
        .is_empty());

    let quiet = Quiet::<DefaultBackend> {
        fc: Linear::build(()).unwrap(),
    };
    assert!(!ParameterGroup::<DefaultBackend, f32>::from_module(&quiet)
        .unwrap()
        .is_empty());
}
