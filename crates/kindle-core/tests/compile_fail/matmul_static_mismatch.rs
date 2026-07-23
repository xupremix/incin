extern crate kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_macros::s;

fn main() {
    let t1: Tensor<s![2, 3], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![4, 5], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();

    // Inner dimensions must match: (2, 3) x (4, 5) is invalid because 3 != 4.
    let _ = t1.matmul(&t2);
}
