extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;

fn main() {
    let t1: Tensor<s![2, 3], DummyBackend<f32, Cpu>> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<s![4, 3], DummyBackend<f32, Cpu>> = Tensor::zeros([4, 3]).unwrap();

    // Trying to stack fails because typenum::U2 != typenum::U4 (they must be the exact same shape)
    let _ = t1.stack::<typenum::U0>(&t2);
}
