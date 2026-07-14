extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;

fn main() {
    let t1: Tensor<s![2, 3], DummyBackend<f32, Cpu>> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<s![4, 3], DummyBackend<f32, Cpu>> = Tensor::zeros([4, 3]).unwrap();

    // Trying to concatenate along axis 1 (the typenum::U3 axis) fails because typenum::U2 != typenum::U4
    let _ = t1.concat::<s![4, 3], typenum::U1>(&t2);
}
