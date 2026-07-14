extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_macros::s;

fn main() {
    let pred: Tensor<s![16, 10], DummyBackend<f32, Cpu>> = Tensor::static_zeros().unwrap();
    let target: Tensor<s![16, 5], DummyBackend<u32, Cpu>> = Tensor::static_zeros().unwrap();
    
    let loss_fn = CrossEntropyLoss::new();
    
    // This should fail because target should be [Batch] (1D), but it is [16, 5] (2D).
    let _loss = loss_fn.forward(&pred, &target);
}
