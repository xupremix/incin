extern crate incin_core as incin;
use incin_core::test_utils::DummyBackend;
use incin_core::prelude::*;
use incin_macros::s;

fn main() {
    let pred: Tensor<s![16, 10], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();
    let target: Tensor<s![16, 5], DummyBackend<u32, Cpu>> = Tensor::zeros(()).unwrap();
    
    let loss_fn = CrossEntropyLoss::new();
    
    // This should fail because target should be [Batch] (1D), but it is [16, 5] (2D).
    let _loss = loss_fn.forward(&pred, &target);
}
