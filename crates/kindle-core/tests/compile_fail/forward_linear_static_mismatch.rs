extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_core::nn::*;

#[derive(Clone, Default)]

fn main() {
    let layer = Linear::<s![3, 4], DummyBackend<f32, Cpu>>::new().unwrap();
    // 5 != 3
    let input = Tensor::<s![2, 5], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
    layer.forward(input).unwrap();
}
