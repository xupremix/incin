extern crate kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_core::nn::*;

fn main() {
    let layer = Linear::<s![3, 4], DummyBackend<f32, Cpu>>::build(()).unwrap();
    // Partially dynamic, but the end dimension is still known and wrong!
    let input = Tensor::<s![dyn, 5], DummyBackend<f32, Cpu>>::zeros(2).unwrap();
    layer.forward(input).unwrap();
}
