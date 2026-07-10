use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_core::nn::*;

#[derive(Clone, Default)]

fn main() {
    let layer = Linear::<s![3, 4], DummyBackend<f32, Cpu>>::new().unwrap();
    // Partially dynamic, but the end dimension is still known and wrong!
    let input = Tensor::<s![dyn, 5], DummyBackend<f32, Cpu>>::zeros([2, 5]).unwrap();
    layer.forward(input).unwrap();
}
