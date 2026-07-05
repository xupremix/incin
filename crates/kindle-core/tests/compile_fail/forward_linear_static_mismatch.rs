use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_core::nn::*;

#[derive(Clone, Default)]

fn main() {
    let layer = Linear::<s![3, 4], DummyBackend>::new().unwrap();
    // 5 != 3
    let input = Tensor::<s![2, 5], DummyBackend>::zeros(()).unwrap();
    layer.forward(input).unwrap();
}
