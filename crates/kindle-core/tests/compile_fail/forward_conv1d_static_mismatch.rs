use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_core::nn::*;

#[derive(Clone, Default)]

fn main() {
    let layer = Conv1d::<3, 1, 0, 1, s![16, 3, 3], DummyBackend>::new().unwrap();
    // 3 != 1 (in_channels)
    let input = Tensor::<s![2, 1, 32], DummyBackend>::zeros(()).unwrap();
    layer.forward(input).unwrap();
}
