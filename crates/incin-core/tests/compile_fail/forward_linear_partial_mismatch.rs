extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;

fn main() {
    let layer = Linear::<s![3, 4], DummyBackend<f32, Cpu>>::build(()).unwrap();
    // Partially dynamic, but the end dimension is still known and wrong!
    let input = Tensor::<s![dyn, 5], DummyBackend<f32, Cpu>>::zeros(2).unwrap();
    layer.forward(input).unwrap();
}
