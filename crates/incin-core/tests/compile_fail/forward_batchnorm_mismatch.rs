extern crate incin_core as incin;
use incin_core::test_utils::DummyBackend;
use incin_core::prelude::*;

fn main() {
    let layer = BatchNorm2d::<s![32], DummyBackend<Cpu>>::build((1e-5, 0.1)).unwrap();
    // The input has 16 channels, but the layer expects 32.
    let input = Tensor::<s![2, 16, 8, 8], DummyBackend<Cpu>>::zeros(()).unwrap();
    layer.forward(input).unwrap();
}
