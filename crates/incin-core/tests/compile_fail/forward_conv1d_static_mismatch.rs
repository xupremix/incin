extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;

fn main() {
    let layer = Conv1d::<s![16, 3, 3, 1, 1, 1], DummyBackend<f32, Cpu>>::build(()).unwrap();
    // Input has 4 channels, but the layer expects 3. Length (32) is large
    // enough that this is purely a channel mismatch, not the separate
    // arithmetic-underflow case a too-small/too-tight kernel would cause.
    let input = Tensor::<s![2, 4, 32], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
    layer.forward(input).unwrap();
}
