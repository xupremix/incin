extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;

fn main() {
    let layer = Conv2d::<s![16, 3, 3, 1, 1, 1], DummyBackend<Cpu>>::build(()).unwrap();
    // Input has 4 channels, but the layer expects 3. Spatial size (32) is
    // large enough that this is purely a channel mismatch, not the separate
    // arithmetic-underflow case a too-small/too-tight kernel would cause
    // (see conv2d_invalid_shape.rs for that one).
    let input = Tensor::<s![2, 4, 32, 32], DummyBackend<Cpu>>::zeros(()).unwrap();
    layer.forward(input).unwrap();
}
