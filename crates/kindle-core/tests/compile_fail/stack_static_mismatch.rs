use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;

fn main() {
    let t1: Tensor<s![U2, U3], DummyBackend<f32, Cpu>> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<s![U4, U3], DummyBackend<f32, Cpu>> = Tensor::zeros([4, 3]).unwrap();

    // Trying to stack fails because U2 != U4 (they must be the exact same shape)
    let _ = t1.stack::<U0>(&t2);
}
