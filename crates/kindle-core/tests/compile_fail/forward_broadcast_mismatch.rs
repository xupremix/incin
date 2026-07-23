use kindle_core::prelude::dummy::DummyBackend;
use kindle_core::prelude::*;

fn main() {
    let t1: Tensor<(typenum::U1, typenum::U2), DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();
    let t2: Tensor<(typenum::U3, typenum::U3), DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();

    // Broadcast of (1, 2) and (3, 3) should fail since 2 != 3 and neither is 1!
    let _out = t1.broadcast_add(&t2).unwrap();
}
