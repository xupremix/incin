extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::prelude::dummy::DummyBackend;
use incin_macros::s;

fn main() {
    let t1: Tensor<s![2, 3], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![4, 3], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();

    // Trying to stack fails because typenum::U2 != typenum::U4 (they must be the exact same shape)
    let _ = t1.stack::<typenum::U0>(&t2);
}
