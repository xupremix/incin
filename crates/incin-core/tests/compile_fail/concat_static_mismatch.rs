extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;
use incin_macros::s;

fn main() {
    let t1: Tensor<s![2, 3], DummyBackend<Cpu>> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![4, 3], DummyBackend<Cpu>> = Tensor::zeros(()).unwrap();

    // Trying to concatenate along axis 1 (the typenum::U3 axis) fails because typenum::U2 != typenum::U4
    let _ = t1.concat::<s![4, 3], Next<Here>>(&t2);
}
