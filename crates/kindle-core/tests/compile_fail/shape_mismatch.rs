use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;

fn main() {
    let t1: Tensor<[usize; 2], DummyBackend<f32, Cpu>> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<[usize; 3], DummyBackend<f32, Cpu>> = Tensor::zeros([2, 3, 4]).unwrap();

    // Mismatched shapes should not compile
    t1.add(&t2).unwrap();
}
