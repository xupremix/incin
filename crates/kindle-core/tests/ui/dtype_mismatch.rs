use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;

fn main() {
    let t1: Tensor<Dyn, DummyBackend, f32> = Tensor::zeros(alloc::vec![2]).unwrap();
    let t2: Tensor<Dyn, DummyBackend, f64> = Tensor::zeros(alloc::vec![2]).unwrap();

    // Mismatched dtypes should not compile
    t1.add(&t2).unwrap();
}
