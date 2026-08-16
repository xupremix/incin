use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;

fn main() {
    let t1: Tensor<Dyn, DummyBackend<Cpu>, f32> = Tensor::zeros(vec![2]).unwrap();
    let t2: Tensor<Dyn, DummyBackend<Cpu>, f64> = Tensor::zeros(vec![2]).unwrap();

    // Mismatched dtypes should not compile
    t1.add_exact(&t2).unwrap();
}
