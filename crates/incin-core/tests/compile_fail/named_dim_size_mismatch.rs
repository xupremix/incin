extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;
use incin_macros::s;

incin_core::dim!(Batch);

fn main() {
    let t1: Tensor<s![Batch, 10], DummyBackend<f32, Cpu>> = Tensor::zeros((32usize, ())).unwrap();
    let t2: Tensor<s![Batch, 20], DummyBackend<f32, Cpu>> = Tensor::zeros((32usize, ())).unwrap();

    // Same dim name (Batch) but a different literal size on the other axis
    // (10 vs 20) — must NOT compile.
    let _ = t1.add(&t2);
}
