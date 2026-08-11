extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;
use incin_macros::s;

incin_core::dim!(Batch, Seq);

fn main() {
    let t1: Tensor<s![Batch, 10], DummyBackend<Cpu>> = Tensor::zeros((32usize, ())).unwrap();
    let t2: Tensor<s![Seq, 10], DummyBackend<Cpu>> = Tensor::zeros((32usize, ())).unwrap();

    // Batch and Seq are distinct types even though both wrap a `usize` of
    // the same runtime value (32) here — this must NOT compile.
    let _ = t1.add(&t2);
}
