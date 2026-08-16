extern crate incin_core as incin;
use incin_core::{advanced::{Here, Next}, prelude::*};
use incin_core::test_utils::DummyBackend;
use incin_macros::s;

incin_core::dim!(Batch, OtherBatch);

fn main() {
    // Concatenating along axis 1 (Feature) requires axis 0 to be the exact
    // same type on both operands — Batch and OtherBatch are different types
    // even though nothing here says they're a different runtime size.
    let a: Tensor<s![Batch, 4], DummyBackend<Cpu>> = Tensor::zeros((32usize, ())).unwrap();
    let b: Tensor<s![OtherBatch, 8], DummyBackend<Cpu>> = Tensor::zeros((32usize, ())).unwrap();
    let _ = a.concat::<s![OtherBatch, 8], Next<Here>>(&b);
}
