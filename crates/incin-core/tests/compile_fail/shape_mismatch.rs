use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;

type Shape23 = DimCons<typenum::U2, DimCons<typenum::U3, Nil>>;
type Shape234 = DimCons<typenum::U2, DimCons<typenum::U3, DimCons<typenum::U4, Nil>>>;

fn main() {
    let t1: Tensor<Shape23, DummyBackend<Cpu>> = Tensor::zeros(()).unwrap();
    let t2: Tensor<Shape234, DummyBackend<Cpu>> = Tensor::zeros(()).unwrap();

    // Mismatched shapes should not compile
    t1.add(&t2).unwrap();
}
