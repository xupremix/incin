extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;

fn main() {
    let layer = Linear::<s![3, 4], DummyBackend<Cpu>>::build(()).unwrap();
    // 5 != 3
    let input = Tensor::<s![2, 5], DummyBackend<Cpu>>::zeros(()).unwrap();
    layer.forward(input).unwrap();
}
