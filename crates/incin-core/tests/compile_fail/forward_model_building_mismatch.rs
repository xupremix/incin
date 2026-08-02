extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;


fn main() {
    let l1 = Linear::<s![10, 20], DummyBackend<f32, Cpu>>::build(()).unwrap();
    let l2 = Linear::<s![30, 40], DummyBackend<f32, Cpu>>::build(()).unwrap();

    let input = Tensor::<s![2, 10], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
    let x = l1.forward(input).unwrap();
    
    // Output of l1 is Tensor<[2, 20]>.
    // l2 expects an input ending in 30.
    // This forward pass will fail to compile because 20 != 30.
    let _y = l2.forward(x).unwrap();
}
