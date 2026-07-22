extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_core::nn::*;


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
