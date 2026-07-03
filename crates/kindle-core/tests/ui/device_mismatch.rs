use kindle_core::prelude::*;

fn main() {
    let a: Tensor<Dyn, CandleBackend, f32, Cpu, Grad> = Tensor::zeros(()).unwrap();
    let b: Tensor<Dyn, CandleBackend, f32, Cuda, Grad> = Tensor::zeros(()).unwrap();
    
    // This should fail to compile because Cpu != Cuda
    let _c = a.add(&b);
}
