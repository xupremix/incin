use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_core::nn::{BatchNorm2d, Module};
use typenum::{U16, U32};

fn main() {
    let device = Cpu;
    // Layer expects 32 channels
    let layer: BatchNorm2d<U32, DummyBackend> = BatchNorm2d::new((), &device).unwrap();
    
    // Tensor has 16 channels, this should fail!
    let tensor: Tensor<(usize, U16, usize, usize), DummyBackend> = Tensor::zeros((1, 1, 1)).unwrap();
    
    let _out = layer.forward(tensor).unwrap();
}
