use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_core::nn::{BatchNorm2d, Module};
use typenum::{typenum::U16, typenum::U32};

fn main() {
    let device = Cpu;
    // Layer expects 32 channels
    let layer: BatchNorm2d<typenum::U32, DummyBackend<f32, Cpu>> = BatchNorm2d::new((), &device).unwrap();
    
    // Tensor has 16 channels, this should fail!
    let tensor: Tensor<(usize, typenum::U16, usize, usize), DummyBackend<f32, Cpu>> = Tensor::zeros((1, 1, 1)).unwrap();
    
    let _out = layer.forward(tensor).unwrap();
}
