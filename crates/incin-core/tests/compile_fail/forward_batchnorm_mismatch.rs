extern crate incin_core as incin;
use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;

fn main() {
    let layer = BatchNorm2d::<s![32], CpuBackendImpl>::build((1e-5, 0.1)).unwrap();
    // The input has 16 channels, but the layer expects 32.
    let input = Tensor::<s![2, 16, 8, 8], CpuBackendImpl>::zeros(()).unwrap();
    layer.forward(input).unwrap();
}
