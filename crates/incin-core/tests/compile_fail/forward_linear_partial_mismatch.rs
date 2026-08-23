extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

fn main() {
    let layer = Linear::<s![3, 4], CpuBackendImpl>::build(()).unwrap();
    // Partially dynamic, but the end dimension is still known and wrong!
    let input = Tensor::<s![dyn, 5], CpuBackendImpl>::zeros(2).unwrap();
    layer.forward(input).unwrap();
}
