//! Integration coverage for `main` on the documented public surface.
extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

fn main() {
    let layer = Linear::<s![3, 4], CpuBackendImpl>::build(()).unwrap();
    // 5 != 3
    let input = Tensor::<s![2, 5], CpuBackendImpl>::zeros(()).unwrap();
    layer.forward(input).unwrap();
}
