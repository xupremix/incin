//! Integration coverage for `main` on the documented public surface.
use incin_core::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

fn main() {
    let t1: Tensor<Dyn, CpuBackendImpl, f32> = Tensor::zeros(vec![2]).unwrap();
    let t2: Tensor<Dyn, CpuBackendImpl, f64> = Tensor::zeros(vec![2]).unwrap();

    // Mismatched dtypes should not compile
    t1.add_exact(&t2).unwrap();
}
