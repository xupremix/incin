//! Integration coverage for `main` on the documented public surface.
extern crate incin_core as incin;
use incin_core::{advanced::Here, prelude::*};
use incin_backends::cpu::CpuBackendImpl;
use incin_macros::s;

fn main() {
    let t1: Tensor<s![2, 3], CpuBackendImpl> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![4, 3], CpuBackendImpl> = Tensor::zeros(()).unwrap();

    // Trying to stack fails because typenum::U2 != typenum::U4 (they must be the exact same shape)
    let _ = t1.stack_structural::<Here>(&t2);
}
