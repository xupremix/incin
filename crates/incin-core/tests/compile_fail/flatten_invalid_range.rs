extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;
use incin_macros::s;

fn main() {
    let tensor = Tensor::<s![2, 3, 4], CpuBackendImpl>::zeros(()).unwrap();
    let _ = tensor.flatten::<2, 1>();
}
