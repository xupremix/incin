extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_backends::cpu::CpuBackendImpl;
use incin_macros::{s, idx};

fn main() {
    let t = Tensor::<s![10, 20], CpuBackendImpl>::zeros(()).unwrap();
    let _ = t.slice::<idx![..., -2]>();
}
