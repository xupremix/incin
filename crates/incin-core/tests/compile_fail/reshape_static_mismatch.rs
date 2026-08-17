extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_backends::cpu::CpuBackendImpl;
use incin_macros::s;

fn main() {
    let t = Tensor::<s![2, 3], CpuBackendImpl>::zeros(()).unwrap();
    
    let _ = t.reshape(shape![2, 4]);
}
