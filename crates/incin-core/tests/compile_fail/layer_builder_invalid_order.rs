extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;

fn main() {
    type B = CpuBackendImpl;
    let _ = Linear::<Dyn, B, Dyn>::build((true, 10, 20));
}
