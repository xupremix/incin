extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_macros::s;
use incin_backends::cpu::CpuBackendImpl;

/// Backend.
type Backend = CpuBackendImpl;

fn main() {
    let t = Tensor::<s![1, 3, 2, 2], Backend>::zeros(()).unwrap();
    let w = Tensor::<s![8, 3, 3, 3], Backend>::zeros(()).unwrap();
    
    // Kernel larger than image with no padding causes negative shape -> compile error!
    let _c = t.conv2d::<typenum::U1, typenum::U0, s![8, 3, 3, 3]>(&w, None).unwrap();
}
