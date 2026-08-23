//! Integration coverage for `main` on the documented public surface.
use incin_core::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

type Shape23 = DimCons<typenum::U2, DimCons<typenum::U3, Nil>>;
type Shape234 = DimCons<typenum::U2, DimCons<typenum::U3, DimCons<typenum::U4, Nil>>>;

fn main() {
    let t1: Tensor<Shape23, CpuBackendImpl> = Tensor::zeros(()).unwrap();
    let t2: Tensor<Shape234, CpuBackendImpl> = Tensor::zeros(()).unwrap();

    // Mismatched shapes should not compile
    t1.add_exact(&t2).unwrap();
}
