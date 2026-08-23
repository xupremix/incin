//! Integration coverage for `main` on the documented public surface.
extern crate incin_core as incin;

use incin::prelude::*;

type B = incin_backends::cpu::CpuBackendImpl;

fn main() {
    // Static broadcast incompatibility is rejected at the `+` operator
    // boundary, rather than becoming a runtime panic.
    let lhs = Tensor::<s![32], B>::zeros(()).unwrap();
    let rhs = Tensor::<s![64], B>::zeros(()).unwrap();
    let _ = lhs + rhs;
}
