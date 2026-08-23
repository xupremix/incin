//! Integration coverage for `main` on the documented public surface.
use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::{Dyn, Tensor};

fn main() {
    let _ = Tensor::<Dyn, CpuBackendImpl>::from_parts_witnessed();
}

