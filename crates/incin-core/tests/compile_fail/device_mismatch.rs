//! Two tensors that live on different devices cannot be combined. The proof
//! uses the two real backends this crate's dev-dependencies provide, so the
//! rejection is the one a user actually meets rather than one a stand-in
//! backend arranged.
//!
//! The tensors arrive as parameters rather than being constructed here. A
//! constructor call would contribute its own device-argument errors, and a
//! fixture that fails for two reasons no longer proves the one it names.

use incin_backends::cpu::CpuBackendImpl;
use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::prelude::*;

fn add_across_devices(
    a: &Tensor<Dyn, CpuBackendImpl, f32, Grad>,
    b: &Tensor<Dyn, WgpuBackendImpl<Wgpu>, f32, Grad>,
) {
    // This should fail to compile because Cpu != Wgpu.
    let _c = a.add_exact(b);
}

fn main() {}
