//! WGPU advertises only f32 storage. A static f64 tensor must therefore fail
//! by trait resolution rather than reaching a runtime creation method.

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::prelude::{SupportsDType, Wgpu};

fn requires_f64<B: SupportsDType<f64>>() {}

fn main() {
    requires_f64::<WgpuBackendImpl<Wgpu>>();
}
