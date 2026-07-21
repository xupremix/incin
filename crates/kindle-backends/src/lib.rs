#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;

pub use kindle_core::prelude::*;

#[cfg(feature = "cpu")]
/// Unified backend alias. Defaults to CpuBackend when `cpu` feature is active.
pub type KindleBackend<T = f32, D = kindle_core::prelude::Cpu> = cpu::CpuBackend<T, D>;

#[cfg(all(feature = "wgpu", not(feature = "cpu")))]
/// Unified backend alias. Defaults to WgpuBackend when `wgpu` feature is active without `cpu`.
pub type KindleBackend<T = f32, D = kindle_core::prelude::Wgpu> = wgpu::WgpuBackend<T, D>;

#[cfg(all(feature = "cuda", not(feature = "cpu"), not(feature = "wgpu")))]
/// Unified backend alias. Defaults to CudaBackend when `cuda` feature is active without `cpu` or `wgpu`.
pub type KindleBackend<T = f32, D = kindle_core::prelude::Cuda> = cuda::CudaBackend<T, D>;

pub mod prelude {
    #[cfg(feature = "cpu")]
    pub use super::cpu::CpuBackend;

    #[cfg(feature = "cuda")]
    #[allow(unused_imports)]
    pub use super::cuda::*;

    #[cfg(feature = "wgpu")]
    pub use super::wgpu::WgpuBackend;

    #[cfg(feature = "legacy")]
    pub use super::legacy::candle::CandleBackend;

    #[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda"))]
    pub use super::KindleBackend;
}

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "legacy")]
pub mod legacy;

#[cfg(feature = "telemetry")]
pub mod telemetry;

#[cfg(feature = "telemetry")]
pub use telemetry::set_emitter;
