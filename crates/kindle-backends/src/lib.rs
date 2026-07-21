#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;

pub use kindle_core::prelude::*;

pub mod prelude {
    #[cfg(feature = "cpu")]
    pub use super::cpu::CpuBackend;
    
    #[cfg(feature = "cuda")]
    pub use super::cuda::*;

    #[cfg(feature = "wgpu")]
    pub use super::wgpu::WgpuBackend;

    #[cfg(feature = "legacy")]
    pub use super::legacy::candle::CandleBackend;
}

#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "legacy")]
pub mod legacy;
