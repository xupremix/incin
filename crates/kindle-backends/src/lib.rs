#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;

pub use kindle_core::prelude::*;

pub mod backend_kind;
pub use backend_kind::BackendFor;
pub mod dispatch;
pub use dispatch::DispatchBackend;

/// Unified backend selected by its float element type and device.
///
/// Static devices resolve to concrete implementations. `KindleBackend<T, Dyn>`
/// resolves to [`DispatchBackend`] and selects its implementation at runtime.
pub type KindleBackend<T = f32, D = kindle_core::prelude::Cpu> = <D as BackendFor<T>>::Backend;

pub mod prelude {
    #[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda"))]
    pub use super::{BackendFor, KindleBackend};
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
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
