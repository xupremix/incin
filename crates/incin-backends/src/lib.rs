#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;

pub use incin_core::prelude::*;

pub mod backend_kind;
pub use backend_kind::BackendFor;
pub mod dispatch;
pub use dispatch::DispatchBackend;

/// Runtime detection of the best device this machine can actually run on.
#[cfg(feature = "std")]
pub mod detect;
#[cfg(feature = "std")]
pub use detect::{detect_device, detect_device_in};

pub(crate) mod dtype_policy;

#[cfg(any(feature = "cpu", feature = "cuda"))]
pub(crate) mod iteration;
#[cfg(feature = "cuda")]
pub(crate) mod kernel;
#[cfg(feature = "cuda")]
pub(crate) mod tuning;

/// Unified backend selected by its float element type and device.
///
/// Static devices resolve to concrete implementations. `IncinBackend<T, Dyn>`
/// resolves to [`DispatchBackend`] and selects its implementation at runtime.
pub type IncinBackend<T = f32, D = incin_core::prelude::Cpu> = <D as BackendFor<T>>::Backend;

pub mod prelude {
    #[cfg(feature = "std")]
    pub use super::detect::{detect_device, detect_device_in};
    #[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda"))]
    pub use super::{BackendFor, IncinBackend};
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "external-candle")]
/// Third-party backend integrations that are separate from native Incin backends.
pub mod external;

#[cfg(feature = "telemetry")]
pub mod telemetry;

#[cfg(feature = "telemetry")]
pub use telemetry::set_emitter;
