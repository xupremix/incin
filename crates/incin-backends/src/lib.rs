#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate alloc;


pub mod backend_kind;
pub mod capability;
/// The generated support tables in `docs/capabilities.md` (`UX-013`).
#[cfg(feature = "std")]
pub mod capability_docs;
pub mod codegen;
pub use backend_kind::BackendFor;
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda"))]
pub(crate) mod descriptor_bind;
pub mod dispatch;
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda"))]
mod dispatch_executor;
pub use dispatch::DispatchBackend;

/// Collective transport contracts and optional implementations.
#[cfg(feature = "distributed")]
pub mod dist;

/// Runtime detection of the best device this machine can actually run on.
#[cfg(feature = "std")]
pub mod detect;
#[cfg(feature = "std")]
pub use detect::{detect_device, detect_device_in};

// Only the two backends that allocate device buffers by byte length. The
// external Candle adapter was in this list and never called `byte_len` — Candle
// owns its own allocations — so `--features external-candle` without a GPU
// backend compiled a module whose only function was dead, which
// `-D warnings` rejects. `unsupported` below keeps `external-candle`, because
// the Candle adapter does use those macros.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) mod bytes;
pub(crate) mod dtype_policy;
// Every caller of these macros is a GPU or external backend, so a CPU-only
// build declared four macros it could not use and warned about all of them.
// Gated the same way `bytes` above it already is.
#[cfg(any(
    feature = "cuda",
    feature = "wgpu",
    feature = "external-candle",
    feature = "metal"
))]
pub(crate) mod unsupported;

#[cfg(any(feature = "cpu", feature = "cuda"))]
pub mod iteration;

/// Compile-time SIMD lane-width resolution for type-specialized kernels.
///
/// [`simd_lanes`] returns the number of elements of type `T` that fit into one
/// SIMD register on the compile target.  It is a pure `const fn` — no runtime
/// feature detection, no overhead.
pub mod simd;
pub use simd::simd_lanes;

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "wgpu",
    feature = "autotune"
))]
pub(crate) mod kernel;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "wgpu",
    feature = "autotune"
))]
pub mod tuning;

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

#[cfg(feature = "metal")]
pub mod metal;

/// Third-party backend integrations, and the conformance suite an author of one
/// runs against their own backend.
///
/// Unconditional since `EXE-010`. The module was gated on `external-candle`,
/// which made the backend-authoring surface reachable only by enabling one
/// particular integration; the Candle adapter inside it keeps that gate.
pub mod external;

#[cfg(feature = "telemetry")]
pub mod telemetry;

#[cfg(feature = "telemetry")]
pub use telemetry::set_emitter;
