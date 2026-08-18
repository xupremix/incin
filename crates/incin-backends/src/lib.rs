#![cfg_attr(not(feature = "std"), no_std)]
#[macro_use]
extern crate alloc;

pub mod backend_kind;
pub mod capability;
/// The generated support tables in `docs/capabilities.md` (`UX-013`).
#[cfg(feature = "std")]
pub mod capability_docs;
pub mod codegen;
#[macro_use]
#[cfg(any(
    feature = "cpu",
    feature = "wgpu",
    feature = "cuda",
    feature = "metal",
    feature = "external-candle"
))]
pub(crate) mod descriptor_bind;
pub mod dispatch;
mod dispatch_capability;
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

// Originally just the two backends that allocate device buffers by byte
// length. The external Candle adapter was in this list and never called
// `byte_len` — Candle owns its own allocations — so `--features
// external-candle` without a GPU backend compiled a module whose only
// function was dead, which `-D warnings` rejects. `unsupported` below keeps
// `external-candle`, because the Candle adapter does use those macros.
// `checked_numel` widened the gate to `cpu`: it is the one implementation
// `cpu::stride` and `iteration` both call, and a CPU-only build needs it too.
#[cfg(any(feature = "cpu", feature = "cuda", feature = "wgpu"))]
pub(crate) mod bytes;
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

pub(crate) mod layout;

#[cfg(any(feature = "cpu", feature = "cuda"))]
pub(crate) mod quant;

/// Compile-time SIMD lane-width resolution for type-specialized kernels.
///
/// [`simd_lanes`] returns the number of elements of type `T` that fit into one
/// SIMD register on the compile target.  It is a pure `const fn` — no runtime
/// feature detection, no overhead.
pub mod simd;
pub use simd::simd_lanes;

/// Test-only deterministic backend fault injection.
#[cfg(all(feature = "test-utils", feature = "cpu"))]
#[doc(hidden)]
pub mod test_utils {
    pub use crate::cpu::var::{AssignFailureGuard, fail_assign_on};
}

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

pub type EngineBackend<E, D> = crate::target::EngineBackend<E, D>;
pub type NativeBackend<D> = crate::target::NativeBackend<D>;

/// Unified backend selected by device.
///
/// Static devices resolve to concrete implementations. `IncinBackend<Dyn>`
/// resolves to [`DispatchBackend`] and selects its implementation at runtime.
pub type IncinBackend<D = incin_core::tensor::device::Cpu> = NativeBackend<D>;

pub mod nn_target;
pub mod target;

pub mod prelude {
    #[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda"))]
    pub use super::IncinBackend;
    #[cfg(feature = "std")]
    pub use super::detect::{detect_device, detect_device_in};
    pub use super::{EngineBackend, NativeBackend};

    // Extension methods only resolve when their trait is in scope, so the
    // traits are exported alongside the types they operate on.
    pub use super::nn_target::InitOnTarget;
    #[cfg(feature = "external-candle")]
    pub use super::target::Candle;
    pub use super::target::{
        DtypeTarget, EngineOn, EngineSpec, GeneratedFill, Native, PrecisionSpec, RuntimeEngine,
        Target, TargetExt, TensorData, TensorTarget, precision,
    };
    pub use incin_core::shapes::ShapeSpec;
}

#[cfg(feature = "cpu")]
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
