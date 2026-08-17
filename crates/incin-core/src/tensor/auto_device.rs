//! Automatic device selection, in the two places it can honestly happen.
//!
//! There are exactly two questions, and they have different answers at
//! different times:
//!
//! 1. **Which backend was this binary built to target?** Answered at compile
//!    time by `BestDevice` / [`best_device!`](crate::best_device), from the
//!    enabled Cargo features, in the order CUDA → WGPU → CPU.
//! 2. **Which backend can actually run here?** Answered at runtime by probing
//!    real hardware. That lives in `incin-backends::detect_device`, returns a
//!    [`DeviceId`](crate::tensor::device::DeviceId), and is used with the fully
//!    dynamic tier.
//!
//! # Why the compile-time half does not probe hardware
//!
//! `PROPOSALS.md` §1.2.2 fixes the meaning of a static device: *"'Static
//! device' means a compile-time logical selector, not compile-time hardware
//! discovery."* The Macro policy section adds that macros are *"inappropriate
//! for hardware discovery"*, and decision `D-006` records that device
//! existence is never provable before runtime, which is why every `Device`
//! constructor is fallible.
//!
//! Those rules exist because the build host and the run host are routinely
//! different machines:
//!
//! * `docker build` does not expose the GPU even on a GPU host — `--gpus` is a
//!   *run* flag. A probing macro would compile every containerized deployment
//!   to CPU, which is the most common way an ML binary ships.
//! * Cross-compilation probes the wrong machine by construction.
//! * A GPU-less CI runner would silently produce a binary with *different
//!   types* than a developer's machine, so a type error appears only in CI.
//! * Proc-macro expansion is cached and has no `rerun-if-env-changed`
//!   equivalent, so a probe result can outlive the hardware that produced it.
//!
//! Selecting from features is deterministic, reproducible, and cross-compile
//! safe; probing at runtime is correct about the machine that actually
//! executes. Between them there is no case left for probing at build time.

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
use crate::tensor::device::Cpu;
#[cfg(feature = "cuda")]
use crate::tensor::device::CudaN;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use crate::tensor::device::WgpuN;

/// The most capable device family this build can target, at ordinal 0.
///
/// Resolved from the features enabled on `incin-core` itself, in the order
/// CUDA → WGPU → CPU. `incin-backends` forwards its own `cuda`/`wgpu` features
/// here, so enabling `incin-backends/cuda` is what makes this `CudaN<U0>`.
///
/// This names a *logical selector*. It does not assert that the hardware
/// exists; that is checked when the device is initialized, and is why
/// `Device::to_incin` returns a `Result`.
///
/// ```rust
/// # extern crate incin_core as incin;
/// use incin::prelude::*;
/// // Naming the selector and building its stored field costs nothing and
/// // cannot fail, whichever family the features picked.
/// let field = <BestDevice as Device>::init(());
/// // Resolving it is where the hardware question is finally asked, which is
/// // why this returns a `Result` instead of a `DeviceId`.
/// let resolved = <BestDevice as Device>::to_incin(&field);
/// # let _ = resolved;
/// ```
#[cfg(feature = "cuda")]
pub type BestDevice = CudaN<typenum::U0>;

/// The most capable device family this build can target, at ordinal 0.
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type BestDevice = WgpuN<typenum::U0>;

/// The most capable device family this build can target, at ordinal 0.
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type BestDevice = Cpu;

/// [`BestDevice`] at a caller-chosen ordinal.
///
/// `N` selects the device ordinal for the GPU families. The CPU has no
/// ordinal, so when this build resolves to [`Cpu`] the parameter is accepted
/// and ignored rather than failing to compile — code written for a GPU build
/// still compiles on a CPU-only one, which is the whole point of the alias.
#[cfg(feature = "cuda")]
pub type BestDeviceAt<N> = CudaN<N>;

/// [`BestDevice`] at a caller-chosen ordinal.
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type BestDeviceAt<N> = WgpuN<N>;

/// [`BestDevice`] at a caller-chosen ordinal, which the CPU ignores.
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type BestDeviceAt<N> = CpuAt<N>;

/// [`Cpu`] carrying an ignored ordinal parameter.
///
/// Exists only so [`BestDeviceAt`] has the same arity in every build. A bare
/// `type BestDeviceAt<N> = Cpu;` would also compile, but this spells out that
/// the ordinal is deliberately discarded rather than accidentally dropped.
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub type CpuAt<N> = <(Cpu, core::marker::PhantomData<N>) as IgnoreOrdinal>::Device;

/// Projects an ordinal-carrying pair back to its device.
#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub trait IgnoreOrdinal {
    /// The device, with the ordinal discarded.
    type Device;
}

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
impl<N> IgnoreOrdinal for (Cpu, core::marker::PhantomData<N>) {
    type Device = Cpu;
}

/// Names the best device this build can target.
///
/// * `best_device!()` — ordinal 0, i.e. [`BestDevice`].
/// * `best_device!(U2)` — ordinal 2 via a `typenum` type, i.e.
///   [`BestDeviceAt<U2>`](BestDeviceAt). Ignored on a CPU-only build.
///
/// This is a naming convenience over the two aliases and nothing more. It
/// expands to a public type path, performs no discovery, and touches neither
/// the filesystem nor the network — the conditions `PROPOSALS.md` puts on every
/// public macro. Resolving the `cfg` inside this crate rather than inside the
/// macro body is load bearing: a `#[cfg(feature = "cuda")]` written in a
/// `macro_rules!` body is evaluated against the *calling* crate's features, so
/// it would read as disabled in every downstream crate and silently select CPU.
///
/// ```rust
/// type Dev = incin_core::best_device!();
/// type Second = incin_core::best_device!(incin_core::typenum::U1);
/// ```
#[macro_export]
macro_rules! best_device {
    () => { $crate::tensor::auto_device::BestDevice };
    ($ordinal:ty) => { $crate::tensor::auto_device::BestDeviceAt<$ordinal> };
}
