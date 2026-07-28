//! Device types for the incin tensor system.
//!
//! There are three tiers of device specification, ordered from most
//! dynamic to most static:
//!
//! ## Tier 1 — Fully Runtime (`Dyn`)
//!
//! Neither the backend family nor the device ordinal is known at compile
//! time. The user passes a [`DeviceId`] at construction time and everything
//! is dispatched at runtime.
//!
//! ```text
//! Tensor::<Dyn, IncinBackend<Dyn, Dyn>>::zeros(([2, 3], DTypeId::F32, DeviceId::cuda(1)))
//! ```
//!
//! ## Tier 2 — Partial Compile-Time (`Cuda` / `Wgpu`)
//!
//! The backend family (CUDA or WGPU) is known at compile time, but the
//! specific device ordinal (which GPU to use) is provided at runtime as a
//! `usize`. Useful when you know you want CUDA but need to select the GPU
//! based on e.g. command-line flags.
//!
//! ```text
//! Tensor::<s![2, 3], IncinBackend<f32, Cuda>>::zeros(2)  // runtime ordinal 2
//! ```
//!
//! ## Tier 3 — Fully Static Selection (`CudaN<N>` / `WgpuN<N>`)
//!
//! Both the backend family and the device ordinal are encoded at the type
//! level via [`typenum`] unsigned integers. The tensor type fully describes
//! the requested logical device address. No constructor argument is required.
//! Hardware existence, driver compatibility, and capabilities are necessarily
//! validated when the program initializes that device at runtime.
//!
//! ```text
//! Tensor::<s![2, 3], IncinBackend<f32, CudaN<U1>>>::zeros(())  // always GPU 1
//! ```

use core::fmt::Debug;
use core::marker::PhantomData;

use crate::prelude::{Dyn, Result};

/// A type-level compute device.
///
/// | Type             | Backend at compile time | Ordinal at compile time | Constructor arg |
/// |------------------|------------------------|-------------------------|-----------------|
/// | `Dyn`            | ✗                      | ✗                       | `DeviceId`      |
/// | `Cuda` / `Wgpu`  | ✓                      | ✗                       | `usize`         |
/// | `CudaN<N>` / `WgpuN<N>` | ✓               | ✓                       | `()`            |
pub trait Device: 'static + Send + Sync + Clone + Eq + PartialEq + Debug + Sized {
    /// The user-facing constructor argument:
    /// - `DeviceId` for `Dyn` (fully runtime)
    /// - `usize` for `Cuda`/`Wgpu` (partial — ordinal at runtime)
    /// - `()` for `CudaN<N>`/`WgpuN<N>` (fully static)
    type Arg: Clone;
    /// The runtime-stored representation:
    /// - `DeviceId` for `Dyn`
    /// - `usize` for `Cuda`/`Wgpu`
    /// - `PhantomData<Self>` for `CudaN<N>`/`WgpuN<N>`
    type Field: Debug + Clone + Default;
    /// Converts a user-facing `Arg` into the stored `Field` representation.
    fn init(arg: Self::Arg) -> Self::Field;
    /// Resolves this device's runtime [`DeviceId`].
    fn to_incin(dev: &Self::Field) -> Result<DeviceId>;
}

/// A [`Device`] whose logical selector is **fully known at compile time** —
/// both the backend family and ordinal are encoded in the type. This does not
/// prove that matching hardware exists on the runtime host. Takes no
/// constructor argument (`Arg = ()`).
///
/// Implemented by `Cpu`, `CudaN<N: Unsigned>`, and `WgpuN<N: Unsigned>`.
pub trait ConstDevice: Default + Device<Arg = ()> {}

// ============================================================================
// Tier 1: Fully Runtime — Dyn
// ============================================================================

impl Device for Dyn {
    /// The runtime-chosen device — user passes a full [`DeviceId`].
    type Arg = DeviceId;
    /// Stored directly — `Dyn`'s whole point is deferring device choice
    /// to runtime, so `Field` is just the `DeviceId` itself.
    type Field = DeviceId;

    /// Already a `DeviceId` — returned as-is.
    fn to_incin(dev: &Self::Field) -> Result<DeviceId> {
        Ok(*dev)
    }

    /// Stores the `DeviceId` verbatim.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}

// ============================================================================
// Tier 2: Partial Compile-Time — Cuda / Wgpu (runtime ordinal)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_partial {
    use super::{Device, DeviceId, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// **Tier 2** CUDA device: backend kind known at compile time, ordinal
    /// supplied at runtime as a `usize`.
    ///
    /// Use this when you know you want CUDA but which GPU to use is
    /// determined at runtime (e.g. via a CLI flag or environment variable).
    ///
    /// For a fully static device selector use `CudaN<N>` where `N` is a
    /// [`typenum`] unsigned (e.g. `Cuda<U0>` for GPU 0).
    pub struct Cuda(pub usize);

    impl Device for Cuda {
        /// Ordinal supplied at construction time.
        type Arg = usize;
        /// Ordinal stored directly.
        type Field = usize;

        fn to_incin(dev: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::cuda(*dev))
        }

        fn init(arg: Self::Arg) -> Self::Field {
            arg
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_partial::Cuda;

#[cfg(feature = "wgpu")]
mod wgpu_partial {
    use super::{Device, DeviceId, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// **Tier 2** WGPU device: backend kind known at compile time, ordinal
    /// supplied at runtime as a `usize`.
    ///
    /// Use this when you know you want WGPU but which adapter to use is
    /// determined at runtime.
    ///
    /// For a fully static device selector use `WgpuN<N>` where `N` is a
    /// [`typenum`] unsigned (e.g. `Wgpu<U0>` for adapter 0).
    pub struct Wgpu(pub usize);

    impl Device for Wgpu {
        /// Ordinal supplied at construction time.
        type Arg = usize;
        /// Ordinal stored directly.
        type Field = usize;

        fn to_incin(dev: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::wgpu(*dev))
        }

        fn init(arg: Self::Arg) -> Self::Field {
            arg
        }
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu_partial::Wgpu;

// ============================================================================
// Tier 3: Fully Static Selection — CudaN<N> / WgpuN<N> (typenum ordinal)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_static {
    use super::{ConstDevice, Device, DeviceId, PhantomData, Result};
    use typenum::{U0, Unsigned};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// **Tier 3** CUDA device: both the backend kind *and* the device
    /// ordinal `N` are fully known at compile time via [`typenum`].
    ///
    /// This is a zero-sized type — no runtime data is stored. The default
    /// ordinal is `U0` (GPU 0).
    ///
    /// ```text
    /// // Always on GPU 0 — no runtime arg required
    /// Tensor::<s![2, 3], IncinBackend<f32, CudaN<U0>>>::zeros(())
    ///
    /// // Always on GPU 2
    /// Tensor::<s![2, 3], IncinBackend<f32, CudaN<U2>>>::zeros(())
    /// ```
    pub struct CudaN<N: Unsigned = U0>(PhantomData<N>);

    impl<N: Unsigned + 'static + Send + Sync + Clone + Eq + PartialEq + Debug> ConstDevice
        for CudaN<N>
    {
    }

    use core::fmt::Debug;

    impl<N: Unsigned + 'static + Send + Sync + Clone + Eq + PartialEq + Debug> Device for CudaN<N> {
        /// No constructor argument — the device ordinal `N` is compile-time-fixed.
        type Arg = ();
        /// Zero-sized: `N` alone identifies the device.
        type Field = PhantomData<Self>;

        /// Resolves to CUDA device ordinal `N::USIZE`.
        fn to_incin(_: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::cuda(N::USIZE))
        }

        /// No-op: nothing to convert.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_static::CudaN;

#[cfg(feature = "wgpu")]
mod wgpu_static {
    use super::{ConstDevice, Device, DeviceId, PhantomData, Result};
    use typenum::{U0, Unsigned};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// **Tier 3** WGPU device: both the backend kind *and* the adapter
    /// ordinal `N` are fully known at compile time via [`typenum`].
    ///
    /// This is a zero-sized type — no runtime data is stored. The default
    /// ordinal is `U0` (adapter 0).
    ///
    /// ```text
    /// // Always on adapter 0 — no runtime arg required
    /// Tensor::<s![2, 3], IncinBackend<f32, WgpuN<U0>>>::zeros(())
    /// ```
    pub struct WgpuN<N: Unsigned = U0>(PhantomData<N>);

    impl<N: Unsigned + 'static + Send + Sync + Clone + Eq + PartialEq + Debug> ConstDevice
        for WgpuN<N>
    {
    }

    use core::fmt::Debug;

    impl<N: Unsigned + 'static + Send + Sync + Clone + Eq + PartialEq + Debug> Device for WgpuN<N> {
        /// No constructor argument — the adapter ordinal `N` is compile-time-fixed.
        type Arg = ();
        /// Zero-sized: `N` alone identifies the device.
        type Field = PhantomData<Self>;

        /// Resolves to WGPU adapter ordinal `N::USIZE`.
        fn to_incin(_: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::wgpu(N::USIZE))
        }

        /// No-op: nothing to convert.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu_static::WgpuN;

// ============================================================================
// CPU — always fully static (there is only one CPU)
// ============================================================================

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// The CPU device. A zero-sized type — there is only one CPU, so no ordinal
/// is needed. This is a **Tier 3** (fully static) device selector.
pub struct Cpu;

impl ConstDevice for Cpu {}

impl Device for Cpu {
    /// No constructor argument needed — there is only one CPU device.
    type Arg = ();
    /// Zero-sized: there is nothing to store.
    type Field = PhantomData<Self>;

    /// Always resolves to `DeviceId::cpu()`.
    fn to_incin(_: &Self::Field) -> Result<DeviceId> {
        Ok(DeviceId::cpu())
    }

    /// No-op: nothing to convert.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}

// ============================================================================
// DeviceId and DeviceKind — runtime device identity
// ============================================================================

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// The runtime-identifiable backend family a `DeviceId` belongs to.
pub enum DeviceKind {
    /// The CPU backend family.
    Cpu,
    /// The CUDA backend family.
    Cuda,
    /// The WGPU backend family.
    Wgpu,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// A runtime device identifier: a backend family ([`DeviceKind`]) plus an
/// ordinal distinguishing multiple devices of the same family (e.g. GPU 0
/// vs. GPU 1). This is the [`Device`] trait's runtime counterpart — every
/// `Device::to_incin` resolves to one of these.
pub struct DeviceId {
    kind: DeviceKind,
    ordinal: usize,
}

impl Default for DeviceId {
    fn default() -> Self {
        Self::cpu()
    }
}

impl DeviceId {
    /// Returns the backend family.
    pub const fn kind(self) -> DeviceKind {
        self.kind
    }

    /// Returns the ordinal within the backend family.
    pub const fn ordinal(self) -> usize {
        self.ordinal
    }

    /// The single CPU device, usable in a `const` context.
    pub const CPU: Self = Self {
        kind: DeviceKind::Cpu,
        ordinal: 0,
    };

    /// The single CPU device (ordinal always 0).
    pub fn cpu() -> Self {
        Self {
            kind: DeviceKind::Cpu,
            ordinal: 0,
        }
    }

    /// A CUDA device at ordinal `ord`.
    pub fn cuda(ord: usize) -> Self {
        Self {
            kind: DeviceKind::Cuda,
            ordinal: ord,
        }
    }

    /// A WGPU device at ordinal `ord`.
    pub fn wgpu(ord: usize) -> Self {
        Self {
            kind: DeviceKind::Wgpu,
            ordinal: ord,
        }
    }
}

/// Whether this build was compiled with the `cuda` feature enabled
/// (does not check for actual CUDA hardware/drivers at runtime).
pub const fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}
/// Whether this build was compiled with the `wgpu` feature enabled
/// (does not check for an actual GPU adapter at runtime).
pub const fn wgpu_is_available() -> bool {
    cfg!(feature = "wgpu")
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// A CUDA device index, used as a hashable/orderable key (e.g. for
/// per-device kernel caches) distinct from the type-level `Cuda`/`CudaN<N>` markers.
pub struct CudaDevice {
    /// The CUDA device ordinal.
    pub id: usize,
}

impl CudaDevice {
    /// Creates a new instance with the given device ordinal.
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// A WGPU device index, used as a hashable/orderable key distinct from
/// the type-level `Wgpu`/`WgpuN<N>` markers.
pub struct WgpuDevice {
    /// The WGPU device ordinal.
    pub id: usize,
}

impl WgpuDevice {
    /// Creates a new instance with the given device ordinal.
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_variants() {
        let cpu = DeviceId::cpu();
        assert_eq!(cpu.kind(), DeviceKind::Cpu);
        assert_eq!(cpu.ordinal(), 0);

        #[cfg(feature = "cuda")]
        {
            let cuda = DeviceId::cuda(0);
            assert_eq!(cuda.kind(), DeviceKind::Cuda);
            assert_eq!(cuda.ordinal(), 0);

            let cuda2 = DeviceId::cuda(2);
            assert_eq!(cuda2.ordinal(), 2);
        }

        #[cfg(feature = "wgpu")]
        {
            let wgpu = DeviceId::wgpu(0);
            assert_eq!(wgpu.kind(), DeviceKind::Wgpu);
            assert_eq!(wgpu.ordinal(), 0);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tier2_runtime_ordinal() {
        // Tier 2: kind known at compile time, ordinal at runtime
        let field = <Cuda as Device>::init(3);
        let id = Cuda::to_incin(&field).unwrap();
        assert_eq!(id, DeviceId::cuda(3));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tier3_static_ordinal() {
        use typenum::U2;
        // Tier 3: both kind and ordinal known at compile time
        let field = <CudaN<U2> as Device>::init(());
        let id = CudaN::<U2>::to_incin(&field).unwrap();
        assert_eq!(id, DeviceId::cuda(2));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_wgpu_tier2_runtime_ordinal() {
        let field = <Wgpu as Device>::init(1);
        let id = Wgpu::to_incin(&field).unwrap();
        assert_eq!(id, DeviceId::wgpu(1));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_wgpu_tier3_static_ordinal() {
        use typenum::U0;
        let field = <WgpuN<U0> as Device>::init(());
        let id = WgpuN::<U0>::to_incin(&field).unwrap();
        assert_eq!(id, DeviceId::wgpu(0));
    }

    #[test]
    fn test_dyn_tier1_fully_runtime() {
        // Tier 1: both kind and ordinal at runtime
        let id = DeviceId::cuda(5);
        let field = <Dyn as Device>::init(id);
        assert_eq!(Dyn::to_incin(&field).unwrap(), DeviceId::cuda(5));
    }
}
