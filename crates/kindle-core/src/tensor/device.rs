use core::fmt::Debug;
use core::marker::PhantomData;

use crate::prelude::{Dyn, Result};

/// A type-level compute device (`Cpu`, `Cuda<N>`, `Wgpu<N>`, or `Dyn` for
/// runtime-selected devices). Paired with a float element type via
/// `BackendFor<T>` to select a concrete backend.
pub trait Device: 'static + Send + Sync + Clone + Eq + PartialEq + Debug + Sized {
    /// The user-facing constructor argument type (`()` for fixed devices,
    /// `DeviceId` for `Dyn`).
    type Arg: Clone;
    /// The runtime-stored representation (a `PhantomData` for fixed
    /// devices, `DeviceId` for `Dyn`).
    type Field: Debug + Clone;
    /// Converts a user-facing `Arg` into the stored `Field` representation.
    fn init(arg: Self::Arg) -> Self::Field;
    /// Resolves this device's runtime `DeviceId`.
    fn to_kindle(dev: &Self::Field) -> Result<DeviceId>;
}
/// A `Device` whose identity is fully known at compile time (as opposed
/// to `Dyn`, which is resolved at runtime) — takes no constructor argument.
pub trait ConstDevice: Default + Device<Arg = ()> {}

#[cfg(feature = "cuda")]
/// The CUDA device type and its `Device` impl.
pub mod cuda {

    use super::{ConstDevice, Device, DeviceId, PhantomData, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// The CUDA device, indexed by `N` for multi-GPU setups (defaults to
    /// device 0). A zero-sized type — the index lives entirely at the type level.
    pub struct Cuda<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Cuda<N> {}

    impl<const N: usize> Device for Cuda<N> {
        /// No constructor argument — the device index `N` is compile-time-fixed.
        type Arg = ();
        /// Zero-sized: `N` alone identifies the device.
        type Field = PhantomData<Self>;

        /// Resolves to CUDA device ordinal `N`.
        fn to_kindle(_: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::cuda(N))
        }

        /// No-op: nothing to convert.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "wgpu")]
/// The WGPU device type and its `Device` impl.
pub mod wgpu {
    use super::{ConstDevice, Device, DeviceId, PhantomData, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// The WGPU device, indexed by `N` for multi-adapter setups (defaults
    /// to adapter 0). A zero-sized type — the index lives entirely at the type level.
    pub struct Wgpu<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Wgpu<N> {}

    impl<const N: usize> Device for Wgpu<N> {
        /// No constructor argument — the device index `N` is compile-time-fixed.
        type Arg = ();
        /// Zero-sized: `N` alone identifies the device.
        type Field = PhantomData<Self>;

        /// Resolves to WGPU device ordinal `N`.
        fn to_kindle(_: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::wgpu(N))
        }

        /// No-op: nothing to convert.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu::*;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// The CPU device. A zero-sized type — there is only one CPU.
pub struct Cpu;

impl ConstDevice for Cpu {}

impl Device for Cpu {
    /// No constructor argument needed — there is only one CPU device.
    type Arg = ();
    /// Zero-sized: there is nothing to store.
    type Field = PhantomData<Self>;

    /// Always resolves to `DeviceId::cpu()`.
    fn to_kindle(_: &Self::Field) -> Result<DeviceId> {
        Ok(DeviceId::cpu())
    }

    /// No-op: nothing to convert.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}

impl Device for Dyn {
    /// The runtime-chosen device.
    type Arg = DeviceId;
    /// Stored directly — `Dyn`'s whole point is deferring device choice
    /// to runtime, so `Field` is just the `DeviceId` itself.
    type Field = DeviceId;

    /// Already a `DeviceId` — returned as-is.
    fn to_kindle(dev: &Self::Field) -> Result<DeviceId> {
        Ok(*dev)
    }

    /// Stores the `DeviceId` verbatim.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}

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
/// A runtime device identifier: a backend family (`DeviceKind`) plus an
/// ordinal distinguishing multiple devices of the same family (e.g. GPU 0
/// vs. GPU 1). This is the `Device` trait's runtime counterpart — every
/// `Device::to_kindle` resolves to one of these.
pub struct DeviceId {
    kind: DeviceKind,
    ordinal: usize,
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
/// per-device kernel caches) distinct from the type-level `Cuda<N>` marker.
pub struct CudaDevice {
    /// The CUDA device ordinal.
    pub id: usize,
}

impl CudaDevice {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// A WGPU device index, used as a hashable/orderable key distinct from
/// the type-level `Wgpu<N>` marker.
pub struct WgpuDevice {
    /// The WGPU device ordinal.
    pub id: usize,
}

impl WgpuDevice {
    /// Creates a new instance with default (statically inferred) shape arguments.
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

        #[cfg(feature = "cuda")]
        {
            let cuda = DeviceId::cuda(0);
            assert_eq!(cuda.kind(), DeviceKind::Cuda);
            assert_eq!(cuda.ordinal(), 0);
        }

        #[cfg(feature = "wgpu")]
        {
            let wgpu = DeviceId::wgpu(0);
            assert_eq!(wgpu.kind(), DeviceKind::Wgpu);
            assert_eq!(wgpu.ordinal(), 0);
        }
    }
}
