use core::fmt::Debug;
use core::marker::PhantomData;

use crate::prelude::{Dyn, Result};

/// `Device`.
pub trait Device: 'static + Send + Sync + Clone + Eq + PartialEq + Debug + Sized {
    /// `Arg`.
    type Arg: Clone;
    /// `Field`.
    type Field: Debug + Clone;
    /// `init`.
    fn init(arg: Self::Arg) -> Self::Field;
    /// `to_kindle`.
    fn to_kindle(dev: &Self::Field) -> Result<DeviceId>;
}
/// `ConstDevice`.
pub trait ConstDevice: Default + Device<Arg = ()> {}

#[cfg(feature = "cuda")]
/// `cuda`.
pub mod cuda {

    use super::{ConstDevice, Device, DeviceId, PhantomData, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// Implementation of `Cuda` for the respective backend..
    pub struct Cuda<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Cuda<N> {}

    impl<const N: usize> Device for Cuda<N> {
        /// `Arg`.
        type Arg = ();
        /// `Field`.
        type Field = PhantomData<Self>;

        /// `to_kindle`.
        fn to_kindle(_: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::cuda(N))
        }

        /// `init`.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "wgpu")]
/// `wgpu`.
pub mod wgpu {
    use super::{ConstDevice, Device, DeviceId, PhantomData, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// Implementation of `Wgpu` for the respective backend..
    pub struct Wgpu<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Wgpu<N> {}

    impl<const N: usize> Device for Wgpu<N> {
        /// `Arg`.
        type Arg = ();
        /// `Field`.
        type Field = PhantomData<Self>;

        /// `to_kindle`.
        fn to_kindle(_: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::wgpu(N))
        }

        /// `init`.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu::*;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// Implementation of `Cpu` for the respective backend..
pub struct Cpu;

impl ConstDevice for Cpu {}

impl Device for Cpu {
    /// `Arg`.
    type Arg = ();
    /// `Field`.
    type Field = PhantomData<Self>;

    /// `to_kindle`.
    fn to_kindle(_: &Self::Field) -> Result<DeviceId> {
        Ok(DeviceId::cpu())
    }

    /// `init`.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}

impl Device for Dyn {
    /// `Arg`.
    type Arg = DeviceId;
    /// `Field`.
    type Field = DeviceId;

    /// `to_kindle`.
    fn to_kindle(dev: &Self::Field) -> Result<DeviceId> {
        Ok(*dev)
    }

    /// `init`.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// `DeviceKind`.
pub enum DeviceKind {
    /// Implementation of `Cpu` for the respective backend..
    Cpu,
    /// Implementation of `Cuda` for the respective backend..
    Cuda,
    /// Implementation of `Wgpu` for the respective backend..
    Wgpu,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// `DeviceId`.
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

    /// `cpu`.
    pub fn cpu() -> Self {
        Self {
            kind: DeviceKind::Cpu,
            ordinal: 0,
        }
    }

    /// `cuda`.
    pub fn cuda(ord: usize) -> Self {
        Self {
            kind: DeviceKind::Cuda,
            ordinal: ord,
        }
    }

    /// `wgpu`.
    pub fn wgpu(ord: usize) -> Self {
        Self {
            kind: DeviceKind::Wgpu,
            ordinal: ord,
        }
    }
}

/// `fn`.
pub const fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}
/// `fn`.
pub const fn wgpu_is_available() -> bool {
    cfg!(feature = "wgpu")
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// Implementation of `CudaDevice` for the respective backend..
pub struct CudaDevice {
    /// `id`.
    pub id: usize,
}

impl CudaDevice {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// Implementation of `WgpuDevice` for the respective backend..
pub struct WgpuDevice {
    /// `id`.
    pub id: usize,
}

impl WgpuDevice {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;

    #[test]
    /// `test_device_variants`.
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
