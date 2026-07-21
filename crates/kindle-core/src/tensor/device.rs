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
    fn to_kindle(dev: &Self::Field) -> Result<KindleDevice>;
}
/// `DynDevice`.
pub trait DynDevice: Device {}
/// `ConstDevice`.
pub trait ConstDevice: Default + Device<Arg = ()> {}

#[cfg(feature = "cuda")]
/// `cuda`.
pub mod cuda {

    use super::{ConstDevice, Device, DynDevice, KindleDevice, PhantomData, Result};

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
        fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
            Ok(KindleDevice::cuda(N))
        }

        /// `init`.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }

    impl<const N: usize> DynDevice for Cuda<N> {}
}

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "wgpu")]
/// `wgpu`.
pub mod wgpu {
    use super::{ConstDevice, Device, DynDevice, KindleDevice, PhantomData, Result};

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
        fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
            Ok(KindleDevice::wgpu(N))
        }

        /// `init`.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }

    impl<const N: usize> DynDevice for Wgpu<N> {}
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
    fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
        Ok(KindleDevice::cpu())
    }

    /// `init`.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}
impl DynDevice for Cpu {}

impl Device for Dyn {
    /// `Arg`.
    type Arg = KindleDevice;
    /// `Field`.
    type Field = KindleDevice;

    /// `to_kindle`.
    fn to_kindle(dev: &Self::Field) -> Result<KindleDevice> {
        Ok(*dev)
    }

    /// `init`.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}
impl DynDevice for Dyn {}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// `DeviceVariant`.
pub enum DeviceVariant {
    /// Implementation of `Cpu` for the respective backend..
    Cpu,
    #[cfg(feature = "cuda")]
    /// Implementation of `Cuda` for the respective backend..
    Cuda(usize),
    #[cfg(feature = "wgpu")]
    /// Implementation of `Wgpu` for the respective backend..
    Wgpu(usize),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// `KindleDevice`.
pub struct KindleDevice(DeviceVariant);

impl KindleDevice {
    /// `variant`.
    pub fn variant(&self) -> DeviceVariant {
        self.0
    }

    /// `cpu`.
    pub fn cpu() -> Self {
        Self(DeviceVariant::Cpu)
    }

    #[cfg(feature = "cuda")]
    /// `cuda`.
    pub fn cuda(ord: usize) -> Self {
        Self(DeviceVariant::Cuda(ord))
    }

    #[cfg(feature = "wgpu")]
    /// `wgpu`.
    pub fn wgpu(ord: usize) -> Self {
        Self(DeviceVariant::Wgpu(ord))
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
        let cpu = KindleDevice::cpu();
        assert_eq!(cpu.variant(), DeviceVariant::Cpu);

        #[cfg(feature = "cuda")]
        {
            let cuda = KindleDevice::cuda(0);
            assert_eq!(cuda.variant(), DeviceVariant::Cuda(0));
        }

        #[cfg(feature = "wgpu")]
        {
            let wgpu = KindleDevice::wgpu(0);
            assert_eq!(wgpu.variant(), DeviceVariant::Wgpu(0));
        }
    }
}
