use core::fmt::Debug;
use core::marker::PhantomData;

use crate::prelude::{Dyn, Result};

/// Auto-generated documentation for Device.
pub trait Device: 'static + Send + Sync + Clone + Eq + PartialEq + Debug + Sized {
    /// Auto-generated documentation for Arg.
    type Arg: Clone;
    /// Auto-generated documentation for Field.
    type Field: Debug + Clone;
    /// Auto-generated documentation for init.
    fn init(arg: Self::Arg) -> Self::Field;
    /// Auto-generated documentation for to_kindle.
    fn to_kindle(dev: &Self::Field) -> Result<KindleDevice>;
}
/// Auto-generated documentation for DynDevice.
pub trait DynDevice: Device {}
/// Auto-generated documentation for ConstDevice.
pub trait ConstDevice: Default + Device<Arg = ()> {}

#[cfg(feature = "cuda")]
/// Auto-generated documentation for cuda.
pub mod cuda {

    use super::{ConstDevice, Device, DynDevice, KindleDevice, PhantomData, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// Auto-generated documentation for Cuda.
    pub struct Cuda<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Cuda<N> {}

    impl<const N: usize> Device for Cuda<N> {
        /// Auto-generated documentation for Arg.
        type Arg = ();
        /// Auto-generated documentation for Field.
        type Field = PhantomData<Self>;

        /// Auto-generated documentation for to_kindle.
        fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
            Ok(KindleDevice::cuda(N))
        }

        /// Auto-generated documentation for init.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }

    impl<const N: usize> DynDevice for Cuda<N> {}
}

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "metal")]
/// Auto-generated documentation for metal.
pub mod metal {
    use super::{ConstDevice, Device, DynDevice, KindleDevice, PhantomData, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// Auto-generated documentation for Metal.
    pub struct Metal<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Metal<N> {}

    impl<const N: usize> Device for Metal<N> {
        /// Auto-generated documentation for Arg.
        type Arg = ();
        /// Auto-generated documentation for Field.
        type Field = PhantomData<Self>;

        /// Auto-generated documentation for to_kindle.
        fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
            Ok(KindleDevice::metal(N))
        }

        /// Auto-generated documentation for init.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }

    impl<const N: usize> DynDevice for Metal<N> {}
}

#[cfg(feature = "metal")]
pub use metal::*;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// Auto-generated documentation for Cpu.
pub struct Cpu;

impl ConstDevice for Cpu {}

impl Device for Cpu {
    /// Auto-generated documentation for Arg.
    type Arg = ();
    /// Auto-generated documentation for Field.
    type Field = PhantomData<Self>;

    /// Auto-generated documentation for to_kindle.
    fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
        Ok(KindleDevice::cpu())
    }

    /// Auto-generated documentation for init.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}
impl DynDevice for Cpu {}

impl Device for Dyn {
    /// Auto-generated documentation for Arg.
    type Arg = KindleDevice;
    /// Auto-generated documentation for Field.
    type Field = KindleDevice;

    /// Auto-generated documentation for to_kindle.
    fn to_kindle(dev: &Self::Field) -> Result<KindleDevice> {
        Ok(*dev)
    }

    /// Auto-generated documentation for init.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}
impl DynDevice for Dyn {}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Auto-generated documentation for DeviceVariant.
pub enum DeviceVariant {
    /// Auto-generated documentation for Cpu.
    Cpu,
    #[cfg(feature = "cuda")]
    /// Auto-generated documentation for Cuda.
    Cuda(usize),
    #[cfg(feature = "metal")]
    /// Auto-generated documentation for Metal.
    Metal(usize),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Auto-generated documentation for KindleDevice.
pub struct KindleDevice(DeviceVariant);

impl KindleDevice {
    /// Auto-generated documentation for variant.
    pub fn variant(&self) -> DeviceVariant {
        self.0
    }

    /// Auto-generated documentation for cpu.
    pub fn cpu() -> Self {
        Self(DeviceVariant::Cpu)
    }

    #[cfg(feature = "cuda")]
    /// Auto-generated documentation for cuda.
    pub fn cuda(ord: usize) -> Self {
        Self(DeviceVariant::Cuda(ord))
    }

    #[cfg(feature = "metal")]
    /// Auto-generated documentation for metal.
    pub fn metal(ord: usize) -> Self {
        Self(DeviceVariant::Metal(ord))
    }
}

/// Auto-generated documentation for fn.
pub const fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}
/// Auto-generated documentation for fn.
pub const fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// Auto-generated documentation for CudaDevice.
pub struct CudaDevice {
    /// Auto-generated documentation for id.
    pub id: usize,
}

impl CudaDevice {
    /// Auto-generated documentation for new.
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// Auto-generated documentation for MetalDevice.
pub struct MetalDevice {
    /// Auto-generated documentation for id.
    pub id: usize,
}

impl MetalDevice {
    /// Auto-generated documentation for new.
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

#[cfg(test)]
/// Auto-generated documentation for tests.
mod tests {
    use super::*;

    #[test]
    /// Auto-generated documentation for test_device_variants.
    fn test_device_variants() {
        let cpu = KindleDevice::cpu();
        assert_eq!(cpu.variant(), DeviceVariant::Cpu);

        #[cfg(feature = "cuda")]
        {
            let cuda = KindleDevice::cuda(0);
            assert_eq!(cuda.variant(), DeviceVariant::Cuda(0));
        }

        #[cfg(feature = "metal")]
        {
            let metal = KindleDevice::metal(0);
            assert_eq!(metal.variant(), DeviceVariant::Metal(0));
        }
    }
}
