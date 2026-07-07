use core::fmt::Debug;
use core::marker::PhantomData;

use crate::prelude::{Dyn, Result};

pub trait Device: 'static + Send + Sync + Clone + Eq + PartialEq + Debug + Sized {
    type Arg: Clone;
    type Field: Debug + Clone;
    fn init(arg: Self::Arg) -> Self::Field;
    fn to_kindle(dev: &Self::Field) -> Result<KindleDevice>;
}
pub trait DynDevice: Device {}
pub trait ConstDevice: Default + Device<Arg = ()> {}

#[cfg(feature = "cuda")]
pub mod cuda {

    use super::{ConstDevice, Device, KindleDevice, PhantomData, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub struct Cuda<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Cuda<N> {}

    impl<const N: usize> Device for Cuda<N> {
        type Arg = ();
        type Field = PhantomData<Self>;

        fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
            Ok(KindleDevice::cuda(N))
        }

        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }

    impl<const N: usize> DynDevice for Cuda<N> {}
}

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "metal")]
pub mod metal {
    use super::{ConstDevice, Device, KindleDevice, PhantomData, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub struct Metal<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Metal<N> {}

    impl<const N: usize> Device for Metal<N> {
        type Arg = ();
        type Field = PhantomData<Self>;

        fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
            Ok(KindleDevice::metal(N))
        }

        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }

    impl<const N: usize> DynDevice for Metal<N> {}
}

#[cfg(feature = "metal")]
pub use metal::*;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Cpu;

impl ConstDevice for Cpu {}

impl Device for Cpu {
    type Arg = ();
    type Field = PhantomData<Self>;

    fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
        Ok(KindleDevice::cpu())
    }

    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}
impl DynDevice for Cpu {}

impl Device for Dyn {
    type Arg = KindleDevice;
    type Field = KindleDevice;

    fn to_kindle(dev: &Self::Field) -> Result<KindleDevice> {
        Ok(*dev)
    }

    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}
impl DynDevice for Dyn {}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DeviceVariant {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
    #[cfg(feature = "metal")]
    Metal(usize),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct KindleDevice(DeviceVariant);

impl KindleDevice {
    pub fn variant(&self) -> DeviceVariant {
        self.0
    }

    pub fn cpu() -> Self {
        Self(DeviceVariant::Cpu)
    }

    #[cfg(feature = "cuda")]
    pub fn cuda(ord: usize) -> Self {
        Self(DeviceVariant::Cuda(ord))
    }

    #[cfg(feature = "metal")]
    pub fn metal(ord: usize) -> Self {
        Self(DeviceVariant::Metal(ord))
    }
}

pub const fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}
pub const fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
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
