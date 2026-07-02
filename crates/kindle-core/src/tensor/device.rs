use core::fmt::Debug;
use core::marker::PhantomData;

use crate::{
    candle,
    prelude::{Dyn, Result},
};

pub trait Device: 'static + Send + Sync + Clone + Eq + PartialEq + Debug + Sized {
    type Arg;
    type Field: Debug + Clone;
    type Device;
    fn init(arg: Self::Arg) -> Self::Field;
    fn device(dev: &Self::Field) -> Result<Self::Device>;
}
pub trait DynDevice: Device {}
pub trait ConstDevice: Default + Device<Arg = ()> {}

#[cfg(feature = "cuda")]
pub mod cuda {

    use super::{ConstDevice, Device, PhantomData, Result, candle};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub struct Cuda<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Cuda<N> {}

    impl<const N: usize> Device for Cuda<N> {
        type Arg = ();
        type Field = PhantomData<Self>;
        type Device = candle::Device;

        fn device(_: &Self::Field) -> Result<Self::Device> {
            Ok(candle::Device::new_cuda(N)?)
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
    use super::{ConstDevice, Device, PhantomData, Result, candle};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub struct Metal<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Metal<N> {}

    impl<const N: usize> Device for Metal<N> {
        type Arg = ();
        type Field = PhantomData<Self>;
        type Device = candle::Device;

        fn device(_: &Self::Field) -> Result<Self::Device> {
            Ok(candle::Device::new_metal(N)?)
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
    type Device = candle::Device;

    fn device(_: &Self::Field) -> Result<Self::Device> {
        Ok(candle::Device::Cpu)
    }

    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}
impl DynDevice for Cpu {}

impl Device for Dyn {
    type Arg = KindleDevice;
    type Field = KindleDevice;
    type Device = candle::Device;

    fn device(dev: &Self::Field) -> Result<<Dyn as Device>::Device> {
        Ok(match &dev.0 {
            DeviceVariant::Cpu => candle::Device::Cpu,
            #[cfg(feature = "cuda")]
            DeviceVariant::Cuda(ord) => candle::Device::new_cuda(ord)?,
            #[cfg(feature = "metal")]
            DeviceVariant::Metal(ord) => candle::Device::new_metal(ord)?,
        })
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
