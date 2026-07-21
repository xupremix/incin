use core::fmt::Debug;
use core::marker::PhantomData;

use crate::prelude::{Dyn, Result};

/// Core abstraction for `Device` within the Kindle framework..
pub trait Device: 'static + Send + Sync + Clone + Eq + PartialEq + Debug + Sized {
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg: Clone;
    /// Core abstraction for `Field` within the Kindle framework..
    type Field: Debug + Clone;
    /// Core abstraction for `init` within the Kindle framework..
    fn init(arg: Self::Arg) -> Self::Field;
    /// Core abstraction for `to_kindle` within the Kindle framework..
    fn to_kindle(dev: &Self::Field) -> Result<KindleDevice>;
}
/// Core abstraction for `DynDevice` within the Kindle framework..
pub trait DynDevice: Device {}
/// Core abstraction for `ConstDevice` within the Kindle framework..
pub trait ConstDevice: Default + Device<Arg = ()> {}

#[cfg(feature = "cuda")]
/// Core abstraction for `cuda` within the Kindle framework..
pub mod cuda {

    use super::{ConstDevice, Device, DynDevice, KindleDevice, PhantomData, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// Implementation of `Cuda` for the respective backend..
    pub struct Cuda<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Cuda<N> {}

    impl<const N: usize> Device for Cuda<N> {
        /// Core abstraction for `Arg` within the Kindle framework..
        type Arg = ();
        /// Core abstraction for `Field` within the Kindle framework..
        type Field = PhantomData<Self>;

        /// Core abstraction for `to_kindle` within the Kindle framework..
        fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
            Ok(KindleDevice::cuda(N))
        }

        /// Core abstraction for `init` within the Kindle framework..
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }

    impl<const N: usize> DynDevice for Cuda<N> {}
}

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "wgpu")]
/// Core abstraction for `wgpu` within the Kindle framework..
pub mod wgpu {
    use super::{ConstDevice, Device, DynDevice, KindleDevice, PhantomData, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// Implementation of `Wgpu` for the respective backend..
    pub struct Wgpu<const N: usize = 0>;

    impl<const N: usize> ConstDevice for Wgpu<N> {}

    impl<const N: usize> Device for Wgpu<N> {
        /// Core abstraction for `Arg` within the Kindle framework..
        type Arg = ();
        /// Core abstraction for `Field` within the Kindle framework..
        type Field = PhantomData<Self>;

        /// Core abstraction for `to_kindle` within the Kindle framework..
        fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
            Ok(KindleDevice::wgpu(N))
        }

        /// Core abstraction for `init` within the Kindle framework..
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
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg = ();
    /// Core abstraction for `Field` within the Kindle framework..
    type Field = PhantomData<Self>;

    /// Core abstraction for `to_kindle` within the Kindle framework..
    fn to_kindle(_: &Self::Field) -> Result<KindleDevice> {
        Ok(KindleDevice::cpu())
    }

    /// Core abstraction for `init` within the Kindle framework..
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}
impl DynDevice for Cpu {}

impl Device for Dyn {
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg = KindleDevice;
    /// Core abstraction for `Field` within the Kindle framework..
    type Field = KindleDevice;

    /// Core abstraction for `to_kindle` within the Kindle framework..
    fn to_kindle(dev: &Self::Field) -> Result<KindleDevice> {
        Ok(*dev)
    }

    /// Core abstraction for `init` within the Kindle framework..
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}
impl DynDevice for Dyn {}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Core abstraction for `DeviceVariant` within the Kindle framework..
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
/// Core abstraction for `KindleDevice` within the Kindle framework..
pub struct KindleDevice(DeviceVariant);

impl KindleDevice {
    /// Core abstraction for `variant` within the Kindle framework..
    pub fn variant(&self) -> DeviceVariant {
        self.0
    }

    /// Core abstraction for `cpu` within the Kindle framework..
    pub fn cpu() -> Self {
        Self(DeviceVariant::Cpu)
    }

    #[cfg(feature = "cuda")]
    /// Core abstraction for `cuda` within the Kindle framework..
    pub fn cuda(ord: usize) -> Self {
        Self(DeviceVariant::Cuda(ord))
    }

    #[cfg(feature = "wgpu")]
    /// Core abstraction for `wgpu` within the Kindle framework..
    pub fn wgpu(ord: usize) -> Self {
        Self(DeviceVariant::Wgpu(ord))
    }
}

/// Core abstraction for `fn` within the Kindle framework..
pub const fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}
/// Core abstraction for `fn` within the Kindle framework..
pub const fn wgpu_is_available() -> bool {
    cfg!(feature = "wgpu")
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// Implementation of `CudaDevice` for the respective backend..
pub struct CudaDevice {
    /// Core abstraction for `id` within the Kindle framework..
    pub id: usize,
}

impl CudaDevice {
    /// Core abstraction for `new` within the Kindle framework..
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// Implementation of `WgpuDevice` for the respective backend..
pub struct WgpuDevice {
    /// Core abstraction for `id` within the Kindle framework..
    pub id: usize,
}

impl WgpuDevice {
    /// Core abstraction for `new` within the Kindle framework..
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

#[cfg(test)]
/// Core abstraction for `tests` within the Kindle framework..
mod tests {
    use super::*;

    #[test]
    /// Core abstraction for `test_device_variants` within the Kindle framework..
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
