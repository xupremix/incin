//! Wraps the `candle_core` crate, providing `CandleBackend` as a `Backend`
//! implementation backed by Candle's own tensor type.

use incin_core::prelude::Cpu;

mod backend;
pub mod convert;
mod executor;
mod ops;

pub use convert::{from_candle_device, from_candle_dtype, to_candle_device, to_candle_dtype};
pub use executor::CandleStorage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandleBackend<D = Cpu>(core::marker::PhantomData<D>);

impl<D> CandleBackend<D> {
    /// Construct the stateless Candle executor.
    #[must_use]
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<D> Default for CandleBackend<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
/// Unit tests for the candle dtype and device conversion helpers.
mod tests {
    use super::*;

    use candle_core as candle;
    use incin_core::prelude::{DTypeId, DeviceId};
    #[test]
    /// Checks that `to_candle_dtype` maps `F32` and `I64` to the
    /// corresponding candle dtypes.
    fn test_to_candle_dtype() {
        assert_eq!(
            to_candle_dtype(DTypeId::F32.into()).unwrap(),
            candle::DType::F32
        );
        assert_eq!(
            to_candle_dtype(DTypeId::I64.into()).unwrap(),
            candle::DType::I64
        );
    }

    #[test]
    /// Checks that `to_candle_device` maps the CPU device kind to
    /// `candle::Device::Cpu`.
    fn test_to_candle_device() {
        let cpu = DeviceId::cpu();
        let c_dev = to_candle_device(&cpu).unwrap();
        assert!(matches!(c_dev, candle::Device::Cpu));
    }
}
