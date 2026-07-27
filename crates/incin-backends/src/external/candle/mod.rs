//! Wraps the `candle_core` crate, providing `CandleBackend` as a `Backend`
//! implementation backed by Candle's own tensor type.

mod backend;
pub mod convert;
mod ops;

pub use convert::{to_candle_device, to_candle_dtype};

/// # Backend Float Element Limitation (B-4)
/// **Known Limitation:** `CandleBackend` ignores its compile-time `T` generic
/// for inner allocation precision and relies on the dynamic `DTypeId`
/// supplied to creation methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandleBackend<T, D>(core::marker::PhantomData<(T, D)>);

#[cfg(test)]
/// Unit tests for the candle dtype and device conversion helpers.
mod tests {
    use super::*;

    #[test]
    /// Checks that `to_candle_dtype` maps `F32` and `I64` to the
    /// corresponding candle dtypes.
    fn test_to_candle_dtype() {
        assert_eq!(to_candle_dtype(DTypeId::F32).unwrap(), candle::DType::F32);
        assert_eq!(to_candle_dtype(DTypeId::I64).unwrap(), candle::DType::I64);
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
