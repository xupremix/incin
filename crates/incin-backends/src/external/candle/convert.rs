//! Conversions between Incin and Candle device and dtype identifiers.

use crate::external::*;
use candle_core as candle;

/// Converts a incin `DeviceId` into a candle `Device`, mapping CPU/CUDA/wgpu
/// device kinds and erroring on any device kind Candle doesn't support.
pub fn to_candle_device(dev: &DeviceId) -> Result<candle::Device> {
    use incin_core::prelude::DeviceKind;
    match dev.kind() {
        DeviceKind::Cpu => Ok(candle::Device::Cpu),
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => Ok(candle::Device::new_cuda(dev.ordinal())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
        #[cfg(feature = "wgpu")]
        DeviceKind::Wgpu => Ok(candle::Device::new_metal(dev.ordinal())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
        _ => Err(Error::UnsupportedBackendOperation {
            op: "to_candle_device",
            backend: "Candle",
        }),
    }
}

/// Maps an Incin dtype to Candle, returning a typed error when Candle has
/// no native representation for it.
pub fn to_candle_dtype(dtype: DTypeId) -> Result<candle::DType> {
    match dtype {
        DTypeId::U8 => Ok(candle::DType::U8),
        DTypeId::U32 => Ok(candle::DType::U32),
        DTypeId::I64 => Ok(candle::DType::I64),
        DTypeId::BF16 => Ok(candle::DType::BF16),
        DTypeId::F16 => Ok(candle::DType::F16),
        DTypeId::F32 => Ok(candle::DType::F32),
        DTypeId::F64 => Ok(candle::DType::F64),
        _ => Err(Error::UnsupportedBackendOperation {
            op: "to_candle_dtype",
            backend: "external Candle",
        }),
    }
}
