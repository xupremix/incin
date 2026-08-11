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

/// Recovers the Incin `DeviceId` a candle `Device` stands for.
///
/// This is the inverse of [`to_candle_device`] and is what lets a foreign
/// tensor report checked [`TensorMeta`](incin_core::exec::TensorMeta): the
/// descriptor contract records the device the data actually sits on, and only
/// candle knows that once it owns the allocation.
pub fn from_candle_device(device: &candle::Device) -> Result<DeviceId> {
    match device.location() {
        candle::DeviceLocation::Cpu => Ok(DeviceId::cpu()),
        candle::DeviceLocation::Cuda { gpu_id } => Ok(DeviceId::cuda(gpu_id)),
        candle::DeviceLocation::Metal { gpu_id } => Ok(DeviceId::wgpu(gpu_id)),
    }
}

/// Recovers the Incin dtype a candle `DType` stands for.
///
/// The inverse of [`to_candle_dtype`]. Candle has no `Q8_0`, so the mapping is
/// total in this direction.
pub fn from_candle_dtype(dtype: candle::DType) -> DTypeDescriptor {
    match dtype {
        candle::DType::U8 => DTypeId::U8.descriptor(),
        candle::DType::U32 => DTypeId::U32.descriptor(),
        candle::DType::I64 => DTypeId::I64.descriptor(),
        candle::DType::BF16 => DTypeId::BF16.descriptor(),
        candle::DType::F16 => DTypeId::F16.descriptor(),
        candle::DType::F32 => DTypeId::F32.descriptor(),
        candle::DType::F64 => DTypeId::F64.descriptor(),
    }
}

/// Maps an Incin dtype to Candle, returning a typed error when Candle has
/// no native representation for it.
pub fn to_candle_dtype(descriptor: DTypeDescriptor) -> Result<candle::DType> {
    let Some(id) = descriptor.builtin_id() else {
        return Err(Error::UnsupportedBackendOperation {
            op: "to_candle_dtype",
            backend: "external Candle",
        });
    };
    match id {
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
