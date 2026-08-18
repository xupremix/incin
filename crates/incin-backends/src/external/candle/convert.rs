//! Conversions between Incin and Candle device and dtype identifiers.

use crate::external::*;
use candle_core as candle;

/// Converts a incin `DeviceId` into a candle `Device`, mapping CPU/CUDA/wgpu
/// device kinds and erroring on any device kind Candle doesn't support.
pub fn to_candle_device(dev: &DeviceId) -> Result<candle::Device> {
    use incin_core::tensor::device::DeviceKind;
    match dev.kind() {
        DeviceKind::Cpu => Ok(candle::Device::Cpu),
        #[cfg(feature = "cuda")]
        DeviceKind::Cuda => Ok(candle::Device::new_cuda(dev.ordinal())
            .map_err(|e: candle_core::Error| anyhow::anyhow!(e))?),
        #[cfg(feature = "metal")]
        DeviceKind::Metal => Ok(candle::Device::new_metal(dev.ordinal())
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
        candle::DeviceLocation::Metal { gpu_id } => Ok(DeviceId::metal(gpu_id)),
    }
}

/// Recovers the Incin dtype a candle `DType` stands for.
///
/// The partial inverse of [`to_candle_dtype`]. Candle 0.9.2 carries `I16`,
/// `I32` and four float8 variants that Incin has no dtype for, so the mapping
/// is not total and this returns a typed error rather than choosing a
/// same-width Incin dtype, which would silently reinterpret the bytes.
///
/// It returns a `Result` because every caller reaches it from a tensor the
/// user supplied. Aborting on a checkpoint that happens to hold an `F8E4M3`
/// weight is a process boundary the error contract does not allow here.
pub fn from_candle_dtype(dtype: candle::DType) -> Result<DTypeDescriptor> {
    Ok(match dtype {
        candle::DType::U8 => DTypeId::U8.descriptor(),
        candle::DType::U32 => DTypeId::U32.descriptor(),
        candle::DType::I64 => DTypeId::I64.descriptor(),
        candle::DType::BF16 => DTypeId::BF16.descriptor(),
        candle::DType::F16 => DTypeId::F16.descriptor(),
        candle::DType::F32 => DTypeId::F32.descriptor(),
        candle::DType::F64 => DTypeId::F64.descriptor(),
        _ => {
            return Err(Error::UnsupportedBackendOperation {
                op: "from_candle_dtype",
                backend: "external Candle",
            });
        }
    })
}

/// Maps an Incin dtype to Candle, returning a typed error when Candle has
/// no native representation for it.
pub fn to_candle_dtype(descriptor: DTypeDescriptor) -> Result<candle::DType> {
    // A dtype Candle cannot represent is a dtype rejection, not an unknown
    // operation: `UnsupportedDType` is what the WGPU and Metal backends return
    // for the same condition, and it carries the descriptor that was refused.
    let unsupported = || Error::UnsupportedDType {
        dtype: descriptor,
        backend: "external Candle",
        op: "to_candle_dtype",
    };
    let Some(id) = descriptor.builtin_id() else {
        return Err(unsupported());
    };
    match id {
        DTypeId::U8 => Ok(candle::DType::U8),
        DTypeId::U32 => Ok(candle::DType::U32),
        DTypeId::I64 => Ok(candle::DType::I64),
        DTypeId::BF16 => Ok(candle::DType::BF16),
        DTypeId::F16 => Ok(candle::DType::F16),
        DTypeId::F32 => Ok(candle::DType::F32),
        DTypeId::F64 => Ok(candle::DType::F64),
        _ => Err(unsupported()),
    }
}
