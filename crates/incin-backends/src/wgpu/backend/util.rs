//! Small shared helpers: element counts, checked `u32` conversions, and
//! device/dtype validation.

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: compute flat element count from shape
// ─────────────────────────────────────────────────────────────────────────────
/// `num_elements`.
pub(crate) fn num_elements(shape: &[usize]) -> Result<usize> {
    ShapeBuf::from_slice(shape)
        .checked_numel(OperationKind::Storage)
        .map_err(Into::into)
}

pub(crate) fn checked_u32(value: usize, expression: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        incin_core::shapes::ShapeError::ArithmeticOverflow {
            operation: OperationKind::Storage,
            expression,
        }
        .into()
    })
}

pub(crate) fn checked_u32_array<const N: usize>(
    values: [usize; N],
    expression: &'static str,
) -> Result<[u32; N]> {
    let mut checked = [0; N];
    for (target, value) in checked.iter_mut().zip(values) {
        *target = checked_u32(value, expression)?;
    }
    Ok(checked)
}

pub(crate) fn validate_wgpu(
    dtype: DTypeDescriptor,
    device: &DeviceId,
    _family: OperationKind,
    op: &'static str,
) -> Result<()> {
    if device.kind() != DeviceKind::Wgpu {
        return Err(Error::DeviceInitializationError {
            expected: "wgpu".to_string(),
            got: format!("{:?}", device.kind()),
        });
    }
    if device.ordinal() != 0 {
        return Err(Error::InvalidDeviceOrdinal {
            backend: "Wgpu",
            ordinal: device.ordinal(),
        });
    }
    validate_wgpu_dtype(dtype, op)
}
