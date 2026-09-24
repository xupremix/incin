use incin_core::backend_authoring::*;
use incin_core::error::{Error, Result};
use incin_core::shapes::Dyn;
use incin_core::tensor::device::{Device, DeviceId};
use incin_core::tensor::dtype::{DType, DTypeDescriptor, DTypeId};

macro_rules! impl_wgpu_supports_dtype {
    ($($t:ty),*) => {
        $(
            impl<D: Device> SupportsDType<$t> for super::backend::WgpuBackendImpl<D> {
                fn resolve_dtype(field: &<$t as DType>::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
                    let dt = <$t as DType>::descriptor(field);
                    validate_wgpu_dtype(dt, "dtype")?;
                    Ok(dt)
                }
            }
        )*
    };
}

impl_wgpu_supports_dtype!(f32, u32, i64, u8, bool, half::f16, half::bf16);

impl<D: Device> SupportsDType<Dyn> for super::backend::WgpuBackendImpl<D> {
    fn resolve_dtype(field: &DTypeDescriptor, _device: &DeviceId) -> Result<DTypeDescriptor> {
        validate_wgpu_dtype(*field, "dtype")?;
        Ok(*field)
    }
}

/// The dtypes this backend can honestly hold and round-trip: `f32` compute
/// storage, `bool` as a physical `f32` of 0.0/1.0 (WGSL storage buffers
/// cannot hold `bool`), and the integer widths the indexing paths need for
/// index operands (`u8`/`u32`/`i64` at their own physical widths).
///
/// `f16`/`bf16`/`f64`/`q8_0` stay refused: no WGPU kernel here reads them
/// and the creation paths only ever build `Vec<f32>`.
pub(crate) fn validate_wgpu_dtype(dtype: DTypeDescriptor, op: &'static str) -> Result<()> {
    match dtype.builtin_id() {
        Some(DTypeId::F32 | DTypeId::Bool | DTypeId::U8 | DTypeId::U32 | DTypeId::I64) => Ok(()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Wgpu",
            op,
        }),
    }
}

/// Precision resolution stays an explicit `F32`-only check, deliberately not
/// routed through [`validate_wgpu_dtype`]: that function now admits the
/// integer and `bool` storage dtypes the indexing rows need, and a widened
/// `native_precision` would then claim `f32` *compute* for an `i64` tensor
/// just because the storage dtype is on the allowed list. Compute here is
/// still `array<f32>` in every shader.
pub(crate) fn native_precision(
    request: &incin_core::exec::PrecisionRequest,
) -> Result<incin_core::exec::ResolvedPrecision> {
    if request.storage != DTypeId::F32.descriptor() {
        return Err(Error::UnsupportedDType {
            dtype: request.storage,
            backend: "Wgpu",
            op: "native_precision",
        });
    }
    Ok(incin_core::exec::ResolvedPrecision::new(
        request.storage,
        DTypeId::F32.descriptor(),
        DTypeId::F32.descriptor(),
        request.output,
        incin_core::exec::LossScaling::None,
    ))
}

impl<D: Device> incin_core::exec::PrecisionCapabilities for super::backend::WgpuBackendImpl<D> {
    fn native_precision(
        &self,
        request: &incin_core::exec::PrecisionRequest,
    ) -> Result<incin_core::exec::ResolvedPrecision> {
        native_precision(request)
    }
}
