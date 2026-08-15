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

impl_wgpu_supports_dtype!(f32, u32, i64, u8, half::f16, half::bf16);

impl<D: Device> SupportsDType<Dyn> for super::backend::WgpuBackendImpl<D> {
    fn resolve_dtype(field: &DTypeDescriptor, _device: &DeviceId) -> Result<DTypeDescriptor> {
        validate_wgpu_dtype(*field, "dtype")?;
        Ok(*field)
    }
}

pub(crate) fn validate_wgpu_dtype(dtype: DTypeDescriptor, op: &'static str) -> Result<()> {
    if dtype.builtin_id() == Some(DTypeId::F32) {
        Ok(())
    } else {
        Err(Error::UnsupportedDType { dtype, backend: "Wgpu", op })
    }
}

pub(crate) fn native_precision(
    request: &incin_core::exec::PrecisionRequest,
) -> Result<incin_core::exec::ResolvedPrecision> {
    validate_wgpu_dtype(request.storage, "native_precision")?;
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
