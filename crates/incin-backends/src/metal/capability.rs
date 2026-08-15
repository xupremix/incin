use incin_core::backend_authoring::*;
use incin_core::error::{Error, Result};
use incin_core::shapes::OperationKind;
use incin_core::tensor::device::{Device, DeviceId};
use incin_core::tensor::dtype::{DType, DTypeDescriptor, DTypeId};

impl<K: DType, D: Device> SupportsDType<K> for super::backend::MetalBackendImpl<D> {
    fn resolve_dtype(field: &K::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
        let descriptor = K::descriptor(field);
        validate_metal_storage_dtype(descriptor, "resolve_dtype")?;
        Ok(descriptor)
    }
}

pub(crate) fn validate_metal_storage_dtype(dtype: DTypeDescriptor, op: &'static str) -> Result<()> {
    if matches!(
        dtype.builtin_id(),
        Some(
            DTypeId::F32
                | DTypeId::F64
                | DTypeId::F16
                | DTypeId::BF16
                | DTypeId::I64
                | DTypeId::Q8_0
        )
    ) {
        Ok(())
    } else {
        Err(Error::UnsupportedDType {
            dtype,
            backend: "Metal",
            op,
        })
    }
}

pub(crate) fn native_precision(
    request: &incin_core::exec::PrecisionRequest,
) -> Result<incin_core::exec::ResolvedPrecision> {
    validate_metal_storage_dtype(request.storage, "native_precision")?;
    let compute = match request.storage.builtin_id() {
        Some(DTypeId::F16 | DTypeId::BF16) => DTypeId::F32.descriptor(),
        _ => request.storage,
    };
    let accumulator = match request.operation {
        OperationKind::Reduction | OperationKind::Normalization
            if matches!(
                request.storage.builtin_id(),
                Some(DTypeId::F16 | DTypeId::BF16)
            ) =>
        {
            DTypeId::F32.descriptor()
        }
        _ => compute,
    };
    Ok(incin_core::exec::ResolvedPrecision::new(
        request.storage,
        compute,
        accumulator,
        request.output,
        incin_core::exec::LossScaling::None,
    ))
}

impl<D: Device> incin_core::exec::PrecisionCapabilities for super::backend::MetalBackendImpl<D> {
    fn native_precision(
        &self,
        request: &incin_core::exec::PrecisionRequest,
    ) -> Result<incin_core::exec::ResolvedPrecision> {
        native_precision(request)
    }
}
