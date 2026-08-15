use half::{bf16, f16};
use incin_core::backend_authoring::*;
use incin_core::error::{Error, Result};
use incin_core::shapes::{Dyn, OperationKind};
use incin_core::tensor::device::{Device, DeviceId};
use incin_core::tensor::dtype::{DType, DTypeDescriptor, DTypeId};

macro_rules! impl_cuda_storage_dtype {
    ($($dtype:ty),+ $(,)?) => {
        $(
            impl<D: Device> SupportsDType<$dtype> for super::backend::CudaBackendImpl<D> {
                fn resolve_dtype(
                    field: &<$dtype as DType>::Field,
                    _device: &DeviceId,
                ) -> Result<DTypeDescriptor> {
                    let descriptor = <$dtype as DType>::descriptor(field);
                    validate_cuda_storage_dtype(descriptor, "resolve_dtype")?;
                    Ok(descriptor)
                }
            }
        )+
    };
}

impl_cuda_storage_dtype!(f32, f64, f16, bf16, i64);

pub(crate) fn validate_cuda_storage_dtype(dtype: DTypeDescriptor, op: &'static str) -> Result<()> {
    let is_supported = matches!(
        dtype.builtin_id(),
        Some(
            DTypeId::F32
                | DTypeId::F64
                | DTypeId::F16
                | DTypeId::BF16
                | DTypeId::I64
                | DTypeId::Q8_0
        )
    );
    if is_supported {
        Ok(())
    } else {
        Err(Error::UnsupportedDType {
            dtype,
            backend: "Cuda",
            op,
        })
    }
}

pub(crate) fn require_cuda_builtin_dtype(
    descriptor: DTypeDescriptor,
    op: &'static str,
) -> Result<DTypeId> {
    validate_cuda_storage_dtype(descriptor, op)?;
    descriptor.builtin_id().ok_or(Error::UnsupportedDType {
        dtype: descriptor,
        backend: "Cuda",
        op,
    })
}

pub(crate) fn native_precision(
    request: &incin_core::exec::PrecisionRequest,
) -> Result<incin_core::exec::ResolvedPrecision> {
    validate_cuda_storage_dtype(request.storage, "native_precision")?;

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

impl<D: Device> incin_core::exec::PrecisionCapabilities for super::backend::CudaBackendImpl<D> {
    fn native_precision(
        &self,
        request: &incin_core::exec::PrecisionRequest,
    ) -> Result<incin_core::exec::ResolvedPrecision> {
        native_precision(request)
    }
}

impl<D: Device> SupportsDType<Dyn> for super::backend::CudaBackendImpl<D> {
    fn resolve_dtype(field: &DTypeDescriptor, _device: &DeviceId) -> Result<DTypeDescriptor> {
        validate_cuda_storage_dtype(*field, "resolve_dtype")?;
        Ok(*field)
    }
}
