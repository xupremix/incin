//! Descriptor execution for the third-party Candle backend.
//!
//! Candle's own tensor type is foreign: it carries dimensions, strides, dtype,
//! and device, but no [`TensorMeta`], and no trait implementation can add a
//! field to a type this crate does not own. [`CandleStorage`] is the seam —
//! it pairs a borrowed-or-owned `candle_core::Tensor` with the checked metadata
//! the descriptor contract requires, validated once at the boundary.
//!
//! This deliberately does *not* change `Backend::Storage<K>`, which stays a raw
//! `candle_core::Tensor`. `StorageBackend::Storage<K>` is a separate associated
//! type on a separate trait, so the descriptor path gets checked metadata while
//! the existing Candle operations keep operating on candle's own type. That
//! separation is what lets a foreign backend join the contract without its
//! adapter being rewritten, which is the property `EXE-010`'s backend-authoring
//! template needs to document.

use incin_core::backend_authoring::{Descriptor, Execute, ExecutionRequest, StorageBackend, op};
use incin_core::exec::{
    Alignment, Capabilities, CapabilityQuery, ExecutionDescriptor, SupportLevel, TensorMeta,
    UnsupportedReason,
};
use incin_core::prelude::{
    BackendError, DType, Device, OperationKind, Result, Shape, ShapeBuf, StrideBuf,
};

use super::CandleBackend;
use super::convert::{from_candle_device, from_candle_dtype};

/// A Candle tensor paired with the checked metadata the descriptor path needs.
#[derive(Debug, Clone)]
pub struct CandleStorage {
    tensor: candle_core::Tensor,
    meta: TensorMeta,
}

impl CandleStorage {
    /// Validate a Candle tensor's own geometry into checked metadata.
    ///
    /// Candle reports dimensions, strides, dtype, and device; every one of them
    /// is re-derived here rather than assumed, so a tensor whose stride layout
    /// does not span the elements it claims is rejected at the boundary instead
    /// of inside a kernel.
    pub fn try_new(tensor: candle_core::Tensor) -> Result<Self> {
        let shape = ShapeBuf::from_slice(tensor.dims());
        let strides = StrideBuf::from_slice(tensor.stride());
        let dtype = from_candle_dtype(tensor.dtype());
        let device = from_candle_device(tensor.device())?;
        let capacity = shape.checked_numel(OperationKind::Storage)?;
        let meta = TensorMeta::try_new(
            shape,
            strides,
            0,
            dtype,
            device,
            // Candle owns the allocation and exposes no alignment guarantee, so
            // the only honest claim is the universal one.
            Alignment::BYTE,
            capacity,
        )
        .map_err(|error| {
            incin_core::prelude::Error::Msg(alloc::format!(
                "invalid Candle storage metadata: {error}"
            ))
        })?;
        Ok(Self { tensor, meta })
    }

    /// The underlying Candle tensor.
    #[must_use]
    pub const fn tensor(&self) -> &candle_core::Tensor {
        &self.tensor
    }

    /// Consume the wrapper, returning the underlying Candle tensor.
    #[must_use]
    pub fn into_tensor(self) -> candle_core::Tensor {
        self.tensor
    }

    /// Checked physical metadata.
    #[must_use]
    pub const fn metadata(&self) -> &TensorMeta {
        &self.meta
    }
}

impl<D: Device> StorageBackend for CandleBackend<D> {
    const BACKEND_NAME: &'static str = "Candle";

    type Storage<K: DType> = CandleStorage;
    type Device = D;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage.metadata()
    }
}

impl<D: Device> Capabilities for CandleBackend<D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        // Candle is a third-party backend with no registry of its own. Its
        // dtype coverage is exactly what `to_candle_dtype` accepts, and
        // claiming anything beyond what this adapter actually routes would be
        // the kind of unearned capability claim `EXE-005` exists to prevent.
        if super::convert::to_candle_dtype(query.dtype).is_err() {
            return SupportLevel::Unsupported(UnsupportedReason::DType {
                operation: query.operation,
                dtype: query.dtype,
            });
        }
        match query.operation {
            OperationKind::MatMul
            | OperationKind::Reshape
            | OperationKind::Zeros
            | OperationKind::Ones
            | OperationKind::UniformRandom
            | OperationKind::NormalRandom => SupportLevel::Native,
            operation => SupportLevel::Unsupported(UnsupportedReason::Operation { operation }),
        }
    }
}

const fn invalid(operation: OperationKind, reason: &'static str) -> BackendError {
    BackendError::InvalidInput { operation, reason }
}

impl<D: Device> Execute<Descriptor<op::MatMulExact>> for CandleBackend<D> {
    type Output = CandleStorage;

    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::MatMulExact>, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let _ = self;
        let operation = OperationKind::MatMulExact;
        let [lhs_handle, rhs_handle] = request.inputs else {
            return Err(invalid(
                operation,
                "matmul expects exactly two tensor inputs",
            ));
        };
        let lhs = lhs_handle
            .downcast_ref::<CandleStorage>()
            .ok_or_else(|| invalid(operation, "matmul input is not Candle storage"))?;
        let rhs = rhs_handle
            .downcast_ref::<CandleStorage>()
            .ok_or_else(|| invalid(operation, "matmul input is not Candle storage"))?;

        let execution_error = |error: candle_core::Error| BackendError::Execution {
            operation,
            message: alloc::format!("{error}").into(),
        };
        let output = lhs
            .tensor()
            .broadcast_matmul(rhs.tensor())
            .map_err(execution_error)?;
        let output = CandleStorage::try_new(output).map_err(|error| BackendError::Execution {
            operation,
            message: alloc::format!("{error}").into(),
        })?;
        let expected = request
            .operation
            .descriptor()
            .output_shape()
            .ok_or_else(|| invalid(operation, "matmul descriptor has no output shape"))?;
        if output.metadata().shape().dims() != expected.dims() {
            return Err(BackendError::Execution {
                operation,
                message: "Candle matmul output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

impl<D: Device> Execute<Descriptor<op::ReshapeExact>> for CandleBackend<D> {
    type Output = CandleStorage;

    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::ReshapeExact>, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let _ = self;
        let operation = OperationKind::ReshapeExact;
        let [handle] = request.inputs else {
            return Err(invalid(
                operation,
                "reshape expects exactly one tensor input",
            ));
        };
        let input = handle
            .downcast_ref::<CandleStorage>()
            .ok_or_else(|| invalid(operation, "reshape input is not Candle storage"))?;
        let shape = &request.operation.descriptor().attributes().shape;
        let execution_error = |error: candle_core::Error| BackendError::Execution {
            operation,
            message: alloc::format!("{error}").into(),
        };
        let output = input
            .tensor()
            .reshape(shape.as_slice())
            .map_err(execution_error)?;
        let output = CandleStorage::try_new(output).map_err(|error| BackendError::Execution {
            operation,
            message: alloc::format!("{error}").into(),
        })?;
        if output.metadata().shape().dims() != shape {
            return Err(BackendError::Execution {
                operation,
                message: "Candle reshape output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

macro_rules! impl_candle_creation_executors {
    ($(($op:ident, $func:ident $(, $arg:ident)*)),* $(,)?) => {$(
        impl<D: Device> Execute<incin_core::backend_authoring::Descriptor<incin_core::backend_authoring::op::$op>> for CandleBackend<D> {
            type Output = CandleStorage;
            fn execute_shaped<ShapeTy: Shape>(
                &self,
                request: ExecutionRequest<'_, incin_core::backend_authoring::Descriptor<incin_core::backend_authoring::op::$op>, Self>,
            ) -> core::result::Result<CandleStorage, BackendError> {
                use incin_core::backend_authoring::CreationOps;
                if !request.inputs.is_empty() {
                    return Err(invalid(OperationKind::$op, "an allocation takes no operand"));
                }
                let attr = request.operation.descriptor().attributes();
                let raw = <Self as CreationOps<Self>>::$func::<f32>(
                    $(attr.$arg,)*
                    &attr.shape,
                    attr.dtype,
                    &attr.device,
                )
                .map_err(|err| {
                    crate::descriptor_bind::kernel_error(
                        Self::BACKEND_NAME,
                        OperationKind::$op,
                        err,
                    )
                })?;
                Ok(raw)
            }
        }
    )*};
}

impl_candle_creation_executors![
    (Zeros, zeros),
    (Ones, ones),
    (UniformRandom, rand),
    (NormalRandom, randn),
    (Full, full, value),
    (Arange, arange, start, step),
    (Linspace, linspace, start, end),
];
