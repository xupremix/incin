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

use incin_core::exec::{
    Alignment, Capabilities, CapabilityQuery, MatMulSpec, ReshapeSpec, SupportLevel, TensorMeta,
    UnsupportedReason,
};
use incin_core::prelude::{
    BackendError, DType, Device, Execute, ExecutionRequest, OperationKind, Result, ShapeBuf,
    StorageBackend, StrideBuf,
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

impl<T: DType, D: Device> StorageBackend for CandleBackend<T, D> {
    type Storage<K: DType> = CandleStorage;
    type Device = D;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage.metadata()
    }
}

impl<T: DType, D: Device> Capabilities for CandleBackend<T, D> {
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
            OperationKind::MatMul | OperationKind::Reshape => SupportLevel::Native,
            operation => SupportLevel::Unsupported(UnsupportedReason::Operation { operation }),
        }
    }
}

const fn invalid(operation: OperationKind, reason: &'static str) -> BackendError {
    BackendError::InvalidInput { operation, reason }
}

impl<T: DType, D: Device> Execute<MatMulSpec> for CandleBackend<T, D> {
    type Output = CandleStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, MatMulSpec, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let [lhs_handle, rhs_handle] = request.inputs else {
            return Err(invalid(
                OperationKind::MatMul,
                "matmul expects exactly two tensor inputs",
            ));
        };
        let lhs = lhs_handle
            .downcast_ref::<CandleStorage>()
            .ok_or_else(|| invalid(OperationKind::MatMul, "matmul input is not Candle storage"))?;
        let rhs = rhs_handle
            .downcast_ref::<CandleStorage>()
            .ok_or_else(|| invalid(OperationKind::MatMul, "matmul input is not Candle storage"))?;

        let execution_error = |error: candle_core::Error| BackendError::Execution {
            operation: OperationKind::MatMul,
            message: alloc::format!("{error}"),
        };

        let transposed = |storage: &CandleStorage| {
            let rank = storage.metadata().shape().rank();
            storage.tensor().transpose(rank - 2, rank - 1)
        };
        let lhs_tensor = if spec.transpose_lhs {
            transposed(lhs).map_err(execution_error)?
        } else {
            lhs.tensor().clone()
        };
        let rhs_tensor = if spec.transpose_rhs {
            transposed(rhs).map_err(execution_error)?
        } else {
            rhs.tensor().clone()
        };

        // `broadcast_matmul` covers both the rank-2 and the batched cases, and
        // applies the same right-aligned batch broadcasting the descriptor's
        // batch strides already describe.
        let output = lhs_tensor
            .broadcast_matmul(&rhs_tensor)
            .map_err(execution_error)?;
        let output = CandleStorage::try_new(output).map_err(|error| BackendError::Execution {
            operation: OperationKind::MatMul,
            message: alloc::format!("{error}"),
        })?;

        if output.metadata().shape().dims() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::MatMul,
                message: "Candle matmul output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}

impl<T: DType, D: Device> Execute<ReshapeSpec> for CandleBackend<T, D> {
    type Output = CandleStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, ReshapeSpec, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let _ = self;
        let spec = request.operation.descriptor();
        let [handle] = request.inputs else {
            return Err(invalid(
                OperationKind::Reshape,
                "reshape expects exactly one tensor input",
            ));
        };
        let input = handle.downcast_ref::<CandleStorage>().ok_or_else(|| {
            invalid(
                OperationKind::Reshape,
                "reshape input is not Candle storage",
            )
        })?;

        if input.metadata().shape().dims() != spec.input.dims() {
            return Err(invalid(
                OperationKind::Reshape,
                "reshape input metadata does not match the validated descriptor",
            ));
        }

        let execution_error = |error: candle_core::Error| BackendError::Execution {
            operation: OperationKind::Reshape,
            message: alloc::format!("{error}"),
        };
        // Candle refuses to reshape a non-contiguous tensor rather than
        // materializing one, so a strided operand surfaces as its own error here
        // instead of being silently copied.
        let output = input
            .tensor()
            .reshape(spec.output.dims())
            .map_err(execution_error)?;
        let output = CandleStorage::try_new(output).map_err(|error| BackendError::Execution {
            operation: OperationKind::Reshape,
            message: alloc::format!("{error}"),
        })?;

        if output.metadata().shape().dims() != spec.output.dims() {
            return Err(BackendError::Execution {
                operation: OperationKind::Reshape,
                message: "Candle reshape output disagrees with the validated descriptor".into(),
            });
        }
        Ok(output)
    }
}
