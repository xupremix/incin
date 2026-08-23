//! Checked byte lengths for tensor storage allocation.

use crate::resource::ResourceLimits;
use crate::shapes::CheckedNumel;
use crate::shapes::error::{OperationKind, ShapeError};
use crate::tensor::dtype::DTypeDescriptor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Bounded and verified allocation byte length (`SEC-011`).
pub struct CheckedByteLen(usize);

impl CheckedByteLen {
    /// Computes and validates the dense byte length for `dims` and `dtype`.
    pub fn from_dims(
        operation: OperationKind,
        dims: &[usize],
        dtype: DTypeDescriptor,
        limits: &ResourceLimits,
    ) -> Result<Self, ShapeError> {
        let numel = CheckedNumel::from_dims(operation, dims, limits)?;
        let bytes = dtype.size_bytes(numel.get(), operation)?;
        if u64::try_from(bytes).map_or(true, |bytes| bytes > limits.max_tensor_bytes) {
            return Err(ShapeError::ArithmeticOverflow {
                operation,
                expression: "tensor byte length exceeds resource limit",
            });
        }
        Ok(Self(bytes))
    }

    #[inline]
    /// Alignment in bytes.
    pub fn get(self) -> usize {
        self.0
    }
}

/// Safely computes byte allocation length using dtype block metrics and limits.
pub fn checked_byte_len_from_dims(
    dims: &[usize],
    dtype: DTypeDescriptor,
    limits: &ResourceLimits,
) -> Result<CheckedByteLen, ShapeError> {
    CheckedByteLen::from_dims(OperationKind::Reshape, dims, dtype, limits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dtype::{
        ConstDType, DTypeDescriptor, DTypeKey, DTypeKind, Q8_0, StorageEncoding,
    };

    #[test]
    fn checked_allocation_lengths_cover_dense_and_block_storage() {
        let mut limits = ResourceLimits::trusted_local_large_model();
        limits.max_rank = 8;
        limits.max_dimension = u64::MAX;
        limits.max_tensor_bytes = u64::MAX;

        assert_eq!(
            CheckedByteLen::from_dims(
                OperationKind::Storage,
                &[2, 3],
                <f32 as ConstDType>::DESCRIPTOR,
                &limits
            )
            .unwrap()
            .get(),
            24
        );
        assert_eq!(
            CheckedByteLen::from_dims(
                OperationKind::Storage,
                &[64],
                <Q8_0 as ConstDType>::DESCRIPTOR,
                &limits
            )
            .unwrap()
            .get(),
            68
        );
        let custom = DTypeDescriptor::new(
            DTypeKey::new("custom", "test_block", 1),
            DTypeKind::Quantized,
            StorageEncoding::block(16, 20, 2),
        );
        assert_eq!(
            CheckedByteLen::from_dims(OperationKind::Storage, &[32], custom, &limits)
                .unwrap()
                .get(),
            40
        );

        limits.max_tensor_bytes = 23;
        assert!(matches!(
            CheckedByteLen::from_dims(
                OperationKind::Storage,
                &[2, 3],
                <f32 as ConstDType>::DESCRIPTOR,
                &limits
            ),
            Err(ShapeError::ArithmeticOverflow { .. })
        ));
    }
}
