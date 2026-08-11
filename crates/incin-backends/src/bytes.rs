//! One checked byte-length computation for every device allocation.
//!
//! Before `EXE-008` each accelerator sized its own buffers. WGPU multiplied an
//! element count by a literal `4` or by `size_of::<f32>()`; CUDA carried three
//! copies of the same `checked_byte_len` helper and multiplied by `4` wherever
//! it did not. Both spellings say "every tensor is dense `f32`", which is why a
//! non-`f32` allocation could be sized as though it were one.
//!
//! [`DTypeId::size_bytes`] is the checked, block-aware replacement. This module
//! is the thin backend-facing wrapper that reports its failures as the crate's
//! [`Error`] type.

use incin_core::prelude::{DTypeDescriptor, OperationKind, Result};

#[cfg(test)]
use incin_core::prelude::DTypeId;

/// Bytes needed to hold `elements` values of `dtype`.
///
/// The multiplication is checked, and `dtype` — not the caller — decides the
/// width, so a block-quantized allocation is sized by its block encoding rather
/// than by its scalar element width.
pub(crate) fn byte_len(
    dtype: impl Into<DTypeDescriptor>,
    elements: usize,
    operation: OperationKind,
) -> Result<usize> {
    Ok(dtype.into().size_bytes(elements, operation)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_dtypes_size_by_element_width() {
        assert_eq!(
            byte_len(DTypeId::F32, 64, OperationKind::Storage).unwrap(),
            256
        );
        assert_eq!(
            byte_len(DTypeId::F64, 64, OperationKind::Storage).unwrap(),
            512
        );
        assert_eq!(
            byte_len(DTypeId::U8, 64, OperationKind::Storage).unwrap(),
            64
        );
    }

    #[test]
    fn quantized_dtypes_size_by_block_not_element_width() {
        // 64 logical values are two 34-byte blocks, not 64 bytes. The scalar
        // element width would undersize this allocation by 4 bytes.
        assert_eq!(
            byte_len(DTypeId::Q8_0, 64, OperationKind::Storage).unwrap(),
            68
        );
        assert_ne!(
            byte_len(DTypeId::Q8_0, 64, OperationKind::Storage).unwrap(),
            64 * DTypeId::Q8_0.encoding().scalar_bytes().unwrap_or(1)
        );
    }

    #[test]
    fn a_partial_quantized_block_is_rejected() {
        assert!(byte_len(DTypeId::Q8_0, 33, OperationKind::Storage).is_err());
    }

    #[test]
    fn an_overflowing_byte_length_is_reported_not_truncated() {
        assert!(byte_len(DTypeId::F64, usize::MAX, OperationKind::Storage).is_err());
    }

    #[test]
    fn an_empty_allocation_is_zero_bytes() {
        assert_eq!(
            byte_len(DTypeId::F32, 0, OperationKind::Storage).unwrap(),
            0
        );
        assert_eq!(
            byte_len(DTypeId::Q8_0, 0, OperationKind::Storage).unwrap(),
            0
        );
    }
}
