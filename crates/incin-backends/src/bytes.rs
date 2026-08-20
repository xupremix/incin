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

use incin_core::error::Result;
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use incin_core::tensor::dtype::DTypeDescriptor;

#[cfg(all(test, any(feature = "cuda", feature = "wgpu")))]
use incin_core::tensor::dtype::DTypeId;

/// Bytes needed to hold `elements` values of `dtype`.
///
/// The multiplication is checked, and `dtype` - not the caller - decides the
/// width, so a block-quantized allocation is sized by its block encoding rather
/// than by its scalar element width.
///
/// Only CUDA and WGPU size a device allocation by explicit byte length; the
/// module itself compiles for `cpu` too, for [`checked_numel`] below.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) fn byte_len(
    dtype: impl Into<DTypeDescriptor>,
    elements: usize,
    operation: OperationKind,
) -> Result<usize> {
    Ok(dtype.into().size_bytes(elements, operation)?)
}

/// Total element count of `shape`, i.e. the product of all dims, via
/// `checked_mul` rather than a bare `.iter().product()`.
///
/// A crafted or accidentally huge user-supplied shape can otherwise overflow
/// `usize` in release builds (overflow checks are off by default) and
/// silently wrap to a small number, undersizing the buffer allocated for it
/// while later stride-based indexing (computed from the same,
/// differently-wrapped shape) reads or writes past the end of it. Every
/// backend needs this same check before it trusts a shape's element count,
/// so this is the one implementation the crate calls, the same way
/// [`byte_len`] is the one implementation for a byte count.
pub(crate) fn checked_numel(shape: &[usize]) -> Result<usize> {
    ShapeBuf::from_slice(shape)
        .checked_numel(OperationKind::Storage)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numel_is_the_checked_product_of_the_dims() {
        assert_eq!(checked_numel(&[2, 3, 4]).unwrap(), 24);
        assert_eq!(checked_numel(&[usize::MAX, 0]).unwrap(), 0);
        assert!(checked_numel(&[usize::MAX, 2]).is_err());
    }
}

#[cfg(all(test, any(feature = "cuda", feature = "wgpu")))]
mod byte_len_tests {
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
