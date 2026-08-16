//! Pure shape/stride math with zero dependencies on autograd or dtype.
//!
//! These functions are the foundation every later `CpuStorage` view
//! operation (`reshape`/`transpose`/`broadcast_as`) is built on. They must be
//! correct and standalone-tested before any storage/tape code touches them.
//!
//! The parts of that foundation the accelerators also need now live in
//! [`crate::layout`] and are re-exported here; what remains below is the part
//! only `CpuStorage` asks for.

use incin_core::error::Result;
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;

pub(crate) use crate::layout::{broadcast_shape, checked_contiguous_strides, contiguous_strides};

/// Returns true if `strides` matches the contiguous (row-major) strides for
/// `shape`.
///
/// Walks the axes from the fastest-moving one instead of materializing the
/// contiguous strides and comparing. This is called on both operands of every
/// elementwise op, so building a vector to answer a question about a vector
/// that already exists was two heap allocations per operation for nothing
/// (PRF-001). The overflow behaviour differs from `contiguous_strides` on
/// purpose: a shape whose stride product overflows cannot equal any real
/// stride list, so this answers `false` where the constructor panics.
pub(crate) fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expected = 1usize;
    for (&dimension, &stride) in shape.iter().zip(strides).rev() {
        if stride != expected {
            return false;
        }
        let Some(next) = expected.checked_mul(dimension) else {
            return false;
        };
        expected = next;
    }
    true
}

/// Total element count of `shape`, i.e. the product of all dims — but via
/// `checked_mul` instead of a bare `.iter().product()`. A crafted or
/// accidentally-huge user-supplied shape can otherwise overflow `usize` in
/// release builds (overflow checks are off by default) and silently wrap to
/// a small number, undersizing the `Vec` allocated for it while later
/// stride-based indexing (computed from the same, differently-wrapped shape)
/// reads/writes past the end of that undersized buffer (C-5).
pub(crate) fn checked_numel(shape: &[usize]) -> Result<usize> {
    ShapeBuf::from_slice(shape)
        .checked_numel(OperationKind::Storage)
        .map_err(Into::into)
}

/// Reads an element count from shape metadata already accepted by
/// `CpuStorage::try_from_parts`.
///
/// This is not an allocation-boundary validator. New or untrusted dimensions
/// must use [`checked_numel`] and propagate its error before constructing
/// storage.
pub(crate) fn validated_numel(shape: &[usize]) -> usize {
    checked_numel(shape)
        .expect("validated CpuStorage shape must have a representable element count")
}

/// The element count for `shape`, read from `S`'s own const when `S` is fully
/// static instead of walked from `shape` at run time.
///
/// This is only sound because a caller reaches `shape` through a descriptor
/// already checked against `S`: the canonical executor receives the validated
/// runtime shape for `S`, so `S::STATIC_NUMEL` and a fresh product over
/// `shape` cannot disagree unless that checking was skipped, which nothing in
/// this crate does. The `debug_assert_eq!` verifies
/// exactly that on every test run and costs nothing in release, matching the
/// pattern `resolved_output_shape` in `cpu::canonical` already uses for the
/// pointwise family.
pub(crate) fn numel_for_evidence(shape: &[usize], static_numel: Option<usize>) -> Result<usize> {
    if let Some(total) = static_numel {
        debug_assert_eq!(
            checked_numel(shape).ok(),
            Some(total),
            "a statically-known numel must match the shape actually produced",
        );
        return Ok(total);
    }
    checked_numel(shape)
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;

    #[test]
    /// `is_contiguous_true_for_fresh_strides`.
    fn is_contiguous_true_for_fresh_strides() {
        let shape = vec![2, 3, 4];
        let strides = contiguous_strides(&shape);
        assert!(is_contiguous(&shape, &strides));
    }

    #[test]
    /// `is_contiguous_false_for_transposed_strides`.
    fn is_contiguous_false_for_transposed_strides() {
        // [2,3] contiguous strides are [3,1]; a transposed view swaps shape
        // to [3,2] but keeps strides [1,3] (permuted, non-contiguous).
        let shape = vec![3, 2];
        let strides = vec![1, 3];
        assert!(!is_contiguous(&shape, &strides));
    }

    #[test]
    fn numel_for_a_static_shape_reads_the_type_level_constant() {
        let shape = [2usize, 3usize];
        assert_eq!(numel_for_evidence(&shape, Some(6)).unwrap(), 6);
    }

    #[test]
    fn numel_for_a_dynamic_shape_matches_checked_numel() {
        let shape = [2usize, 3usize, 4usize];
        assert_eq!(
            numel_for_evidence(&shape, None).unwrap(),
            checked_numel(&shape).unwrap(),
        );
    }
}
