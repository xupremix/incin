//! Pure shape/stride math with zero dependencies on autograd or dtype.
//!
//! These functions are the foundation every later `CpuStorage` view
//! operation (`reshape`/`transpose`/`broadcast_as`) is built on. They must be
//! correct and standalone-tested before any storage/tape code touches them.

use incin_core::prelude::Error;
use incin_core::prelude::Result;

/// Compute row-major (C-contiguous) strides for `shape`.
///
/// The last dimension has stride 1; each earlier dimension's stride is the
/// product of all later dimensions' sizes. An empty shape (scalar / 0-d)
/// returns an empty stride vector.
pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        // checked_mul instead of a bare `*`: in release builds (overflow
        // checks off by default) an unchecked multiply here can silently
        // wrap to a small stride, which downstream code then uses to index
        // into a buffer sized from the same (also-wrapped) element count —
        // an out-of-bounds read/write path (C-5). Panic loudly instead.
        strides[i] = strides[i + 1].checked_mul(shape[i + 1]).unwrap_or_else(|| {
            panic!(
                "shape overflow computing strides: stride {} * dim {} overflows usize (shape: {:?})",
                strides[i + 1],
                shape[i + 1],
                shape
            )
        });
    }
    strides
}

/// Returns true if `strides` matches the contiguous (row-major) strides for
/// `shape`.
pub(crate) fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    strides == contiguous_strides(shape)
}

/// Total element count of `shape`, i.e. the product of all dims — but via
/// `checked_mul` instead of a bare `.iter().product()`. A crafted or
/// accidentally-huge user-supplied shape can otherwise overflow `usize` in
/// release builds (overflow checks are off by default) and silently wrap to
/// a small number, undersizing the `Vec` allocated for it while later
/// stride-based indexing (computed from the same, differently-wrapped shape)
/// reads/writes past the end of that undersized buffer (C-5).
pub(crate) fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            Error::Msg(format!(
                "shape overflow computing element count: shape {:?} overflows usize",
                shape
            ))
        })
    })
}

/// Resolve the broadcast-compatible output shape of two input shapes, using
/// right-aligned numpy/Candle-style broadcast rules.
///
/// For each axis (right-aligned), a missing dimension (shorter shape) is
/// treated as `1`. If the two dims differ and neither is `1`, returns
/// `Error::ShapeMismatch`.
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut out = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let da = *a.get(a.len().wrapping_sub(max_len - i)).unwrap_or(&1usize);
        let db = *b.get(b.len().wrapping_sub(max_len - i)).unwrap_or(&1usize);
        if da != db && da != 1 && db != 1 {
            return Err(Error::ShapeMismatch {
                op: "broadcast_shape",
                expected: a.to_vec(),
                got: b.to_vec(),
                msg: format!(
                    "cannot broadcast shapes {:?} and {:?}: incompatible at right-aligned axis {}",
                    a, b, i
                ),
            });
        }
        out.push(da.max(db));
    }
    Ok(out)
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;

    #[test]
    /// `contiguous_strides_row_major`.
    fn contiguous_strides_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    /// `contiguous_strides_scalar`.
    fn contiguous_strides_scalar() {
        let empty: &[usize] = &[];
        assert_eq!(contiguous_strides(empty), Vec::<usize>::new());
    }

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
    /// `broadcast_shape_both_expand`.
    fn broadcast_shape_both_expand() {
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    /// `broadcast_shape_right_aligned_leading_dim_insert`.
    fn broadcast_shape_right_aligned_leading_dim_insert() {
        assert_eq!(broadcast_shape(&[5], &[3, 5]).unwrap(), vec![3, 5]);
    }

    #[test]
    /// `broadcast_shape_incompatible_errors`.
    fn broadcast_shape_incompatible_errors() {
        let result = broadcast_shape(&[3, 4], &[3, 5]);
        assert!(result.is_err());
    }

    #[test]
    /// `broadcast_shape_scalar_broadcast`.
    fn broadcast_shape_scalar_broadcast() {
        let empty: &[usize] = &[];
        assert_eq!(broadcast_shape(empty, &[3, 4]).unwrap(), vec![3, 4]);
    }
}
