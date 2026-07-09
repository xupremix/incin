//! Pure shape/stride math with zero dependencies on autograd or dtype.
//!
//! These functions are the foundation every later `NativeStorage` view
//! operation (`reshape`/`transpose`/`broadcast_as`) is built on. They must be
//! correct and standalone-tested before any storage/tape code touches them.

use kindle_core::err::Error;
use kindle_core::prelude::Result;

/// Compute row-major (C-contiguous) strides for `shape`.
///
/// The last dimension has stride 1; each earlier dimension's stride is the
/// product of all later dimensions' sizes. An empty shape (scalar / 0-d)
/// returns an empty stride vector.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Returns true if `strides` matches the contiguous (row-major) strides for
/// `shape`.
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    strides == contiguous_strides(shape)
}

/// Resolve the broadcast-compatible output shape of two input shapes, using
/// right-aligned numpy/Candle-style broadcast rules.
///
/// For each axis (right-aligned), a missing dimension (shorter shape) is
/// treated as `1`. If the two dims differ and neither is `1`, returns
/// `Error::ShapeMismatch`.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
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
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn contiguous_strides_scalar() {
        let empty: &[usize] = &[];
        assert_eq!(contiguous_strides(empty), Vec::<usize>::new());
    }

    #[test]
    fn is_contiguous_true_for_fresh_strides() {
        let shape = vec![2, 3, 4];
        let strides = contiguous_strides(&shape);
        assert!(is_contiguous(&shape, &strides));
    }

    #[test]
    fn is_contiguous_false_for_transposed_strides() {
        // [2,3] contiguous strides are [3,1]; a transposed view swaps shape
        // to [3,2] but keeps strides [1,3] (permuted, non-contiguous).
        let shape = vec![3, 2];
        let strides = vec![1, 3];
        assert!(!is_contiguous(&shape, &strides));
    }

    #[test]
    fn broadcast_shape_both_expand() {
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn broadcast_shape_right_aligned_leading_dim_insert() {
        assert_eq!(broadcast_shape(&[5], &[3, 5]).unwrap(), vec![3, 5]);
    }

    #[test]
    fn broadcast_shape_incompatible_errors() {
        let result = broadcast_shape(&[3, 4], &[3, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn broadcast_shape_scalar_broadcast() {
        let empty: &[usize] = &[];
        assert_eq!(broadcast_shape(empty, &[3, 4]).unwrap(), vec![3, 4]);
    }
}
