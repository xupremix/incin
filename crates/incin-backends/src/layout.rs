//! Shape and index arithmetic that belongs to no particular backend.
//!
//! Every backend that walks a tensor element by element needs the same three
//! answers: what the row-major strides of a shape are, what shape two operands
//! broadcast to, and how to step a multi-index. None of that involves a device,
//! a buffer, or a dtype.
//!
//! These lived in `crate::cpu::stride` and `crate::cpu::storage` until the
//! accelerators had accumulated 95 calls into them. That made the CPU backend a
//! build dependency of backends that share none of its code: `lib.rs` carried
//! `#[cfg(any(feature = "cpu", feature = "cuda"))] pub mod cpu;` to keep the
//! CUDA build alive, and `--features metal` and `--features wgpu` did not
//! compile at all without `cpu` also enabled. This module is ungated because
//! nothing in it has a reason to be gated, and `cpu` re-exports it so its own
//! call sites read as they always did.

// `alloc` rather than the `std` prelude, because this module is ungated while
// every backend feature implies `std`: a bare `incin-backends` with no features
// is `no_std`, and is the one configuration in which `Vec` is not already here.
use alloc::string::ToString;
use alloc::vec::Vec;

use incin_core::error::{Error, Result};
use incin_core::shapes::broadcast::broadcast_dim_slices;
#[cfg(any(feature = "cpu", feature = "cuda", feature = "metal"))]
use incin_core::shapes::error::OperationKind;
#[cfg(any(feature = "cpu", feature = "cuda", feature = "metal"))]
use incin_core::shapes::{ShapeBuf, StrideBuf};

/// Compute row-major (C-contiguous) strides for `shape`.
///
/// The last dimension has stride 1; each earlier dimension's stride is the
/// product of all later dimensions' sizes. An empty shape (scalar / 0-d)
/// returns an empty stride vector.
#[cfg(any(feature = "cpu", feature = "cuda", feature = "metal"))]
pub(crate) fn checked_contiguous_strides(shape: &[usize]) -> Result<StrideBuf> {
    // Returned as a `StrideBuf`, which stores rank 8 and below inline. Copying
    // it into a `Vec` cost an allocation per storage construction, which is one
    // per operation, to reach a value the caller then hands straight to
    // `TensorMeta` - where it becomes a `StrideBuf` again.
    incin_core::shapes::StrideBuf::contiguous_for(
        &ShapeBuf::from_slice(shape),
        OperationKind::Storage,
    )
    .map_err(Into::into)
}

/// [`checked_contiguous_strides`] for a shape that has already been validated.
///
/// Panics where the checked form returns an error, so it is only for shapes
/// that some constructor has already accepted.
#[cfg(any(feature = "cpu", feature = "cuda", feature = "metal"))]
pub(crate) fn contiguous_strides(shape: &[usize]) -> StrideBuf {
    checked_contiguous_strides(shape)
        .expect("validated storage shape must have representable contiguous strides")
}

/// Resolve the broadcast-compatible output shape of two input shapes, using
/// right-aligned numpy/Candle-style broadcast rules.
///
/// For each axis (right-aligned), a missing dimension (shorter shape) is
/// treated as `1`. If the two dims differ and neither is `1`, returns
/// `Error::ShapeMismatch`.
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    broadcast_dim_slices(a, b).map_err(|error| Error::ShapeMismatch {
        op: "broadcast_shape",
        expected: a.to_vec(),
        got: b.to_vec(),
        msg: error.to_string(),
    })
}

/// Increment a row-major multi-index in place (odometer-style), matching the
/// iteration order [`contiguous_strides`] assumes.
#[cfg(any(feature = "cpu", feature = "cuda", feature = "metal"))]
pub(crate) fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for i in (0..idx.len()).rev() {
        idx[i] += 1;
        if idx[i] < shape[i] {
            return;
        }
        idx[i] = 0;
    }
}

#[cfg(test)]
/// `tests`.
#[cfg(any(feature = "cpu", feature = "cuda", feature = "metal"))]
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
    fn broadcast_shape_preserves_zero_extent() {
        assert_eq!(broadcast_shape(&[1], &[0]).unwrap(), vec![0]);
        assert_eq!(broadcast_shape(&[0], &[1]).unwrap(), vec![0]);
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

    #[test]
    /// `increment_index_carries_across_axes`.
    fn increment_index_carries_across_axes() {
        let shape = [2, 3];
        let mut idx = [0, 2];
        increment_index(&mut idx, &shape);
        assert_eq!(idx, [1, 0]);
    }

    #[test]
    /// `increment_index_wraps_at_the_end`.
    fn increment_index_wraps_at_the_end() {
        // The odometer has no "done" signal: callers step it exactly `numel`
        // times, so the final increment rolls back to all-zeros rather than
        // reporting exhaustion.
        let shape = [2, 2];
        let mut idx = [1, 1];
        increment_index(&mut idx, &shape);
        assert_eq!(idx, [0, 0]);
    }
}
