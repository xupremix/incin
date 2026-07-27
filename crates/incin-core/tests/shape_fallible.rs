//! Fallible broadcast, reshape, flatten, and matmul geometry (`SHP-004`).
//!
//! `SHP-001` inventoried 39 live `from_dyn(&dims).unwrap()` sites: a shape rule
//! would erase a known-rank output to a `Vec<usize>`, re-parse it, and assert
//! the round-trip succeeded. The assertion was a proof obligation no type
//! stated and no test covered. This file pins the replacement — every one of
//! those paths now returns a `ShapeError` that names the operation, and where
//! the axis is known, the axis.
//!
//! It also pins the two *sentinel* outputs removed alongside them, which are
//! worse than the panics: a wrong shape propagates silently.

use incin_core::prelude::{
    Axis, BroadcastShape, DimensionConstraint, Dyn, MatMulShape, OperationKind, ShapeError,
};

// --- broadcast ----------------------------------------------------------

#[test]
fn broadcast_reports_the_axis_that_disagreed() {
    // Two runtime axes, neither 1, that do not match. Before SHP-004 this was
    // an `assert!` inside `checked_broadcast_dim`; decision D-013 requires it
    // to become a `Result` rather than be deleted, because it is the only
    // guard against two identically-typed named dims with different sizes.
    let err = <(usize,) as BroadcastShape<(usize,)>>::output_shape(&(3,), &(4,)).unwrap_err();
    assert_eq!(
        err,
        ShapeError::DimensionMismatch {
            operation: OperationKind::Broadcast,
            axis: Axis::Index(0),
            lhs: 3,
            rhs: 4,
            constraint: DimensionConstraint::Broadcastable,
        }
    );
}

#[test]
fn broadcast_accepts_the_numpy_compatible_cases() {
    let out = <Dyn as BroadcastShape<Dyn>>::output_shape(&vec![3, 1], &vec![1, 4]).unwrap();
    assert_eq!(out, vec![3, 4]);

    let out = <(usize,) as BroadcastShape<(usize,)>>::output_shape(&(5,), &(5,)).unwrap();
    assert_eq!(out, (5,));
}

#[test]
fn broadcasting_a_size_one_axis_against_a_size_zero_one_yields_zero() {
    // The regression this pins: the rule used to be `lhs.max(rhs)`, which
    // answers 1 here. NumPy's rule is "take the side that isn't 1", which
    // answers 0 — an axis with no elements cannot gain one by being
    // broadcast against.
    let out = <(usize,) as BroadcastShape<(usize,)>>::output_shape(&(1,), &(0,)).unwrap();
    assert_eq!(out, (0,), "a size-1 axis broadcast against 0 must yield 0");

    let out = <(usize,) as BroadcastShape<(usize,)>>::output_shape(&(0,), &(1,)).unwrap();
    assert_eq!(out, (0,));
}

#[test]
fn broadcast_right_aligns_operands_of_different_rank() {
    // The shorter operand's missing axes are implicit 1s. This used to be a
    // four-armed match whose fourth arm was `unreachable!`; treating a missing
    // axis as 1 removes the arm rather than asserting it away.
    let out = <Dyn as BroadcastShape<Dyn>>::output_shape(&vec![5], &vec![3, 5]).unwrap();
    assert_eq!(out, vec![3, 5]);

    let out = <Dyn as BroadcastShape<Dyn>>::output_shape(&vec![3, 1], &vec![4]).unwrap();
    assert_eq!(out, vec![3, 4]);
}

#[test]
fn dyn_broadcast_reports_a_right_aligned_axis_index() {
    let err = <Dyn as BroadcastShape<Dyn>>::output_shape(&vec![2, 3, 4], &vec![2, 9, 4])
        .unwrap_err();
    assert_eq!(err.operation(), OperationKind::Broadcast);
    assert_eq!(err.axis(), Some(Axis::Index(1)));
}

// --- matmul -------------------------------------------------------------

#[test]
fn matmul_rejects_a_disagreeing_contraction() {
    // Nothing checked this before SHP-004: `output_shape` returned
    // `(lhs.0, rhs.1)` without ever looking at K, so a mismatch produced a
    // confidently wrong output shape.
    let err = <(usize, usize) as MatMulShape<(usize, usize)>>::output_shape(&(2, 3), &(4, 5))
        .unwrap_err();
    assert_eq!(
        err,
        ShapeError::DimensionMismatch {
            operation: OperationKind::MatMul,
            axis: Axis::Named("k"),
            lhs: 3,
            rhs: 4,
            constraint: DimensionConstraint::Equal,
        }
    );
    assert_eq!(
        err.to_string(),
        "matmul: axis 'k' mismatch: 3 vs 4, which must be equal"
    );
}

#[test]
fn matmul_accepts_an_agreeing_contraction() {
    let out =
        <(usize, usize) as MatMulShape<(usize, usize)>>::output_shape(&(2, 3), &(3, 5)).unwrap();
    assert_eq!(out, (2, 5));
}

#[test]
fn dyn_matmul_no_longer_returns_a_sentinel_empty_shape() {
    // Every unmatched rank combination used to fall through to `vec![]` — the
    // scalar shape — which then propagated as a real answer.
    let err = <Dyn as MatMulShape<Dyn>>::output_shape(&vec![7], &vec![7, 3]).unwrap_err();
    assert!(
        matches!(err, ShapeError::RankMismatch { actual: 1, .. }),
        "unexpected error {err}"
    );

    let err = <Dyn as MatMulShape<Dyn>>::output_shape(&vec![2, 3], &vec![]).unwrap_err();
    assert!(matches!(err, ShapeError::RankMismatch { actual: 0, .. }));
}

#[test]
fn a_matrix_times_a_vector_is_a_vector() {
    // `[m, k] x [k]` is `[m]`. This used to return `vec![]`, a scalar.
    let out = <Dyn as MatMulShape<Dyn>>::output_shape(&vec![2, 3], &vec![3]).unwrap();
    assert_eq!(out, vec![2]);

    let err = <Dyn as MatMulShape<Dyn>>::output_shape(&vec![2, 3], &vec![9]).unwrap_err();
    assert_eq!(err.axis(), Some(Axis::Named("k")));
}

#[test]
fn dyn_matmul_contracts_batched_operands() {
    let out = <Dyn as MatMulShape<Dyn>>::output_shape(&vec![8, 2, 3], &vec![8, 3, 5]).unwrap();
    assert_eq!(out, vec![8, 2, 5]);

    let err = <Dyn as MatMulShape<Dyn>>::output_shape(&vec![8, 2, 3], &vec![8, 9, 5]).unwrap_err();
    assert_eq!(err.axis(), Some(Axis::Named("k")));
}

#[test]
fn the_flattened_batch_convention_is_preserved() {
    // A rank-4 lhs against a rank-2 rhs is the existing "flattened batch"
    // path: `[N, C, H, W] x [C*H*W, out] -> [N, out]`. Its contracted extents
    // deliberately do not match axis-for-axis, so it keeps its own arm and is
    // not routed through the contraction check.
    let out = <Dyn as MatMulShape<Dyn>>::output_shape(&vec![4, 4, 52, 52], &vec![10816, 10])
        .unwrap();
    assert_eq!(out, vec![4, 10]);
}

// --- the removed chain --------------------------------------------------

#[test]
fn a_target_shape_that_rejects_its_dims_reports_rather_than_panics() {
    // `field_from_dims` is the checked replacement for the round-trip. A
    // fully static target cannot accept dims that disagree with it, and now
    // says so instead of unwrapping `None`.
    use incin_core::prelude::field_from_dims;
    use incin_core::typenum::{U2, U3};

    let ok = field_from_dims::<(U2, U3)>(OperationKind::Reshape, &[2, 3]);
    assert!(ok.is_ok());

    let wrong_dim = field_from_dims::<(U2, U3)>(OperationKind::Reshape, &[2, 4]).unwrap_err();
    assert_eq!(
        wrong_dim,
        ShapeError::TargetShapeRejected {
            operation: OperationKind::Reshape,
            rank: 2
        }
    );

    let wrong_rank = field_from_dims::<(U2, U3)>(OperationKind::Reshape, &[2, 3, 1]).unwrap_err();
    assert_eq!(
        wrong_rank,
        ShapeError::TargetShapeRejected {
            operation: OperationKind::Reshape,
            rank: 3
        }
    );
    assert_eq!(
        wrong_rank.to_string(),
        "reshape: the computed rank-3 shape does not fit the target shape type"
    );
}

#[test]
fn every_fallible_shape_path_routes_into_the_crate_error() {
    // The whole point of making these fallible is that a caller can use `?`.
    use incin_core::prelude::Error;

    fn caller() -> Result<(usize, usize), Error> {
        Ok(<(usize, usize) as MatMulShape<(usize, usize)>>::output_shape(&(2, 3), &(4, 5))?)
    }

    let err = caller().unwrap_err();
    assert!(matches!(err, Error::Shape(_)), "got {err}");
}
