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
    Axis, BroadcastShape, DimensionConstraint, Dyn, MatMulShape, OperationKind, Shape, ShapeBuf,
    ShapeError,
};
extern crate incin_core as incin;
use incin_macros::s;

type RuntimeMatrix = s![dyn, dyn];
type RuntimeStatic = s![dyn, 3];
type Static23 = s![2, 3];

#[test]
fn structural_shape_resolution_rejects_bad_runtime_arguments_without_panicking() {
    let error = <RuntimeStatic as Shape>::try_from_dims(&[4, 4]).unwrap_err();
    assert!(matches!(
        error,
        ShapeError::TargetShapeRejected {
            operation: OperationKind::Storage,
            rank: 2
        }
    ));
}

// --- broadcast ----------------------------------------------------------

#[test]
fn broadcast_reports_the_axis_that_disagreed() {
    // Two runtime axes, neither 1, that do not match. Before SHP-004 this was
    // an `assert!` inside `checked_broadcast_dim`; decision D-013 requires it
    // to become a `Result` rather than be deleted, because it is the only
    // guard against two identically-typed named dims with different sizes.
    let err = <Dyn as BroadcastShape<Dyn>>::output_shape(
        &incin_core::shapes::ShapeBuf::from_slice(&[3]),
        &incin_core::shapes::ShapeBuf::from_slice(&[4]),
    )
    .unwrap_err();
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
    let out = <Dyn as BroadcastShape<Dyn>>::output_shape(
        &incin_core::shapes::ShapeBuf::from_slice(&[3, 1]),
        &incin_core::shapes::ShapeBuf::from_slice(&[1, 4]),
    )
    .unwrap();
    assert_eq!(out.as_ref(), &[3, 4]);

    let out = <Dyn as BroadcastShape<Dyn>>::output_shape(
        &incin_core::shapes::ShapeBuf::from_slice(&[5]),
        &incin_core::shapes::ShapeBuf::from_slice(&[5]),
    )
    .unwrap();
    assert_eq!(out.as_ref(), &[5]);
}

#[test]
fn broadcasting_a_size_one_axis_against_a_size_zero_one_yields_zero() {
    // The regression this pins: the rule used to be `lhs.max(rhs)`, which
    // answers 1 here. NumPy's rule is "take the side that isn't 1", which
    // answers 0 — an axis with no elements cannot gain one by being
    // broadcast against.
    let out = <Dyn as BroadcastShape<Dyn>>::output_shape(
        &incin_core::shapes::ShapeBuf::from_slice(&[1]),
        &incin_core::shapes::ShapeBuf::from_slice(&[0]),
    )
    .unwrap();
    assert_eq!(
        out.as_ref(),
        &[0],
        "a size-1 axis broadcast against 0 must yield 0"
    );

    let out = <Dyn as BroadcastShape<Dyn>>::output_shape(
        &incin_core::shapes::ShapeBuf::from_slice(&[0]),
        &incin_core::shapes::ShapeBuf::from_slice(&[1]),
    )
    .unwrap();
    assert_eq!(out.as_ref(), &[0]);
}

#[test]
fn broadcast_right_aligns_operands_of_different_rank() {
    // The shorter operand's missing axes are implicit 1s. This used to be a
    // four-armed match whose fourth arm was `unreachable!`; treating a missing
    // axis as 1 removes the arm rather than asserting it away.
    let out = <Dyn as BroadcastShape<Dyn>>::output_shape(
        &ShapeBuf::from_slice(&[5]),
        &ShapeBuf::from_slice(&[3, 5]),
    )
    .unwrap();
    assert_eq!(out.as_ref(), &[3, 5]);

    let out = <Dyn as BroadcastShape<Dyn>>::output_shape(
        &ShapeBuf::from_slice(&[3, 1]),
        &ShapeBuf::from_slice(&[4]),
    )
    .unwrap();
    assert_eq!(out.as_ref(), &[3, 4]);
}

#[test]
fn dyn_broadcast_reports_a_right_aligned_axis_index() {
    let err = <Dyn as BroadcastShape<Dyn>>::output_shape(
        &ShapeBuf::from_slice(&[2, 3, 4]),
        &ShapeBuf::from_slice(&[2, 9, 4]),
    )
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
    let err = <RuntimeMatrix as MatMulShape<RuntimeMatrix>>::output_shape(
        &ShapeBuf::from_slice(&[2, 3]),
        &ShapeBuf::from_slice(&[4, 5]),
    )
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
    let out = <RuntimeMatrix as MatMulShape<RuntimeMatrix>>::output_shape(
        &ShapeBuf::from_slice(&[2, 3]),
        &ShapeBuf::from_slice(&[3, 5]),
    )
    .unwrap();
    assert_eq!(out.as_ref(), &[2, 5]);
}

#[test]
fn dyn_matmul_no_longer_returns_a_sentinel_empty_shape() {
    // Every unmatched rank combination used to fall through to `vec![]` — the
    // scalar shape — which then propagated as a real answer.
    let err = <Dyn as MatMulShape<Dyn>>::output_shape(
        &ShapeBuf::from_slice(&[7]),
        &ShapeBuf::from_slice(&[7, 3]),
    )
    .unwrap_err();
    assert!(
        matches!(err, ShapeError::RankMismatch { actual: 1, .. }),
        "unexpected error {err}"
    );

    let err =
        <Dyn as MatMulShape<Dyn>>::output_shape(&ShapeBuf::from_slice(&[2, 3]), &ShapeBuf::SCALAR)
            .unwrap_err();
    assert!(matches!(err, ShapeError::RankMismatch { actual: 0, .. }));
}

#[test]
fn a_matrix_times_a_vector_is_a_vector() {
    // `[m, k] x [k]` is `[m]`. This used to return `vec![]`, a scalar.
    let out = <Dyn as MatMulShape<Dyn>>::output_shape(
        &ShapeBuf::from_slice(&[2, 3]),
        &ShapeBuf::from_slice(&[3]),
    )
    .unwrap();
    assert_eq!(out.as_ref(), &[2]);

    let err = <Dyn as MatMulShape<Dyn>>::output_shape(
        &ShapeBuf::from_slice(&[2, 3]),
        &ShapeBuf::from_slice(&[9]),
    )
    .unwrap_err();
    assert_eq!(err.axis(), Some(Axis::Named("k")));
}

#[test]
fn dyn_matmul_contracts_batched_operands() {
    let out = <Dyn as MatMulShape<Dyn>>::output_shape(
        &ShapeBuf::from_slice(&[8, 2, 3]),
        &ShapeBuf::from_slice(&[8, 3, 5]),
    )
    .unwrap();
    assert_eq!(out.as_ref(), &[8, 2, 5]);

    let err = <Dyn as MatMulShape<Dyn>>::output_shape(
        &ShapeBuf::from_slice(&[8, 2, 3]),
        &ShapeBuf::from_slice(&[8, 9, 5]),
    )
    .unwrap_err();
    assert_eq!(err.axis(), Some(Axis::Named("k")));
}

#[test]
fn dynamic_matmul_requires_explicit_flattening() {
    // Matmul never silently folds a higher-rank lhs into a matrix. Callers
    // must make that layout change explicit before the contraction rule runs.
    let err = <Dyn as MatMulShape<Dyn>>::output_shape(
        &ShapeBuf::from_slice(&[4, 4, 52, 52]),
        &ShapeBuf::from_slice(&[10816, 10]),
    )
    .unwrap_err();
    assert_eq!(err.axis(), Some(Axis::Named("k")));

    let out = <Dyn as MatMulShape<Dyn>>::output_shape(
        &ShapeBuf::from_slice(&[4, 10816]),
        &ShapeBuf::from_slice(&[10816, 10]),
    )
    .unwrap();
    assert_eq!(out.as_ref(), &[4, 10]);
}

// --- the removed chain --------------------------------------------------

#[test]
fn a_target_shape_that_rejects_its_dims_reports_rather_than_panics() {
    // `shape_buf_from_dims` is the checked replacement for the round-trip. A
    // fully static target cannot accept dims that disagree with it, and now
    // says so instead of unwrapping `None`.
    use incin_core::prelude::shape_buf_from_dims;

    let ok = shape_buf_from_dims::<Static23>(OperationKind::Reshape, &[2, 3]);
    assert!(ok.is_ok());

    let wrong_dim = shape_buf_from_dims::<Static23>(OperationKind::Reshape, &[2, 4]).unwrap_err();
    assert_eq!(
        wrong_dim,
        ShapeError::TargetShapeRejected {
            operation: OperationKind::Reshape,
            rank: 2
        }
    );

    let wrong_rank =
        shape_buf_from_dims::<Static23>(OperationKind::Reshape, &[2, 3, 1]).unwrap_err();
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

    fn caller() -> Result<ShapeBuf, Error> {
        Ok(<RuntimeMatrix as MatMulShape<RuntimeMatrix>>::output_shape(
            &ShapeBuf::from_slice(&[2, 3]),
            &ShapeBuf::from_slice(&[4, 5]),
        )?)
    }

    let err = caller().unwrap_err();
    assert!(matches!(err, Error::Shape(_)), "got {err}");
}
