//! `SHP-007`: which operand shapes matrix multiplication relates, and what it
//! checks once it has related them.
//!
//! Two separate obligations, and the file is split along them. The type system
//! decides whether a pair may be multiplied at all; where it cannot settle the
//! contraction — a runtime axis, or a `dim!` name that carries its size at
//! runtime — the check has to happen on the values, exactly once, and produce
//! a diagnostic rather than a wrong shape.

use incin_core::prelude::{Axis, MatMulShape, OperationKind, Shape};
use typenum::{U2, U3, U4, U5};

incin_core::dim!(Batch, Contract);

/// A shape's runtime field, built from the dimensions a test wants.
fn field<S: Shape>(dims: &[usize]) -> S::Field {
    S::from_dyn(dims).expect("test dimensions must match the shape type")
}

/// The dimensions `L` multiplied by `R` resolves to, with the output type
/// pinned to `Expected`.
fn resolved<L, R, Expected>(lhs: &[usize], rhs: &[usize]) -> Vec<usize>
where
    L: MatMulShape<R, Output = Expected> + Shape,
    R: Shape,
    Expected: incin_core::prelude::DynShape,
{
    let out = L::output_shape(&field::<L>(lhs), &field::<R>(rhs)).expect("these operands multiply");
    Expected::dims(&out).as_ref().to_vec()
}

// -- which pairs the type system relates ----------------------------------

#[test]
fn a_runtime_contraction_axis_has_a_rule_at_all() {
    // Before SHP-007 the five rank-2 impls each required the contraction axis
    // to be the *identical* type on both sides, so a contraction nobody can
    // settle statically simply had no rule. The pair did not fail to multiply;
    // it failed to compile.
    assert_eq!(
        resolved::<(U2, usize), (usize, U4), (U2, U4)>(&[2, 3], &[3, 4]),
        vec![2, 4]
    );
}

#[test]
fn the_two_sides_may_spell_the_contraction_differently() {
    assert_eq!(
        resolved::<(U2, U3), (usize, U4), (U2, U4)>(&[2, 3], &[3, 4]),
        vec![2, 4]
    );
    assert_eq!(
        resolved::<(U2, usize), (U3, U4), (U2, U4)>(&[2, 3], &[3, 4]),
        vec![2, 4]
    );
}

#[test]
fn the_outer_axes_keep_whatever_their_own_operand_proved() {
    // `M` and `N` pass through from the operands that named them, so mixing a
    // runtime contraction into the middle does not degrade them. The
    // contraction axis is the only one the two operands have to agree about,
    // and it does not survive into the output.
    assert_eq!(
        resolved::<(usize, U3), (U3, U4), (usize, U4)>(&[7, 3], &[3, 4]),
        vec![7, 4]
    );
    assert_eq!(
        resolved::<(U2, U3), (U3, usize), (U2, usize)>(&[2, 3], &[3, 9]),
        vec![2, 9]
    );
}

// -- what is checked once the pair is related -----------------------------

#[test]
fn a_runtime_contraction_that_disagrees_is_reported() {
    let error = <(U2, usize) as MatMulShape<(usize, U4)>>::output_shape(
        &field::<(U2, usize)>(&[2, 3]),
        &field::<(usize, U4)>(&[5, 4]),
    )
    .expect_err("3 and 5 do not contract");

    assert_eq!(error.operation(), OperationKind::MatMul);
    assert_eq!(error.axis(), Some(Axis::Named("k")));
}

#[test]
fn a_batched_contraction_that_disagrees_is_reported() {
    // This one was silently wrong. `Contract` is one type, so the pair
    // typechecks, but the name carries a runtime size and the two operands may
    // hold different ones. The batched impls never checked it, and the
    // contraction axis does not appear in the output, so nothing downstream
    // could catch it either: the rule returned Ok with a shape of (2, 3, 4)
    // for a multiplication that cannot happen.
    let error = <(U2, U3, Contract) as MatMulShape<(U2, Contract, U4)>>::output_shape(
        &field::<(U2, U3, Contract)>(&[2, 3, 5]),
        &field::<(U2, Contract, U4)>(&[2, 7, 4]),
    )
    .expect_err("5 and 7 do not contract");

    assert_eq!(error.operation(), OperationKind::MatMul);
    assert_eq!(error.axis(), Some(Axis::Named("k")));
}

#[test]
fn a_batch_axis_that_disagrees_is_reported_at_its_own_index() {
    // The same hole one axis over. The output takes its batch axes from the
    // left operand, so a disagreement used to be resolved silently in the
    // left's favour.
    let error = <(Batch, U3, U2) as MatMulShape<(Batch, U2, U4)>>::output_shape(
        &field::<(Batch, U3, U2)>(&[6, 3, 2]),
        &field::<(Batch, U2, U4)>(&[8, 2, 4]),
    )
    .expect_err("a `Batch` cannot be both 6 and 8");

    assert_eq!(error.operation(), OperationKind::MatMul);
    assert_eq!(error.axis(), Some(Axis::Index(0)));
}

#[test]
fn a_batched_multiplication_that_agrees_still_goes_through() {
    // The guard against the checks above rejecting the good case as well.
    assert_eq!(
        resolved::<(Batch, U3, U2), (Batch, U2, U4), (Batch, U3, U4)>(&[6, 3, 2], &[6, 2, 4]),
        vec![6, 3, 4]
    );
}

#[test]
fn an_unbatched_operand_broadcasts_across_the_batch_and_is_still_checked() {
    // (B, M, K) against a plain (K, N): one shared axis, the contraction, and
    // this impl did not check it either.
    assert_eq!(
        resolved::<(U2, U3, Contract), (Contract, U4), (U2, U3, U4)>(&[2, 3, 5], &[5, 4]),
        vec![2, 3, 4]
    );

    let error = <(U2, U3, Contract) as MatMulShape<(Contract, U4)>>::output_shape(
        &field::<(U2, U3, Contract)>(&[2, 3, 5]),
        &field::<(Contract, U4)>(&[7, 4]),
    )
    .expect_err("5 and 7 do not contract");

    assert_eq!(error.axis(), Some(Axis::Named("k")));
}

#[test]
fn the_rank_four_by_rank_two_convention_is_unchanged() {
    // The flattened-batch spelling SHP-004 preserved deliberately: a rank-4
    // left operand against a rank-2 right one contracts the trailing axis and
    // leaves the three leading ones alone. Pinned here because the rank-2
    // collapse rewrote the impl next to it.
    assert_eq!(
        resolved::<(U2, U2, U3, U5), (U5, U4), (U2, U2, U3, U4)>(&[2, 2, 3, 5], &[5, 4]),
        vec![2, 2, 3, 4]
    );
}
