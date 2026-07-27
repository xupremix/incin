//! `SHP-007`: the axis pairs the type system accepts, and the ones it refuses.
//!
//! This is the compile-pass half of the broadcast suite. Every function here
//! names a concrete `BroadcastShape` `Output`, so the file failing to compile
//! *is* the assertion — a pair that stops typechecking cannot reach the
//! runtime checks below it. The refusals live next door in
//! `tests/compile_fail/`, which is where an absent impl can be asserted.
//!
//! A `trybuild` pass case would prove less than this: it would show the pair
//! compiles without showing that `Output` resolves to the shape it should, and
//! a rule that typechecks while naming the wrong output shape is the failure
//! mode worth catching.
//!
//! Before `SHP-007` the same-rank family required every axis to be the
//! identical type, so the pair at the centre of this file — a tensor against a
//! bias shaped `(U1, C, U1, U1)` — did not typecheck at all.

use incin_core::prelude::{BroadcastShape, Dyn, Shape};
use typenum::{U1, U2, U3, U4, U5};

incin_core::dim!(Batch, Seq);

/// A shape's runtime field, built from the dimensions a test wants.
fn field<S: Shape>(dims: &[usize]) -> S::Field {
    S::from_dyn(dims).expect("test dimensions must match the shape type")
}

/// The dimensions `L` broadcast against `R` resolves to, with the output type
/// pinned to `Expected`.
///
/// Binding `Expected` is the point: it makes the *type* the rule produces part
/// of the assertion rather than trusting whatever it happened to infer.
fn resolved<L, R, Expected>(lhs: &[usize], rhs: &[usize]) -> Vec<usize>
where
    L: BroadcastShape<R, Output = Expected> + Shape,
    R: Shape,
    Expected: incin_core::prelude::DynShape,
{
    let out = L::output_shape(&field::<L>(lhs), &field::<R>(rhs))
        .expect("these operands broadcast");
    Expected::dims(&out).as_ref().to_vec()
}

// -- same rank, one side stretched ---------------------------------------

#[test]
fn a_bias_shaped_like_the_channel_axis_broadcasts_over_a_batch() {
    // The case SHP-007 exists for. `(U1, C, U1, U1)` is how a per-channel bias
    // or a normalization statistic is spelled, and before the per-axis rule it
    // could not meet its own tensor without erasing to `Dyn`.
    type Tensor4 = (U2, U3, U4, U4);
    type Bias = (U1, U3, U1, U1);

    assert_eq!(
        resolved::<Tensor4, Bias, (U2, U3, U4, U4)>(&[2, 3, 4, 4], &[1, 3, 1, 1]),
        vec![2, 3, 4, 4]
    );
}

#[test]
fn stretching_works_from_either_side() {
    assert_eq!(
        resolved::<(U1, U4), (U3, U4), (U3, U4)>(&[1, 4], &[3, 4]),
        vec![3, 4]
    );
    assert_eq!(
        resolved::<(U3, U4), (U3, U1), (U3, U4)>(&[3, 4], &[3, 1]),
        vec![3, 4]
    );
}

#[test]
fn both_operands_may_stretch_at_different_axes() {
    // Neither shape is a subset of the other, and the result is larger than
    // both. Nothing about the rule requires one operand to dominate.
    assert_eq!(
        resolved::<(U3, U1), (U1, U4), (U3, U4)>(&[3, 1], &[1, 4]),
        vec![3, 4]
    );
}

#[test]
fn two_axes_of_extent_one_stay_one() {
    assert_eq!(resolved::<(U1, U1), (U1, U1), (U1, U1)>(&[1, 1], &[1, 1]), vec![1, 1]);
}

#[test]
fn identical_axes_still_pass_through_untouched() {
    // The case that worked before the per-axis rule, kept because it is the
    // one the other two impls must not have displaced.
    assert_eq!(
        resolved::<(U2, U3, U4), (U2, U3, U4), (U2, U3, U4)>(&[2, 3, 4], &[2, 3, 4]),
        vec![2, 3, 4]
    );
}

// -- named axes -----------------------------------------------------------

#[test]
fn a_named_axis_stretches_against_a_literal_one() {
    // `Batch` is not the type `U1`, so it sits on the non-stretched side and
    // the output keeps the name. What its runtime size turns out to be is a
    // separate question, answered below.
    assert_eq!(
        resolved::<(Batch, U4), (U1, U4), (Batch, U4)>(&[5, 4], &[1, 4]),
        vec![5, 4]
    );
}

#[test]
fn a_named_axis_whose_runtime_size_is_one_still_stretches_at_runtime() {
    // The type says `Batch` on both sides, so the axis pair is related by the
    // identity impl and the compiler settles nothing about the sizes. The
    // broadcast rule then applies to the values, exactly as decision D-013
    // requires.
    assert_eq!(
        resolved::<(Batch, U4), (Batch, U4), (Batch, U4)>(&[1, 4], &[6, 4]),
        vec![6, 4]
    );
}

#[test]
fn two_different_names_are_still_unrelated() {
    // `Batch` against `Seq` has no impl, in either direction. Asserting the
    // absence of an impl is `tests/compile_fail/named_dim_identity_mismatch.rs`
    // and cannot be written here; what this pins is that adding the stretch
    // impls did not accidentally relate them through `U1`.
    assert_eq!(
        resolved::<(Batch, U1), (Batch, Seq), (Batch, Seq)>(&[3, 1], &[3, 7]),
        vec![3, 7]
    );
}

// -- different ranks ------------------------------------------------------

#[test]
fn a_shorter_operand_right_aligns_and_may_stretch_where_it_reaches() {
    // The axes the shorter operand does not reach pass through; the ones it
    // does reach go through the same per-axis rule as the same-rank case.
    assert_eq!(
        resolved::<(U2, U3, U4, U4), (U3, U1, U1), (U2, U3, U4, U4)>(&[2, 3, 4, 4], &[3, 1, 1]),
        vec![2, 3, 4, 4]
    );
}

#[test]
fn the_shorter_operand_may_be_on_either_side() {
    assert_eq!(
        resolved::<(U3, U1), (U2, U3, U4), (U2, U3, U4)>(&[3, 1], &[2, 3, 4]),
        vec![2, 3, 4]
    );
}

#[test]
fn a_scalar_broadcasts_against_anything() {
    assert_eq!(
        resolved::<(), (U2, U3), (U2, U3)>(&[], &[2, 3]),
        vec![2, 3]
    );
}

// -- what still fails, and where ------------------------------------------

#[test]
fn incompatible_sizes_are_refused_at_runtime_when_the_type_cannot_see_them() {
    // Two `Dyn` operands settle nothing statically, so the rule runs entirely
    // on values. 2 against 5 is neither equal nor 1.
    let error = <Dyn as BroadcastShape<Dyn>>::output_shape(
        &field::<Dyn>(&[2, 4]),
        &field::<Dyn>(&[5, 4]),
    )
    .expect_err("2 and 5 do not broadcast");

    assert_eq!(
        error.operation(),
        incin_core::prelude::OperationKind::Broadcast
    );
}

#[test]
fn a_zero_axis_absorbs_the_stretch_rather_than_being_absorbed() {
    // `U1` against `U0` must give 0, not 1. The rule picks the side that is not
    // 1 rather than the larger of the two, which is the distinction a `max`
    // would get wrong.
    assert_eq!(
        resolved::<(U1, U4), (typenum::U0, U4), (typenum::U0, U4)>(&[1, 4], &[0, 4]),
        vec![0, 4]
    );
}

#[test]
fn a_five_element_axis_is_not_related_to_a_four_element_one() {
    // Guards the `NotOne` marker from over-reaching: it must admit every
    // typenum that is not `U1`, and must not make two of them interchangeable.
    // `(U5, U4)` against `(U4, U4)` has no impl, which is why this test names
    // the compatible spelling instead and leaves the refusal to the compiler.
    assert_eq!(
        resolved::<(U5, U4), (U1, U4), (U5, U4)>(&[5, 4], &[1, 4]),
        vec![5, 4]
    );
}
