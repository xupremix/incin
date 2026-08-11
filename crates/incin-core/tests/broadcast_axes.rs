//! `SHP-007`: the axis pairs the type system accepts, and the ones it refuses.
//!
//! This is the compile-pass half of the broadcast suite. Every function here
//! names a concrete `BroadcastShape` `Output`, so the file failing to compile
//! *is* the assertion: a pair that stops typechecking cannot reach the
//! runtime checks below it. The refusals live next door in
//! `tests/compile_fail/`, which is where an absent impl can be asserted.
//!
//! A `trybuild` pass case would prove less than this: it would show the pair
//! compiles without showing that `Output` resolves to the shape it should, and
//! a rule that typechecks while naming the wrong output shape is the failure
//! mode worth catching.
//!
//! Before `SHP-007` the same-rank family required every axis to be the
//! identical type, so the pair at the centre of this file, a tensor against a
//! bias shaped `(U1, C, U1, U1)`, did not typecheck at all.

use incin::prelude::s;
use incin_core::prelude::{BroadcastShape, Dyn, Shape, ShapeBuf};
use typenum::{U1, U2, U3, U4};
extern crate incin_core as incin;

incin_core::dim!(Batch, Seq);
type S4<A, B, C, D> = incin_core::shapes::DimCons<
    A,
    incin_core::shapes::DimCons<
        B,
        incin_core::shapes::DimCons<C, incin_core::shapes::DimCons<D, incin_core::shapes::Nil>>,
    >,
>;

trait Same<T> {}
impl<T> Same<T> for T {}

fn assert_same<A, B>()
where
    A: Same<B>,
    B: Same<A>,
{
}

/// A shape's runtime field, built from the dimensions a test wants.
fn field<S: Shape>(dims: &[usize]) -> ShapeBuf {
    S::try_from_dims(dims).expect("test dimensions must match the shape type")
}

/// The dimensions `L` broadcast against `R` resolve to the symbolic output
/// extent selected by the Stable-Rust broadcast fallback.
fn resolved<L, R>(lhs: &[usize], rhs: &[usize]) -> Vec<usize>
where
    L: BroadcastShape<R> + Shape,
    R: Shape,
{
    let out =
        L::output_shape(&field::<L>(lhs), &field::<R>(rhs)).expect("these operands broadcast");
    out.as_ref().to_vec()
}

// -- same rank, one side stretched ---------------------------------------

#[test]
fn a_bias_shaped_like_the_channel_axis_broadcasts_over_a_batch() {
    // The case SHP-007 exists for. `(U1, C, U1, U1)` is how a per-channel bias
    // or a normalization statistic is spelled, and before the per-axis rule it
    // could not meet its own tensor without erasing to `Dyn`.
    type Tensor4 = S4<U2, U3, U4, U4>;
    type Bias = S4<U1, U3, U1, U1>;
    type Output = <Tensor4 as BroadcastShape<Bias>>::Output;
    type Expected = S4<U2, U3, U4, U4>;
    assert_same::<Output, Expected>();
    assert_eq!(
        resolved::<Tensor4, Bias>(&[2, 3, 4, 4], &[1, 3, 1, 1]),
        vec![2, 3, 4, 4]
    );
}

#[test]
fn stretching_works_from_either_side() {
    assert_eq!(resolved::<s![1, 4], s![3, 4]>(&[1, 4], &[3, 4]), vec![3, 4]);
    assert_eq!(resolved::<s![3, 4], s![3, 1]>(&[3, 4], &[3, 1]), vec![3, 4]);
}

#[test]
fn both_operands_may_stretch_at_different_axes() {
    // Neither shape is a subset of the other, and the result is larger than
    // both. Nothing about the rule requires one operand to dominate.
    assert_eq!(resolved::<s![3, 1], s![1, 4]>(&[3, 1], &[1, 4]), vec![3, 4]);
}

#[test]
fn broadcast_output_normalizes_static_extents() {
    type Output = <s![3, 1] as BroadcastShape<s![1, 4]>>::Output;
    type Expected = s![3, 4];

    assert_same::<Output, Expected>();
}

#[test]
fn two_axes_of_extent_one_stay_one() {
    assert_eq!(resolved::<s![1, 1], s![1, 1]>(&[1, 1], &[1, 1]), vec![1, 1]);
}

#[test]
fn identical_axes_still_pass_through_untouched() {
    // The case that worked before the per-axis rule, kept because it is the
    // one the other two impls must not have displaced.
    assert_eq!(
        resolved::<s![2, 3, 4], s![2, 3, 4]>(&[2, 3, 4], &[2, 3, 4]),
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
        resolved::<s![Batch, usize,], s![1, 4]>(&[5, 4], &[1, 4]),
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
        resolved::<s![Batch, usize,], s![Batch, usize,]>(&[1, 4], &[6, 4]),
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
        resolved::<s![Batch, 1], s![Batch, Seq]>(&[3, 1], &[3, 7]),
        vec![3, 7]
    );
}

// -- different ranks ------------------------------------------------------

#[test]
fn a_shorter_operand_right_aligns_and_may_stretch_where_it_reaches() {
    // The axes the shorter operand does not reach pass through; the ones it
    // does reach go through the same per-axis rule as the same-rank case.
    assert_eq!(
        resolved::<s![2, 3, 4, 4], s![3, 1, 1]>(&[2, 3, 4, 4], &[3, 1, 1]),
        vec![2, 3, 4, 4]
    );
}

#[test]
fn the_shorter_operand_may_be_on_either_side() {
    assert_eq!(
        resolved::<s![3, 1], s![2, 3, 4]>(&[3, 1], &[2, 3, 4]),
        vec![2, 3, 4]
    );
}

#[test]
fn a_scalar_broadcasts_against_anything() {
    assert_eq!(resolved::<s![], s![2, 3]>(&[], &[2, 3]), vec![2, 3]);
}

// -- what still fails, and where ------------------------------------------

#[test]
fn incompatible_sizes_are_refused_at_runtime_when_the_type_cannot_see_them() {
    // Two `Dyn` operands settle nothing statically, so the rule runs entirely
    // on values. 2 against 5 is neither equal nor 1.
    let error =
        <Dyn as BroadcastShape<Dyn>>::output_shape(&field::<Dyn>(&[2, 4]), &field::<Dyn>(&[5, 4]))
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
    assert_eq!(resolved::<s![1, 4], s![0, 4]>(&[1, 4], &[0, 4]), vec![0, 4]);
}

#[test]
fn a_five_element_axis_is_not_related_to_a_four_element_one() {
    // Guards the static broadcast rule from over-reaching: it must admit every
    // singleton stretch without making unequal non-singletons interchangeable.
    // `(U5, U4)` against `(U4, U4)` has no impl, which is why this test names
    // the compatible spelling instead and leaves the refusal to the compiler.
    assert_eq!(resolved::<s![5, 4], s![1, 4]>(&[5, 4], &[1, 4]), vec![5, 4]);
}

// -- a runtime axis meeting a static one ----------------------------------

#[test]
fn a_runtime_axis_may_sit_anywhere_rather_than_only_in_front() {
    // Before SHP-007 a `usize` axis was only related by the mixed families,
    // every one of which required it to be axis 0 and the remaining axes to be
    // identical on both sides. `(U3, usize)` had no partner at all.
    assert_eq!(
        resolved::<s![3, usize], s![3, 4]>(&[3, 4], &[3, 4]),
        vec![3, 4]
    );
}

#[test]
fn meeting_a_static_axis_recovers_the_size_the_runtime_one_lost() {
    // The output type is `(U3, U4)`, not `(usize, U4)`. A `usize` axis that
    // broadcasts against `U3` is either 3 or 1, and the result is 3 in both
    // cases, so the static side is the more precise answer and keeping it is
    // free. What was a runtime axis on the way in is a proved one on the way
    // out.
    assert_eq!(
        resolved::<s![usize, 4], s![3, 4]>(&[3, 4], &[3, 4]),
        vec![3, 4]
    );
    assert_eq!(
        resolved::<s![usize, 4], s![3, 4]>(&[1, 4], &[3, 4]),
        vec![3, 4]
    );
}

#[test]
fn runtime_static_broadcast_keeps_the_provable_output_extent() {
    type AnonymousOut = <s![usize] as BroadcastShape<s![64]>>::Output;
    type NamedOut = <s![Batch] as BroadcastShape<s![64]>>::Output;
    type ExpectedAnonymous = s![64];
    type ExpectedNamed = s![Batch: 64];

    assert_same::<AnonymousOut, ExpectedAnonymous>();
    assert_same::<NamedOut, ExpectedNamed>();
}

#[test]
fn runtime_broadcast_with_static_one_remains_runtime() {
    type Out = <s![usize] as BroadcastShape<s![1]>>::Output;
    assert_same::<Out, s![usize]>();
}

#[test]
fn meeting_a_literal_one_keeps_the_axis_dynamic() {
    // The mirror of the case above, and the reason it cannot be stated as "the
    // static side always wins": `U1` proves nothing about the result, so the
    // runtime axis is what survives.
    assert_eq!(
        resolved::<s![usize, 4], s![1, 4]>(&[7, 4], &[1, 4]),
        vec![7, 4]
    );
}

#[test]
fn two_runtime_axes_stay_runtime() {
    assert_eq!(
        resolved::<s![usize, 4], s![usize, 4]>(&[5, 4], &[1, 4]),
        vec![5, 4]
    );
}

#[test]
fn a_runtime_axis_that_contradicts_its_static_partner_is_rejected() {
    // The type admitted the pair; the value does not satisfy it. This is the
    // check that has to happen at runtime, and it happens once.
    let error = <s![usize, 4] as BroadcastShape<s![3, 4]>>::output_shape(
        &field::<s![usize, 4]>(&[5, 4]),
        &field::<s![3, 4]>(&[3, 4]),
    )
    .expect_err("5 and 3 do not broadcast");

    assert_eq!(
        error.operation(),
        incin_core::prelude::OperationKind::Broadcast
    );
}
