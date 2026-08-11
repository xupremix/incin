//! `EXE-002` from outside the crate.
//!
//! The `compile_fail` cases next door prove what an external caller cannot do:
//! reach `Validated::new`, name its fields, or add a descriptor to the sealed
//! taxonomy. This file proves the other half — that everything a caller is
//! *supposed* to do still works across the crate boundary, so the seal did not
//! also lock out legitimate use.
//!
//! The provenance itself is the interesting part. A backend decides how hard
//! to specialize based on how much the compiler settled, so these tests pin
//! the classification of each shape family rather than just checking that the
//! type exists.

extern crate incin_core as incin;

use incin_core::exec::ProofLevel;
use incin_core::prelude::Dyn;
use incin_macros::s;

incin_core::dim!(Batch, Seq);

// -- what the compiler settled -------------------------------------------

#[test]
fn a_shape_written_entirely_in_typenum_is_fully_proved() {
    assert_eq!(ProofLevel::of::<s![3, 4]>(), ProofLevel::Static);
    assert!(ProofLevel::of::<s![3, 4]>().is_static());
    assert!(ProofLevel::of::<s![3, 4]>().has_static_rank());
}

#[test]
fn a_scalar_is_static_because_it_has_no_axis_to_be_dynamic() {
    assert_eq!(ProofLevel::of::<()>(), ProofLevel::Static);
}

#[test]
fn a_single_runtime_axis_is_enough_to_weaken_the_shape() {
    // This is the property that makes the level trustworthy: it reports the
    // weakest thing about the shape, not the strongest.
    assert_eq!(ProofLevel::of::<s![3, dyn]>(), ProofLevel::Mixed);
    assert_eq!(ProofLevel::of::<s![dyn, 3, 4]>(), ProofLevel::Mixed);
}

#[test]
fn naming_an_axis_types_it_without_sizing_it() {
    // `dim!(Batch)` buys compile-time rejection of a `Seq` used where a
    // `Batch` belongs. It does not tell anyone how large the batch is, so the
    // shape is Mixed rather than Static.
    assert_eq!(ProofLevel::of::<s![Batch, 4]>(), ProofLevel::Mixed);
    assert_eq!(ProofLevel::of::<s![Batch, Seq]>(), ProofLevel::Mixed);
}

#[test]
fn an_unranked_shape_settles_nothing_in_advance() {
    assert_eq!(ProofLevel::of::<Dyn>(), ProofLevel::Dynamic);
    assert!(!ProofLevel::of::<Dyn>().has_static_rank());
    assert!(!ProofLevel::of::<Dyn>().is_static());
}

#[test]
fn a_rank_one_static_shape_is_still_static() {
    // Guards the fold in the tuple macro: `true && U1::STATIC_SIZE` must not
    // collapse to Mixed at the smallest arity.
    assert_eq!(ProofLevel::of::<s![1]>(), ProofLevel::Static);
}

// -- combining operands ---------------------------------------------------

#[test]
fn an_operation_is_only_as_proved_as_its_weakest_operand() {
    let lhs = ProofLevel::of::<s![3, 4]>();
    let rhs = ProofLevel::of::<s![3, dyn]>();
    assert_eq!(lhs.meet(rhs), ProofLevel::Mixed);
    assert_eq!(rhs.meet(lhs), ProofLevel::Mixed, "order cannot matter");
}

#[test]
fn a_dynamic_operand_dominates_everything() {
    let dynamic = ProofLevel::of::<Dyn>();
    for other in [
        ProofLevel::of::<s![3, 4]>(),
        ProofLevel::of::<s![3, dyn]>(),
        ProofLevel::of::<()>(),
    ] {
        assert_eq!(dynamic.meet(other), ProofLevel::Dynamic);
    }
}

#[test]
fn folding_three_operands_matches_the_conv_case_the_rfc_could_not_express() {
    // Conv2d lowers input, weight, and bias. The RFC sketched
    // `ProofLevel::of::<L, R>()`, which has nowhere to put the third; folding
    // with `meet` does, and agrees with the pairwise answer.
    let input = ProofLevel::of::<s![dyn, 3, 4, 4]>();
    let weight = ProofLevel::of::<s![4, 3, 1, 1]>();
    let bias = ProofLevel::of::<s![4]>();

    let folded = [weight, bias]
        .into_iter()
        .fold(input, |acc, next| acc.meet(next));

    assert_eq!(folded, ProofLevel::Mixed);
    assert_eq!(folded, input.meet(weight).meet(bias));
}

// -- ordering and rendering ----------------------------------------------

#[test]
fn the_ordering_runs_strongest_to_weakest() {
    assert!(ProofLevel::Static < ProofLevel::Mixed);
    assert!(ProofLevel::Mixed < ProofLevel::Dynamic);
}

#[test]
fn a_level_renders_as_a_word_a_diagnostic_can_use() {
    assert_eq!(ProofLevel::Static.to_string(), "static");
    assert_eq!(ProofLevel::Mixed.to_string(), "mixed");
    assert_eq!(ProofLevel::Dynamic.to_string(), "dynamic");
}

#[test]
fn of_is_usable_in_a_const_so_a_backend_can_branch_at_compile_time() {
    // The point of `of` being `const`: a specializing backend can put the
    // decision in a `const` and have the branch disappear.
    const STATIC_PAIR: ProofLevel = ProofLevel::of::<s![3, 4]>();
    const MIXED_PAIR: ProofLevel = ProofLevel::of::<s![3, dyn]>();
    const COMBINED: ProofLevel = STATIC_PAIR.meet(MIXED_PAIR);

    assert_eq!(STATIC_PAIR, ProofLevel::Static);
    assert_eq!(COMBINED, ProofLevel::Mixed);
}
