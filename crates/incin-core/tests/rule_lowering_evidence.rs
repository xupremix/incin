//! `EXE-003`: rules must seal full `ShapeEvidence`, not just a proof level.
//!
//! Every rule used to mint `Validated::new(descriptor, level)` - a level with
//! empty static geometry - while production dispatch re-mints full evidence
//! from `combined_evidence()`. Two things follow, and both are pinned here.
//!
//! First, a backend handed a rule-minted `Validated` directly (the
//! `ExecutionRequest` seam is public) saw `proof_level().is_static()` with no
//! extents or element count to specialize on: the statics the frontend proved
//! were thrown away one layer above the backend boundary. The positive cases
//! below fail before the seal, because `static_numel()` was `None`.
//!
//! Second, a rule whose *runtime* shape contradicted its output *type* still
//! sealed - the level came from the type, never checked against the field.
//! `ReshapeShape` proves element counts at the type level, so a lying runtime
//! target with the same numel sailed through; `reduce_shape` computes over
//! runtime dims without consulting the output type at all. The refusal cases
//! below returned `Ok` before the seal.
//!
//! The level itself may upgrade relative to the old `meet` of *input* levels:
//! what travels is the output type's own proof, verified against the runtime
//! field, so a mixed input reducing to a fully static output now reports what
//! the output actually proves.

extern crate incin_core as incin;

use incin::exec::catalog::AxisAttributes;
use incin::exec::{
    ExecutionDescriptor, MatMulRule, ProofLevel, ReduceKeepRule, ReduceRule, ReshapeRule, ShapeRule,
};
use incin::prelude::{Dyn, OperationKind, ShapeBuf, ShapeError};
use incin::shapes::idx::{Here, Next};
use incin_macros::s;

// -- matmul ---------------------------------------------------------------

#[test]
fn a_static_matmul_rule_seals_the_output_types_full_evidence() {
    let validated = <MatMulRule as ShapeRule<(s![3, 4], s![4, 3])>>::lower(
        &(ShapeBuf::from_slice(&[3, 4]), ShapeBuf::from_slice(&[4, 3])),
        (),
    )
    .expect("3x4 times 4x3 is a legal contraction");

    let evidence = validated.shape_evidence();
    assert_eq!(evidence.proof(), ProofLevel::Static);
    assert_eq!(evidence.static_rank(), Some(2));
    assert_eq!(evidence.static_numel(), Some(9));
    assert_eq!(evidence.static_extents(), &[Some(3), Some(3)][..]);
    assert_eq!(
        validated.descriptor().output_shape().unwrap().dims(),
        &[3, 3],
        "the sealed geometry is the geometry the descriptor carries"
    );
}

#[test]
fn a_matmul_rule_reports_a_static_output_even_when_the_contraction_axis_was_runtime() {
    // Inputs are Mixed - the shared K extent is only known at runtime - but
    // the output's M and N are fixed by the types, so the output type proves
    // more than `meet(lhs, rhs)` did. Pre-seal this sealed `Mixed` with no
    // statics; the seal carries the output type's own (stronger, checked)
    // answer.
    let validated = <MatMulRule as ShapeRule<(s![3, dyn], s![dyn, 3])>>::lower(
        &(ShapeBuf::from_slice(&[3, 7]), ShapeBuf::from_slice(&[7, 3])),
        (),
    )
    .expect("3x7 times 7x3 contracts at runtime K=7");

    let evidence = validated.shape_evidence();
    assert_eq!(evidence.proof(), ProofLevel::Static);
    assert_eq!(evidence.static_extents(), &[Some(3), Some(3)][..]);
    assert_eq!(evidence.static_numel(), Some(9));
}

#[test]
fn a_fully_dynamic_matmul_still_seals_dynamic_evidence() {
    // The seal must not overclaim: an all-runtime contraction earns no
    // statics, exactly as before.
    let validated = <MatMulRule as ShapeRule<(Dyn, Dyn)>>::lower(
        &(ShapeBuf::from_slice(&[3, 7]), ShapeBuf::from_slice(&[7, 5])),
        (),
    )
    .expect("runtime 3x7 times 7x5 is legal");

    let evidence = validated.shape_evidence();
    assert_eq!(evidence.proof(), ProofLevel::Dynamic);
    assert_eq!(evidence.static_rank(), None);
    assert_eq!(evidence.static_numel(), None);
    assert_eq!(evidence.static_extents(), &[][..]);
}

// -- reshape --------------------------------------------------------------

#[test]
fn a_static_reshape_rule_seals_the_output_types_full_evidence() {
    let validated = <ReshapeRule as ShapeRule<(s![2, 6], s![3, 4])>>::lower(
        &(ShapeBuf::from_slice(&[2, 6]), ShapeBuf::from_slice(&[3, 4])),
        (),
    )
    .expect("twelve elements may become 3x4");

    let evidence = validated.shape_evidence();
    assert_eq!(evidence.proof(), ProofLevel::Static);
    assert_eq!(evidence.static_rank(), Some(2));
    assert_eq!(evidence.static_numel(), Some(12));
    assert_eq!(evidence.static_extents(), &[Some(3), Some(4)][..]);
}

#[test]
fn a_reshape_rule_refuses_a_runtime_target_the_output_type_does_not_match() {
    // `ReshapeShape<S, T>` proves the *element counts* agree (12 = 12), so
    // the bound is satisfied, and `ShapeAttributes` only checks numel too -
    // the catalog happily infers `[4, 3]`. Nothing compared that field
    // against `T = s![3, 4]` before the seal: the old code sealed `Static`
    // with empty statics over a transposed field. `validate_dims` on the
    // output type is what refuses it now.
    let error = <ReshapeRule as ShapeRule<(s![2, 6], s![3, 4])>>::lower(
        &(ShapeBuf::from_slice(&[2, 6]), ShapeBuf::from_slice(&[4, 3])),
        (),
    )
    .expect_err("a [4, 3] runtime target is not the proved type s![3, 4]");

    assert!(matches!(
        error,
        ShapeError::TargetShapeRejected {
            operation: OperationKind::Reshape,
            rank: 2,
        }
    ));
}

// -- reduce ---------------------------------------------------------------

#[test]
fn a_reduce_rule_seals_the_output_types_full_evidence() {
    let validated = <ReduceRule as ShapeRule<(s![2, 3, 4], Next<Next<Here>>)>>::lower(
        &ShapeBuf::from_slice(&[2, 3, 4]),
        AxisAttributes { axis: 2 },
    )
    .expect("axis 2 of a 2x3x4 input is in range");

    let evidence = validated.shape_evidence();
    assert_eq!(evidence.proof(), ProofLevel::Static);
    assert_eq!(evidence.static_rank(), Some(2));
    assert_eq!(evidence.static_numel(), Some(6));
    assert_eq!(evidence.static_extents(), &[Some(2), Some(3)][..]);
}

#[test]
fn a_reduce_rule_refuses_runtime_dims_the_output_type_does_not_match() {
    // `reduce_shape` computes `[9, 9]` from the runtime field alone, and the
    // catalog's own inference does the same, so `agree` compared `[9, 9]` to
    // `[9, 9]` and passed. The output *type* for this cursor is `s![2, 3]`;
    // only the seal checks the field against it, which is what turns the old
    // `Ok(Static)` - a level claimed over dimensions the type rejects - into
    // a refusal.
    let error = <ReduceRule as ShapeRule<(s![2, 3, 4], Next<Next<Here>>)>>::lower(
        &ShapeBuf::from_slice(&[9, 9, 9]),
        AxisAttributes { axis: 2 },
    )
    .expect_err("runtime [9, 9, 9] cannot reduce to the proved type s![2, 3]");

    assert!(matches!(
        error,
        ShapeError::TargetShapeRejected {
            operation: OperationKind::SumDim,
            rank: 2,
        }
    ));
}

#[test]
fn a_reduce_rule_upgrades_a_mixed_input_to_the_static_output_it_proves() {
    // The input has a runtime axis, so the old `meet`-style level would have
    // been Mixed - but removing that axis leaves a fully static output, and
    // the output type proves it. The seal carries the output's answer, after
    // checking the runtime field agrees.
    let validated = <ReduceRule as ShapeRule<(s![2, 3, dyn], Next<Next<Here>>)>>::lower(
        &ShapeBuf::from_slice(&[2, 3, 7]),
        AxisAttributes { axis: 2 },
    )
    .expect("axis 2 of a 2x3x7 input is in range");

    let evidence = validated.shape_evidence();
    assert_eq!(
        evidence.proof(),
        ProofLevel::Static,
        "the proved output s![2, 3] is fully static even though the input was not"
    );
    assert_eq!(evidence.static_extents(), &[Some(2), Some(3)][..]);
    assert_eq!(evidence.static_numel(), Some(6));
}

// -- keepdim reduce -------------------------------------------------------

#[test]
fn a_keepdim_rule_seals_the_output_types_full_evidence() {
    let validated = <ReduceKeepRule as ShapeRule<(s![2, 3, 4], Next<Here>)>>::lower(
        &ShapeBuf::from_slice(&[2, 3, 4]),
        AxisAttributes { axis: 1 },
    )
    .expect("axis 1 of a 2x3x4 input is in range");

    let evidence = validated.shape_evidence();
    assert_eq!(evidence.proof(), ProofLevel::Static);
    assert_eq!(evidence.static_rank(), Some(3));
    assert_eq!(evidence.static_numel(), Some(8));
    assert_eq!(evidence.static_extents(), &[Some(2), Some(1), Some(4)][..]);
    assert_eq!(
        validated.descriptor().output_shape().unwrap().dims(),
        &[2, 1, 4]
    );
}

#[test]
fn a_keepdim_rule_upgrades_a_runtime_axis_to_the_static_one_it_proves() {
    // The reduced axis was runtime (`usize`), but keepdim rebinds it to a
    // static 1 (`usize::KeepDim = U1`), so the output type `s![2, 1]` is
    // fully static. Pre-seal this reported the *input's* Mixed level with no
    // statics; the seal reports what the output proves, once the field
    // `[2, 1]` has been checked against that type.
    let validated = <ReduceKeepRule as ShapeRule<(s![2, dyn], Next<Here>)>>::lower(
        &ShapeBuf::from_slice(&[2, 7]),
        AxisAttributes { axis: 1 },
    )
    .expect("axis 1 of a 2x7 input is in range");

    let evidence = validated.shape_evidence();
    assert_eq!(evidence.proof(), ProofLevel::Static);
    assert_eq!(evidence.static_extents(), &[Some(2), Some(1)][..]);
    assert_eq!(evidence.static_numel(), Some(2));
}
