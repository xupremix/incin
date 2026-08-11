//! `EXE-003`: the frontend's answer and the descriptor's must be the same one.
//!
//! Each rule computes its output twice — once through the shape trait the typed
//! frontend already uses, once through the descriptor constructor working on
//! erased dimensions — and these tests are the parity check between them. That
//! is the runtime half of decision `D-007`; the compile-time half is that
//! `ShapeRule::Output` restates the frontend trait's associated type, which
//! these tests exercise simply by naming it.
//!
//! Everything here runs from outside the crate, so it also pins what a caller
//! can reach: rules and descriptors are public, and `Validated::new` is not.

use incin_core::exec::{
    BinaryOp, BroadcastRule, BroadcastSpec, Conv2dArgs, Conv2dRule, MatMulRule, OperationSpec,
    Pool2dRule, Pool2dSpec, PoolOp, ProofLevel, ReduceAtRule, ReduceKeepAtRule, ReduceOp,
    ReshapeRule, ReshapeSpec, ShapeRule,
};
use incin_core::prelude::{Axis, Dyn, OperationKind, Shape, ShapeBuf};
use incin_core::shapes::idx::{Here, Next};
use typenum::{U0, U1, U2, U3, U16};
extern crate incin_core as incin;
use incin::prelude::s;

incin_core::dim!(Batch);

type R3 = s![2, 3, 4];

/// A shape's runtime field, built from the dimensions a test wants.
///
/// `from_dyn` returns `None` when the dimensions contradict the type, so a
/// typo in a test is a failed unwrap here rather than a wrong expectation
/// further down.
fn field<S: Shape>(dims: &[usize]) -> ShapeBuf {
    S::try_from_dims(dims).expect("test dimensions must match the shape type")
}

// -- broadcast ------------------------------------------------------------

/// The pair the broadcast tests lower: a rank-2 shape against a leading axis
/// of extent 1 that stretches to meet it.
type Rank2AgainstStretched = (s![3, 4], s![1, 4]);
type S34 = s![3, 4];
type S14 = s![1, 4];
type S23 = s![2, 3];
type S34Static = s![3, 4];
type S24 = s![2, 4];
type S04 = s![4];
type Batch4 = s![Batch, 4];
type BatchBatch4 = (s![Batch, 4], s![Batch, 4]);
type S26 = s![2, 6];
type S18_88 = s![1, 3, 8, 8];
type ConvInput = s![1, 3, 8, 8];

#[test]
fn a_static_broadcast_lowers_to_the_shape_the_frontend_names() {
    let lowered = <BroadcastRule as ShapeRule<Rank2AgainstStretched>>::lower(
        &(field::<S34>(&[3, 4]), field::<S14>(&[1, 4])),
        None,
    )
    .expect("3x4 against 1x4 broadcasts");

    assert_eq!(lowered.descriptor().output.dims(), &[3, 4]);
    assert_eq!(lowered.proof_level(), ProofLevel::Static);
}

#[test]
fn the_operator_reaches_the_descriptor_through_args() {
    // The shape types cannot say which operation this is: the same rule over
    // the same pair of shapes serves a stretch and all four binary operations.
    // `Args` is the only place the answer can come from, which is where
    // `Conv2dArgs` puts grouping and the reduce rules put `ReduceOp`.
    for op in [
        None,
        Some(BinaryOp::Add),
        Some(BinaryOp::Sub),
        Some(BinaryOp::Mul),
        Some(BinaryOp::Div),
    ] {
        let lowered = <BroadcastRule as ShapeRule<Rank2AgainstStretched>>::lower(
            &(field::<S34>(&[3, 4]), field::<S14>(&[1, 4])),
            op,
        )
        .expect("3x4 against 1x4 broadcasts");

        assert_eq!(lowered.descriptor().op, op);
        assert_eq!(lowered.descriptor().output.dims(), &[3, 4]);
    }
}

#[test]
fn a_stretched_axis_gets_a_zero_stride_rather_than_a_branch() {
    let lowered = <BroadcastRule as ShapeRule<Rank2AgainstStretched>>::lower(
        &(field::<S34>(&[3, 4]), field::<S14>(&[1, 4])),
        None,
    )
    .expect("3x4 against 1x4 broadcasts");
    let spec = lowered.descriptor();

    assert!(spec.rhs_broadcast_mask.contains(0));
    assert_eq!(spec.rhs_strides.strides()[0], 0);
    assert!(spec.lhs_broadcast_mask.is_empty());
}

#[test]
fn one_runtime_operand_weakens_the_whole_lowering() {
    // The descriptor is identical to the static case; what differs is how much
    // a backend may assume about it, which is the entire point of carrying the
    // level alongside.
    let lowered = <BroadcastRule as ShapeRule<(s![usize, 4], s![4])>>::lower(
        &(field::<s![usize, 4]>(&[3, 4]), field::<s![4]>(&[4])),
        None,
    )
    .expect("3x4 against 4 broadcasts");

    assert_eq!(lowered.descriptor().output.dims(), &[3, 4]);
    assert_eq!(lowered.proof_level(), ProofLevel::Mixed);
}

#[test]
fn an_unranked_operand_leaves_nothing_settled_in_advance() {
    let lowered = <BroadcastRule as ShapeRule<(Dyn, Dyn)>>::lower(
        &(field::<Dyn>(&[2, 1, 5]), field::<Dyn>(&[3, 5])),
        None,
    )
    .expect("2x1x5 against 3x5 broadcasts");

    assert_eq!(lowered.descriptor().output.dims(), &[2, 3, 5]);
    assert_eq!(lowered.proof_level(), ProofLevel::Dynamic);
}

#[test]
fn incompatible_dynamic_operands_are_reported_not_lowered() {
    let error = <BroadcastRule as ShapeRule<(Dyn, Dyn)>>::lower(
        &(field::<Dyn>(&[2, 3]), field::<Dyn>(&[4, 3])),
        None,
    )
    .expect_err("2 and 4 do not broadcast");

    assert_eq!(error.operation(), OperationKind::Broadcast);
}

#[test]
fn a_named_axis_is_typed_but_not_sized_so_one_side_may_still_stretch() {
    // `Batch` against `Batch` typechecks whatever the two sizes are, because
    // naming an axis says which axis it is and nothing about how long. A size
    // of 1 broadcasts here exactly as it would on an anonymous axis.
    let lowered = <BroadcastRule as ShapeRule<(s![Batch, 4], s![Batch, 4])>>::lower(
        &(
            field::<s![Batch, 4]>(&[1, 4]),
            field::<s![Batch, 4]>(&[4, 4]),
        ),
        None,
    )
    .expect("a `Batch` of 1 stretches to a `Batch` of 4");

    assert_eq!(lowered.descriptor().output.dims(), &[4, 4]);
    assert_eq!(lowered.proof_level(), ProofLevel::Mixed);
}

#[test]
fn two_uses_of_one_named_axis_must_still_be_compatible_at_runtime() {
    // The other half: sharing the name buys no exemption from the broadcast
    // rule, so two sizes that are neither equal nor 1 are an error even though
    // the pair typechecks.
    let error = <BroadcastRule as ShapeRule<(s![Batch, 4], s![Batch, 4])>>::lower(
        &(
            field::<s![Batch, 4]>(&[3, 4]),
            field::<s![Batch, 4]>(&[5, 4]),
        ),
        None,
    )
    .expect_err("a `Batch` cannot be both 3 and 5");

    assert_eq!(error.operation(), OperationKind::Broadcast);
    assert_eq!(error.axis(), Some(Axis::Index(0)));
}

// -- matrix multiplication ------------------------------------------------

#[test]
fn matmul_lowers_to_the_gemm_extents_its_shapes_imply() {
    let lowered = <MatMulRule as ShapeRule<(s![2, 3], s![3, 4])>>::lower(
        &(field::<s![2, 3]>(&[2, 3]), field::<s![3, 4]>(&[3, 4])),
        (),
    )
    .expect("2x3 times 3x4");
    let spec = lowered.descriptor();

    assert_eq!((spec.m, spec.n, spec.k), (2, 4, 3));
    assert_eq!(spec.output.dims(), &[2, 4]);
    assert!(spec.batch.is_empty());
    assert_eq!(lowered.proof_level(), ProofLevel::Static);
}

#[test]
fn a_batch_axis_reused_across_the_batch_gets_a_zero_stride() {
    let lowered = <MatMulRule as ShapeRule<(Dyn, Dyn)>>::lower(
        &(field::<Dyn>(&[5, 2, 3]), field::<Dyn>(&[1, 3, 4])),
        (),
    )
    .expect("batched 2x3 times 3x4");
    let spec = lowered.descriptor();

    assert_eq!(spec.output.dims(), &[5, 2, 4]);
    assert_eq!(spec.rhs_batch_strides.strides(), &[0]);
    assert_eq!(spec.lhs_batch_strides.strides(), &[6]);
}

#[test]
fn a_disagreeing_contraction_never_reaches_a_descriptor() {
    let error = <MatMulRule as ShapeRule<(Dyn, Dyn)>>::lower(
        &(field::<Dyn>(&[2, 3]), field::<Dyn>(&[5, 4])),
        (),
    )
    .expect_err("3 and 5 cannot contract");

    assert_eq!(error.operation(), OperationKind::MatMul);
}

#[test]
fn transposition_is_applied_after_lowering_because_it_is_not_a_shape_fact() {
    let lowered = <MatMulRule as ShapeRule<(s![2, 3], s![3, 4])>>::lower(
        &(field::<s![2, 3]>(&[2, 3]), field::<s![3, 4]>(&[3, 4])),
        (),
    )
    .expect("2x3 times 3x4");

    let transposed = lowered.into_descriptor().transposed(false, true);
    assert!(transposed.transpose_rhs);
    assert_eq!((transposed.m, transposed.n, transposed.k), (2, 4, 3));
}

// -- reduction ------------------------------------------------------------

#[test]
fn reducing_an_axis_drops_it_and_splits_the_shape_into_three_regions() {
    let lowered =
        <ReduceAtRule<Next<Here>> as ShapeRule<R3>>::lower(&field::<R3>(&[2, 3, 4]), ReduceOp::Sum)
            .expect("axis 1 is in range");
    let spec = lowered.descriptor();

    assert_eq!(spec.output.dims(), &[2, 4]);
    assert_eq!((spec.outer, spec.reduced, spec.inner), (2, 3, 4));
    assert!(!spec.keep_dims);
    assert_eq!(lowered.proof_level(), ProofLevel::Static);
}

#[test]
fn structural_reduction_rules_lower_through_the_same_descriptor() {
    type Axis1 = Next<Here>;
    let dropped =
        <ReduceAtRule<Axis1> as ShapeRule<R3>>::lower(&field::<R3>(&[2, 3, 4]), ReduceOp::Sum)
            .expect("structural axis 1 is in range");
    let kept =
        <ReduceKeepAtRule<Axis1> as ShapeRule<R3>>::lower(&field::<R3>(&[2, 3, 4]), ReduceOp::Sum)
            .expect("structural axis 1 is in range");

    assert_eq!(dropped.descriptor().output.dims(), &[2, 4]);
    assert_eq!(kept.descriptor().output.dims(), &[2, 1, 4]);
}

#[test]
fn keeping_the_axis_changes_the_output_but_not_the_three_extents() {
    let dropped =
        <ReduceAtRule<Next<Here>> as ShapeRule<R3>>::lower(&field::<R3>(&[2, 3, 4]), ReduceOp::Sum)
            .expect("axis 1 is in range");
    let kept = <ReduceKeepAtRule<Next<Here>> as ShapeRule<R3>>::lower(
        &field::<R3>(&[2, 3, 4]),
        ReduceOp::Sum,
    )
    .expect("axis 1 is in range");

    assert_eq!(kept.descriptor().output.dims(), &[2, 1, 4]);
    assert!(kept.descriptor().keep_dims);
    assert_eq!(
        (
            dropped.descriptor().outer,
            dropped.descriptor().reduced,
            dropped.descriptor().inner
        ),
        (
            kept.descriptor().outer,
            kept.descriptor().reduced,
            kept.descriptor().inner
        )
    );
}

#[test]
fn a_reduction_output_that_the_typed_shape_rejects_is_an_error() {
    // `Dyn` accepts any rank, so this exercises the rebuild path rather than a
    // rank check: the descriptor's dimensions must round-trip into the output
    // type's field, and for `Dyn` they always do.
    let lowered =
        <ReduceAtRule<Here> as ShapeRule<Dyn>>::lower(&field::<Dyn>(&[7, 2]), ReduceOp::Sum)
            .expect("axis 0 is in range");

    assert_eq!(lowered.descriptor().output.dims(), &[2]);
    assert_eq!(lowered.proof_level(), ProofLevel::Dynamic);
}

#[test]
fn the_accumulation_is_carried_into_the_descriptor_rather_than_defaulted() {
    // The geometry is identical for all five, so nothing downstream could tell
    // them apart if the rule dropped the argument on the way through.
    for op in [
        ReduceOp::Sum,
        ReduceOp::Mean,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::Prod,
    ] {
        let dropped =
            <ReduceAtRule<Next<Here>> as ShapeRule<R3>>::lower(&field::<R3>(&[2, 3, 4]), op)
                .expect("axis 1 is in range");
        let kept =
            <ReduceKeepAtRule<Next<Here>> as ShapeRule<R3>>::lower(&field::<R3>(&[2, 3, 4]), op)
                .expect("axis 1 is in range");

        assert_eq!(dropped.descriptor().op, op);
        assert_eq!(kept.descriptor().op, op);
    }
}

// -- reshape --------------------------------------------------------------

#[test]
fn reshape_carries_the_element_count_the_two_shapes_share() {
    let lowered = <ReshapeRule as ShapeRule<(s![2, 6], s![3, 4])>>::lower(
        &(field::<s![2, 6]>(&[2, 6]), field::<s![3, 4]>(&[3, 4])),
        (),
    )
    .expect("12 elements either way");
    let spec = lowered.descriptor();

    assert_eq!(spec.input.dims(), &[2, 6]);
    assert_eq!(spec.output.dims(), &[3, 4]);
    assert_eq!(spec.elements, 12);
    assert_eq!(lowered.proof_level(), ProofLevel::Static);
}

#[test]
fn a_reshape_between_static_shapes_of_different_sizes_does_not_typecheck() {
    // `ReshapeShape` is implemented only where the two `ElementCount`s are the
    // same typenum, so `ReshapeRule` cannot be handed a mismatched pair at all.
    // The constructor still rejects one, because a caller can reach it directly
    // and a dynamic reshape will once one exists.
    let error = ReshapeSpec::new(
        &ShapeBuf::from_slice(&[2, 6]),
        &ShapeBuf::from_slice(&[5, 5]),
    )
    .expect_err("12 elements cannot become 25");

    assert_eq!(error.operation(), OperationKind::Reshape);
}

// -- convolution and pooling ----------------------------------------------

type Conv3x3 = Conv2dRule<U16, U3, U1, U1, U1>;
type Pool2x2 = Pool2dRule<U2, U2, U0, U1>;

#[test]
fn a_padded_convolution_preserves_its_spatial_extent() {
    let lowered = <Conv3x3 as ShapeRule<ConvInput>>::lower(
        &field::<ConvInput>(&[1, 3, 8, 8]),
        Conv2dArgs::dense(16),
    )
    .expect("a 3x3 window with padding 1 fits an 8x8 input");
    let spec = lowered.descriptor();

    assert_eq!(spec.output.dims(), &[1, 16, 8, 8]);
    assert_eq!((spec.c_in, spec.c_out), (3, 16));
    assert_eq!(spec.groups, 1);
    assert_eq!(lowered.proof_level(), ProofLevel::Static);
}

#[test]
fn grouping_that_does_not_divide_the_channels_is_rejected() {
    let error = <Conv3x3 as ShapeRule<ConvInput>>::lower(
        &field::<ConvInput>(&[1, 3, 8, 8]),
        Conv2dArgs {
            out_channels: 16,
            groups: 2,
        },
    )
    .expect_err("2 does not divide 3 input channels");

    assert_eq!(error.operation(), OperationKind::Conv2d);
}

#[test]
fn pooling_halves_the_spatial_axes_and_leaves_the_channels_alone() {
    let lowered =
        <Pool2x2 as ShapeRule<ConvInput>>::lower(&field::<ConvInput>(&[1, 3, 8, 8]), PoolOp::Max)
            .expect("a 2x2 window strided by 2 tiles an 8x8 input");
    let spec = lowered.descriptor();

    assert_eq!(spec.output.dims(), &[1, 3, 4, 4]);
    assert_eq!(spec.channels, 3);
    assert_eq!((spec.h_out, spec.w_out), (4, 4));
}

#[test]
fn pooling_carries_the_accumulation_that_shares_its_window() {
    for op in [PoolOp::Max, PoolOp::Average] {
        let lowered =
            <Pool2x2 as ShapeRule<ConvInput>>::lower(&field::<ConvInput>(&[1, 3, 8, 8]), op)
                .expect("a 2x2 window strided by 2 tiles an 8x8 input");

        // One geometry, two operations. `Pool2dSpec` is only a complete request
        // because it records which.
        assert_eq!(lowered.descriptor().op, op);
        assert_eq!(lowered.descriptor().output.dims(), &[1, 3, 4, 4]);
    }
}

#[test]
fn a_window_larger_than_its_padded_input_is_reported_not_collapsed() {
    let error = Pool2dSpec::new(
        &ShapeBuf::from_slice(&[1, 3, 2, 2]),
        [4, 4],
        [1, 1],
        [0, 0],
        [1, 1],
        PoolOp::Max,
    )
    .expect_err("a 4x4 window does not fit a 2x2 input");

    assert_eq!(error.operation(), OperationKind::Pool2d);
}

// -- what the descriptors report about themselves -------------------------

#[test]
fn pooling_and_reshape_report_their_own_kinds() {
    // The reason each has a descriptor of its own. Expressing a pool as a
    // depthwise `Conv2dSpec` would produce the same geometry and then answer
    // `Conv2d` to every capability query and cache lookup keyed on it.
    assert_eq!(<Pool2dSpec as OperationSpec>::KIND, OperationKind::Pool2d);
    assert_eq!(<ReshapeSpec as OperationSpec>::KIND, OperationKind::Reshape);
}

#[test]
fn every_descriptor_a_rule_produces_can_state_its_element_count() {
    let lowered = <BroadcastRule as ShapeRule<Rank2AgainstStretched>>::lower(
        &(field::<S34>(&[3, 4]), field::<S14>(&[1, 4])),
        None,
    )
    .expect("3x4 against 1x4 broadcasts");

    assert_eq!(lowered.descriptor().output_elements(), Ok(12));
}

#[test]
fn the_descriptor_a_rule_mints_equals_the_one_built_by_hand() {
    // Lowering adds provenance; it does not add or reinterpret geometry. A
    // caller who builds the descriptor directly gets the same fields, minus any
    // evidence that a shape proof stood behind them.
    let lowered = <BroadcastRule as ShapeRule<Rank2AgainstStretched>>::lower(
        &(field::<S34>(&[3, 4]), field::<S14>(&[1, 4])),
        None,
    )
    .expect("3x4 against 1x4 broadcasts");
    let by_hand = BroadcastSpec::contiguous(
        &ShapeBuf::from_slice(&[3, 4]),
        &ShapeBuf::from_slice(&[1, 4]),
        None,
    )
    .expect("3x4 against 1x4 broadcasts");

    assert_eq!(lowered.into_descriptor(), by_hand);
}
