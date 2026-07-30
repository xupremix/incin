//! `DST-003`: logical placement proofs and their sealed runtime projection.
//!
//! Positive compile-time cases are ordinary generic calls: compilation is the
//! assertion that the bound exists. Rejections live in
//! `tests/placement_compile_fail/`, because a trait implemented too broadly
//! would pass every positive case below.

#![cfg(feature = "distributed")]

mod support;

use std::collections::BTreeMap;
use std::path::Path;

use incin_core::dist::mesh::{Data, MeshSpec, Pipeline, TensorParallel};
use incin_core::dist::{
    CompletePlacement, DistributedError, DistributedInputs, ElementwisePlacement, LegalTransition,
    Local, Partial, PipelineStage, Placement, PlacementBuf, PlacementKind, PlacementTransition,
    PlacementTransitionRule, ReduceShardedAxis, Replicated, ShardDivisible, ShardRemainderPolicy,
    Sharded, Sum, validate_pipeline_stage, validate_shard,
};
use incin_core::exec::{ReduceOp, ReshapeSpec};
use incin_core::prelude::{Shape, ShapeBuf};
use incin_core::typenum::{U0, U1, U2, U3, U4, U10, U12};

type Mesh = MeshSpec<Data<U1>, TensorParallel<U3>>;
type HybridMesh = MeshSpec<Data<U2>, TensorParallel<U3>>;
type PipelineMesh = MeshSpec<Data<U1>, TensorParallel<U1>, Pipeline<U3>>;
type OtherMesh = MeshSpec<Data<U3>>;

fn reshape(input: &[usize], output: &[usize]) -> ReshapeSpec {
    ReshapeSpec::new(&ShapeBuf::from_slice(input), &ShapeBuf::from_slice(output)).unwrap()
}

fn static_local<Extent, Degree>() -> usize
where
    Extent: ShardDivisible<Degree>,
    Degree: incin_core::typenum::Unsigned,
{
    Extent::LOCAL
}

fn transition<From, To>() -> PlacementTransition
where
    From: LegalTransition<To>,
    To: Placement,
{
    From::TRANSITION
}

fn complete<P: CompletePlacement>() {}

fn elementwise_output<Lhs, Rhs, Output>()
where
    Lhs: ElementwisePlacement<Rhs, Output = Output>,
    Rhs: Placement,
    Output: CompletePlacement,
{
}

fn sharded_reduction_output<Input, Reduction, Output>()
where
    Input: ReduceShardedAxis<Reduction, Output = Output>,
    Reduction: incin_core::dist::PartialReduction,
    Output: Placement,
{
}

#[test]
fn every_typestate_has_one_runtime_projection() {
    assert_eq!(Local::kind(), PlacementKind::Local);
    assert_eq!(Replicated::<Mesh>::kind(), PlacementKind::Replicated);
    assert_eq!(
        Sharded::<Mesh, U1>::kind(),
        PlacementKind::Sharded { axis: 1 }
    );
    assert_eq!(
        Partial::<Mesh, Sum>::kind(),
        PlacementKind::Partial {
            reduction: ReduceOp::Sum
        }
    );
    assert_eq!(
        PipelineStage::<PipelineMesh, 2>::kind(),
        PlacementKind::PipelineStage { index: 2 }
    );

    assert!(PlacementKind::Local.is_complete());
    assert!(!Partial::<Mesh, Sum>::kind().is_complete());
    assert!(!PlacementKind::Local.is_distributed());
    assert!(Replicated::<Mesh>::kind().is_distributed());
}

#[test]
fn static_shard_divisibility_exposes_the_integral_local_extent() {
    assert_eq!(static_local::<U12, U3>(), 4);
    assert_eq!(static_local::<U12, U4>(), 3);
    assert_eq!(static_local::<U10, U1>(), 10);
}

#[test]
fn legal_transitions_name_exactly_the_data_movement_they_require() {
    assert_eq!(transition::<Local, Local>(), PlacementTransition::Identity);
    assert_eq!(
        transition::<Replicated<Mesh>, Sharded<Mesh, U0>>(),
        PlacementTransition::LocalShard
    );
    assert_eq!(
        transition::<Sharded<Mesh, U0>, Replicated<Mesh>>(),
        PlacementTransition::AllGather
    );
    assert_eq!(
        transition::<Partial<Mesh, Sum>, Replicated<Mesh>>(),
        PlacementTransition::AllReduce
    );
    assert_eq!(
        transition::<Partial<Mesh, Sum>, Sharded<Mesh, U0>>(),
        PlacementTransition::ReduceScatter
    );
    assert_eq!(
        transition::<PipelineStage<PipelineMesh, 2>, PipelineStage<PipelineMesh, 2>>(),
        PlacementTransition::Identity
    );
}

#[test]
fn complete_and_elementwise_bounds_accept_the_supported_families() {
    complete::<Local>();
    complete::<Replicated<Mesh>>();
    complete::<Sharded<Mesh, U0>>();
    complete::<PipelineStage<PipelineMesh, 0>>();

    elementwise_output::<Local, Local, Local>();
    elementwise_output::<Replicated<Mesh>, Replicated<Mesh>, Replicated<Mesh>>();
    elementwise_output::<Sharded<Mesh, U0>, Sharded<Mesh, U0>, Sharded<Mesh, U0>>();
    elementwise_output::<Replicated<Mesh>, Sharded<Mesh, U0>, Sharded<Mesh, U0>>();
    elementwise_output::<Sharded<Mesh, U0>, Replicated<Mesh>, Sharded<Mesh, U0>>();
    sharded_reduction_output::<Sharded<Mesh, U0>, Sum, Partial<Mesh, Sum>>();
}

#[test]
fn a_runtime_resolved_dimension_uses_the_same_exact_division_rule() {
    let global = ShapeBuf::from_slice(&[6, 12]);

    assert_eq!(
        validate_shard(&global, 1, 3, ShardRemainderPolicy::Reject)
            .unwrap()
            .dims(),
        &[6, 4]
    );
    assert_eq!(
        validate_shard(&global, 1, 5, ShardRemainderPolicy::Reject),
        Err(DistributedError::NonDivisible {
            axis: 1,
            extent: 12,
            shards: 5,
        })
    );
}

#[test]
fn runtime_shard_rejections_are_specific() {
    let global = ShapeBuf::from_slice(&[6, 12]);

    assert_eq!(
        validate_shard(&global, 1, 0, ShardRemainderPolicy::Reject),
        Err(DistributedError::ZeroShards)
    );
    assert_eq!(
        validate_shard(&global, 2, 3, ShardRemainderPolicy::Reject),
        Err(DistributedError::AxisOutOfBounds { axis: 2, rank: 2 })
    );
    assert_eq!(
        validate_shard(&global, 1, 5, ShardRemainderPolicy::PadAndMask),
        Err(DistributedError::UnsupportedRemainderPolicy {
            policy: ShardRemainderPolicy::PadAndMask,
        })
    );
    assert_eq!(
        validate_shard(&global, 1, 5, ShardRemainderPolicy::Ragged),
        Err(DistributedError::UnsupportedRemainderPolicy {
            policy: ShardRemainderPolicy::Ragged,
        })
    );
}

#[test]
fn pipeline_indices_are_checked_against_the_runtime_mesh_degree() {
    assert_eq!(validate_pipeline_stage(0, 3), Ok(()));
    assert_eq!(validate_pipeline_stage(2, 3), Ok(()));
    assert_eq!(
        validate_pipeline_stage(3, 3),
        Err(DistributedError::PipelineStageOutOfRange {
            index: 3,
            stages: 3
        })
    );
}

#[test]
fn a_valid_transition_mints_an_inspectable_distributed_proof() {
    type Global = (U3, U12);
    type Rule = PlacementTransitionRule<Replicated<Mesh>, Sharded<Mesh, U1>>;

    let operation = reshape(&[36], &[3, 12]);
    let inputs = DistributedInputs::<_, Global>::new(
        operation.clone(),
        Global::from_dyn(&[3, 12]).unwrap(),
        vec![
            ShapeBuf::from_slice(&[3, 4]),
            ShapeBuf::from_slice(&[3, 4]),
            ShapeBuf::from_slice(&[3, 4]),
        ],
        PlacementBuf::from([Replicated::<Mesh>::kind()]),
    );

    let validated = Rule::lower(&inputs).unwrap();

    assert_eq!(validated.operation(), &operation);
    assert_eq!(validated.global_shape().dims(), &[3, 12]);
    assert_eq!(validated.local_shapes().len(), 3);
    assert_eq!(
        validated.input_placements().as_slice(),
        &[PlacementKind::Replicated]
    );
    assert_eq!(
        validated.output_placement(),
        PlacementKind::Sharded { axis: 1 }
    );
    assert_eq!(validated.transition(), PlacementTransition::LocalShard);
}

#[test]
fn distributed_lowering_rejects_metadata_that_does_not_match_its_types() {
    type Global = (U3, U12);
    type Rule = PlacementTransitionRule<Replicated<Mesh>, Sharded<Mesh, U1>>;

    let wrong_descriptor = DistributedInputs::<_, Global>::new(
        reshape(&[30], &[3, 10]),
        Global::from_dyn(&[3, 12]).unwrap(),
        vec![ShapeBuf::from_slice(&[3, 4]); 3],
        PlacementBuf::from([PlacementKind::Replicated]),
    );
    assert_eq!(
        Rule::lower(&wrong_descriptor),
        Err(DistributedError::GlobalShapeMismatch)
    );

    let wrong_placement = DistributedInputs::<_, Global>::new(
        reshape(&[36], &[3, 12]),
        Global::from_dyn(&[3, 12]).unwrap(),
        vec![ShapeBuf::from_slice(&[3, 4]); 3],
        PlacementBuf::from([Sharded::<Mesh, U1>::kind()]),
    );
    assert!(matches!(
        Rule::lower(&wrong_placement),
        Err(DistributedError::UnexpectedInputPlacement { input: 0, .. })
    ));

    let incomplete_partition = DistributedInputs::<_, Global>::new(
        reshape(&[36], &[3, 12]),
        Global::from_dyn(&[3, 12]).unwrap(),
        vec![ShapeBuf::from_slice(&[3, 5]); 3],
        PlacementBuf::from([PlacementKind::Replicated]),
    );
    assert_eq!(
        Rule::lower(&incomplete_partition),
        Err(DistributedError::LocalExtentMismatch {
            local: 0,
            axis: 1,
            expected: 4,
            found: 5,
        })
    );
}

#[test]
fn local_shape_cardinality_is_derived_from_the_logical_mesh() {
    type Global = (U3, U12);
    type HybridRule = PlacementTransitionRule<Replicated<HybridMesh>, Sharded<HybridMesh, U1>>;

    let hybrid = DistributedInputs::<_, Global>::new(
        reshape(&[36], &[3, 12]),
        Global::from_dyn(&[3, 12]).unwrap(),
        vec![ShapeBuf::from_slice(&[3, 4]); 6],
        PlacementBuf::from([Replicated::<HybridMesh>::kind()]),
    );
    assert_eq!(HybridRule::lower(&hybrid).unwrap().local_shapes().len(), 6);

    let too_few = DistributedInputs::<_, Global>::new(
        reshape(&[36], &[3, 12]),
        Global::from_dyn(&[3, 12]).unwrap(),
        vec![ShapeBuf::from_slice(&[3, 4]); 3],
        PlacementBuf::from([Replicated::<HybridMesh>::kind()]),
    );
    assert_eq!(
        HybridRule::lower(&too_few),
        Err(DistributedError::LocalShapeCount {
            placement: PlacementKind::Sharded { axis: 1 },
            expected: 6,
            found: 3,
        })
    );

    type LocalRule = PlacementTransitionRule<Local, Local>;
    let too_many_local = DistributedInputs::<_, Global>::new(
        reshape(&[36], &[3, 12]),
        Global::from_dyn(&[3, 12]).unwrap(),
        vec![ShapeBuf::from_slice(&[3, 12]); 2],
        PlacementBuf::from([PlacementKind::Local]),
    );
    assert_eq!(
        LocalRule::lower(&too_many_local),
        Err(DistributedError::LocalShapeCount {
            placement: PlacementKind::Local,
            expected: 1,
            found: 2,
        })
    );
}

#[test]
fn pipeline_lowering_checks_the_index_encoded_by_the_typestate() {
    type Global = (U3, U4);
    type ValidRule =
        PlacementTransitionRule<PipelineStage<PipelineMesh, 2>, PipelineStage<PipelineMesh, 2>>;
    type InvalidRule =
        PlacementTransitionRule<PipelineStage<PipelineMesh, 3>, PipelineStage<PipelineMesh, 3>>;

    let operation = reshape(&[12], &[3, 4]);
    let valid = DistributedInputs::<_, Global>::new(
        operation.clone(),
        Global::from_dyn(&[3, 4]).unwrap(),
        vec![ShapeBuf::from_slice(&[3, 4])],
        PlacementBuf::from([PipelineStage::<PipelineMesh, 2>::kind()]),
    );
    assert!(ValidRule::lower(&valid).is_ok());

    let invalid = DistributedInputs::<_, Global>::new(
        operation,
        Global::from_dyn(&[3, 4]).unwrap(),
        vec![ShapeBuf::from_slice(&[3, 4])],
        PlacementBuf::from([PipelineStage::<PipelineMesh, 3>::kind()]),
    );
    assert_eq!(
        InvalidRule::lower(&invalid),
        Err(DistributedError::PipelineStageOutOfRange {
            index: 3,
            stages: 3,
        })
    );
}

#[test]
fn a_partial_value_becomes_complete_only_through_its_collective_transition() {
    type Global = (U3, U4);
    type Rule = PlacementTransitionRule<Partial<Mesh, Sum>, Replicated<Mesh>>;

    let operation = reshape(&[12], &[3, 4]);
    let inputs = DistributedInputs::<_, Global>::new(
        operation,
        Global::from_dyn(&[3, 4]).unwrap(),
        vec![ShapeBuf::from_slice(&[3, 4]); 3],
        PlacementBuf::from([Partial::<Mesh, Sum>::kind()]),
    );

    let validated = Rule::lower(&inputs).unwrap();
    assert_eq!(validated.output_placement(), PlacementKind::Replicated);
    assert!(validated.output_placement().is_complete());
    assert_eq!(validated.transition(), PlacementTransition::AllReduce);
}

#[test]
fn placement_rejections_are_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/placement_compile_fail/*.rs");
}

#[test]
fn every_placement_case_names_the_rule_it_pins() {
    support::compile_fail_cases_name_their_reason(
        Path::new("tests/placement_compile_fail"),
        &BTreeMap::from([
            ("cross_mesh_transition", "E0277"),
            ("illegal_partial_transition", "E0277"),
            ("partial_is_not_complete", "E0277"),
            ("pipeline_stage_transition_is_explicit", "E0277"),
            ("shard_not_divisible", "E0277"),
            ("validated_distributed_fields_are_private", "E0451"),
            ("validated_distributed_new_is_private", "E0624"),
        ]),
    );
}

#[test]
fn marker_implementations_do_not_require_traits_from_the_mesh_type() {
    fn accepts<P: Placement>() {}

    // `MeshSpec` itself implements neither `Clone` nor `Debug`. Placement
    // markers must not inherit either bound merely because they carry it.
    accepts::<Replicated<Mesh>>();
    accepts::<Sharded<Mesh, U0>>();
    accepts::<Partial<Mesh, Sum>>();
    accepts::<PipelineStage<PipelineMesh, 0>>();

    // Keep a second mesh type in this suite so the cross-mesh compile-fail case
    // cannot accidentally collapse two aliases to one type.
    accepts::<Replicated<OtherMesh>>();
}
