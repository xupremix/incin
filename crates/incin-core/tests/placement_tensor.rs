//! `DST-004`: one `Tensor` carries global shape, rank-local storage, dtype,
//! device, gradient, and placement metadata.
//!
//! Static placement/shape/dtype cases compile only when their trait proofs
//! exist. The `Dyn` case exercises the same boundary with runtime values.

#![cfg(feature = "distributed")]

mod support;

use std::collections::BTreeMap;
use std::path::Path;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::dist::mesh::{Data, MeshSpec, TensorParallel};
use incin_core::dist::{
    ConstPlacement, DistributedInputs, Local, Placement, PlacementBuf, PlacementKind,
    PlacementTransitionRule, Replicated, Sharded,
};
use incin_core::exec::ReshapeSpec;
use incin_core::prelude::{DTypeId, Dyn, Grad, PlacedTensorError, Shape, ShapeBuf, Tensor};
use incin_core::typenum::{U1, U2, U8};

type B = CpuBackendImpl;
type Mesh = MeshSpec<Data<U1>, TensorParallel<U2>>;
type Global = (U2, U8);
type ReplicatedTensor = Tensor<Global, B, f32, Grad, Replicated<Mesh>>;
type ShardedTensor = Tensor<Global, B, f32, Grad, Sharded<Mesh, U1>>;

#[test]
fn local_placement_metadata_remains_zero_sized() {
    assert_eq!(
        core::mem::size_of::<<Local as Placement>::Field>(),
        0,
        "ordinary local tensors must not store a distributed rank"
    );
    assert_eq!(
        core::mem::size_of::<<Sharded<Mesh, U1> as Placement>::Field>(),
        core::mem::size_of::<usize>()
    );
}

fn reshape(input: &[usize], output: &[usize]) -> ReshapeSpec {
    ReshapeSpec::new(&ShapeBuf::from_slice(input), &ShapeBuf::from_slice(output)).unwrap()
}

fn replicated_proof() -> incin_core::dist::ValidatedDistributed<ReshapeSpec> {
    let inputs = DistributedInputs::<_, Global>::new(
        reshape(&[2, 8], &[2, 8]),
        Global::from_dyn(&[2, 8]).unwrap(),
        vec![ShapeBuf::from_slice(&[2, 8]), ShapeBuf::from_slice(&[2, 8])],
        PlacementBuf::from([Replicated::<Mesh>::PLACEMENT]),
    );
    PlacementTransitionRule::<Replicated<Mesh>, Replicated<Mesh>>::lower(&inputs).unwrap()
}

fn sharded_proof() -> incin_core::dist::ValidatedDistributed<ReshapeSpec> {
    let inputs = DistributedInputs::<_, Global>::new(
        reshape(&[2, 8], &[2, 8]),
        Global::from_dyn(&[2, 8]).unwrap(),
        vec![ShapeBuf::from_slice(&[2, 4]), ShapeBuf::from_slice(&[2, 4])],
        PlacementBuf::from([Replicated::<Mesh>::PLACEMENT]),
    );
    PlacementTransitionRule::<Replicated<Mesh>, Sharded<Mesh, U1>>::lower(&inputs).unwrap()
}

#[test]
fn a_static_placed_tensor_keeps_global_and_rank_local_shapes_distinct() {
    let storage = Tensor::<Dyn, B>::zeros(vec![2, 4]).unwrap().into_inner();
    let tensor = ShardedTensor::try_from_distributed_storage(
        storage,
        Global::from_dyn(&[2, 8]).unwrap(),
        Default::default(),
        Default::default(),
        Default::default(),
        1,
        &sharded_proof(),
    )
    .unwrap();

    assert_eq!(tensor.dims().as_ref(), &[2, 8]);
    assert_eq!(tensor.local_dims(), vec![2, 4]);
    assert_eq!(tensor.rank_index(), 1);
    assert_eq!(tensor.placement(), PlacementKind::Sharded { axis: 1 });
    assert_eq!(tensor.dtype(), DTypeId::F32);
}

#[test]
fn static_reshard_requires_both_the_trait_proof_and_matching_runtime_storage() {
    let replicated_storage = Tensor::<Dyn, B>::zeros(vec![2, 8]).unwrap().into_inner();
    let replicated = ReplicatedTensor::try_from_distributed_storage(
        replicated_storage,
        Global::from_dyn(&[2, 8]).unwrap(),
        Default::default(),
        Default::default(),
        Default::default(),
        0,
        &replicated_proof(),
    )
    .unwrap();

    let shard_storage = Tensor::<Dyn, B>::zeros(vec![2, 4]).unwrap().into_inner();
    let sharded: ShardedTensor = replicated
        .try_reshard::<Sharded<Mesh, U1>, _>(shard_storage, 0, &sharded_proof())
        .unwrap();

    assert_eq!(sharded.dims().as_ref(), &[2, 8]);
    assert_eq!(sharded.local_dims(), vec![2, 4]);
}

#[test]
fn dyn_shape_dtype_and_placement_are_checked_at_runtime() {
    type DynamicTensor = Tensor<Dyn, B, Dyn, Grad, Dyn>;

    let storage = Tensor::<Dyn, B, Dyn>::zeros((vec![2, 4], DTypeId::F64))
        .unwrap()
        .into_inner();
    let tensor = DynamicTensor::try_from_distributed_storage(
        storage,
        vec![2, 8],
        DTypeId::F64,
        Default::default(),
        Default::default(),
        0,
        &sharded_proof(),
    )
    .unwrap();

    assert_eq!(tensor.dims(), vec![2, 8]);
    assert_eq!(tensor.local_dims(), vec![2, 4]);
    assert_eq!(tensor.dtype(), DTypeId::F64);
    assert_eq!(tensor.placement(), PlacementKind::Sharded { axis: 1 });
}

#[test]
fn dyn_metadata_mismatches_are_rejected_before_a_tensor_is_minted() {
    type DynamicTensor = Tensor<Dyn, B, Dyn, Grad, Dyn>;

    let wrong_dtype = Tensor::<Dyn, B, Dyn>::zeros((vec![2, 4], DTypeId::F32))
        .unwrap()
        .into_inner();
    let error = DynamicTensor::try_from_distributed_storage(
        wrong_dtype,
        vec![2, 8],
        DTypeId::F64,
        Default::default(),
        Default::default(),
        0,
        &sharded_proof(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        PlacedTensorError::DType {
            expected: DTypeId::F64,
            got: DTypeId::F32,
        }
    );

    let wrong_shape = Tensor::<Dyn, B>::zeros(vec![2, 5]).unwrap().into_inner();
    let error = ShardedTensor::try_from_distributed_storage(
        wrong_shape,
        Global::from_dyn(&[2, 8]).unwrap(),
        Default::default(),
        Default::default(),
        Default::default(),
        0,
        &sharded_proof(),
    )
    .unwrap_err();
    assert!(matches!(error, PlacedTensorError::LocalShape { .. }));

    let wrong_placement = Tensor::<Dyn, B>::zeros(vec![2, 4]).unwrap().into_inner();
    let error = ReplicatedTensor::try_from_distributed_storage(
        wrong_placement,
        Global::from_dyn(&[2, 8]).unwrap(),
        Default::default(),
        Default::default(),
        Default::default(),
        0,
        &sharded_proof(),
    )
    .unwrap_err();
    assert!(matches!(error, PlacedTensorError::OutputPlacement { .. }));
}

#[test]
fn dyn_reshard_uses_the_runtime_legal_transition_table() {
    type DynamicTensor = Tensor<Dyn, B, Dyn, Grad, Dyn>;

    let storage = Tensor::<Dyn, B, Dyn>::zeros((vec![2, 8], DTypeId::F32))
        .unwrap()
        .into_inner();
    let replicated = DynamicTensor::try_from_distributed_storage(
        storage,
        vec![2, 8],
        DTypeId::F32,
        Default::default(),
        Default::default(),
        0,
        &replicated_proof(),
    )
    .unwrap();

    let shard = Tensor::<Dyn, B, Dyn>::zeros((vec![2, 4], DTypeId::F32))
        .unwrap()
        .into_inner();
    let sharded = replicated
        .try_reshard_dyn(
            shard,
            PlacementKind::Sharded { axis: 1 },
            0,
            &sharded_proof(),
        )
        .unwrap();
    assert_eq!(sharded.placement(), PlacementKind::Sharded { axis: 1 });
}

#[test]
fn illegal_static_reshards_are_compile_errors() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/placement_tensor_compile_fail/*.rs");
}

#[test]
fn every_placed_tensor_compile_failure_names_its_rule() {
    support::compile_fail_cases_name_their_reason(
        Path::new("tests/placement_tensor_compile_fail"),
        &BTreeMap::from([("illegal_static_reshard", "E0277")]),
    );
}
