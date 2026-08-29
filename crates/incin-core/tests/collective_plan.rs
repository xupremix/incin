//! `DST-007`: collective descriptors contain every launch invariant and ranks
//! compare one compact summary before entering a transport.

#![cfg(feature = "distributed")]

mod support;

use std::collections::BTreeMap;
use std::path::Path;

use incin_core::dist::mesh::{
    Data, DeviceIdentity, DeviceMesh, LinkClass, MeshAxis, MeshSpec, ProcessLayout, TensorParallel,
    TopologyProbe, TransportVersion,
};
use incin_core::dist::{
    CollectiveError, CollectiveKind, CollectivePlan, CollectivePlanBuilder, Partial, PlacementKind,
    PlanError, Replicated, Sharded, StreamId, Sum, preflight,
};
use incin_core::exec::ReduceOp;
use incin_core::prelude::{DTypeId, DeviceId};
use incin_core::typenum::{U1, U2};

type Mesh = MeshSpec<Data<U1>, TensorParallel<U2>>;
type Shard = Sharded<Mesh, U1>;
type Replica = Replicated<Mesh>;
type PartialSum = Partial<Mesh, Sum>;

#[derive(Clone)]
struct TwoCuda {
    suffix: &'static str,
    local_rank: Option<usize>,
}

impl TopologyProbe for TwoCuda {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        (device.kind() == incin_core::prelude::DeviceKind::Cuda && device.ordinal() < 2).then(
            || {
                DeviceIdentity::new(
                    device,
                    format!("GPU-{}-{}", device.ordinal(), self.suffix),
                    "sm_90".to_string(),
                )
            },
        )
    }

    fn link(&self, from: DeviceId, to: DeviceId) -> LinkClass {
        if from == to {
            LinkClass::SameDevice
        } else {
            LinkClass::Network
        }
    }

    fn transport(&self) -> TransportVersion {
        TransportVersion::new("reference".to_string(), 1, 0, 0)
    }

    fn layout(&self) -> ProcessLayout {
        self.local_rank
            .map_or(ProcessLayout::SingleProcess, |rank| {
                ProcessLayout::ProcessPerRank { rank, world: 2 }
            })
    }
}

fn mesh(suffix: &'static str) -> DeviceMesh<Mesh> {
    DeviceMesh::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &TwoCuda {
            suffix,
            local_rank: None,
        },
    )
    .unwrap()
}

fn network_mesh(local_rank: usize) -> DeviceMesh<Mesh> {
    DeviceMesh::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &TwoCuda {
            suffix: "network",
            local_rank: Some(local_rank),
        },
    )
    .unwrap()
}

fn two_collective_plan(bound: &DeviceMesh<Mesh>) -> CollectivePlan {
    let mut builder = CollectivePlanBuilder::new(bound);
    let gather = builder
        .push_static::<f32, Shard, Replica>(MeshAxis::Tensor, 0, 8, StreamId::new(2), None)
        .unwrap();
    builder
        .push_static::<f32, PartialSum, Replica>(
            MeshAxis::Tensor,
            0,
            16,
            StreamId::new(3),
            Some(gather),
        )
        .unwrap();
    builder.finish()
}

#[test]
fn static_descriptors_derive_kind_counts_bytes_sequence_and_dependencies() {
    let bound = mesh("a");
    let plan = two_collective_plan(&bound);
    let gather = &plan.descriptors()[0];
    let reduce = &plan.descriptors()[1];

    assert_eq!(gather.kind(), CollectiveKind::AllGather);
    assert_eq!(gather.group().ranks(), 2);
    assert_eq!(gather.input_elements(), 8);
    assert_eq!(gather.output_elements(), 16);
    assert_eq!(gather.input_bytes(), 32);
    assert_eq!(gather.output_bytes(), 64);
    assert_eq!(gather.dtype(), DTypeId::F32);
    assert_eq!(gather.source(), PlacementKind::Sharded { axis: 1 });
    assert_eq!(gather.destination(), PlacementKind::Replicated);
    assert_eq!(gather.sequence().get(), 0);
    assert_eq!(gather.depends_on(), None);

    assert_eq!(reduce.kind(), CollectiveKind::AllReduce(ReduceOp::Sum));
    assert_eq!(reduce.input_elements(), 16);
    assert_eq!(reduce.output_elements(), 16);
    assert_eq!(reduce.sequence().get(), 1);
    assert_eq!(reduce.depends_on(), Some(gather.sequence()));
}

#[test]
fn dynamic_dtype_and_placement_take_the_same_checked_path() {
    let bound = mesh("a");
    let mut builder = CollectivePlanBuilder::new(&bound);
    builder
        .push_dyn(
            MeshAxis::Tensor,
            1,
            6,
            DTypeId::F64,
            PlacementKind::Sharded { axis: 1 },
            PlacementKind::Replicated,
            StreamId::default(),
            None,
        )
        .unwrap();
    let plan = builder.finish();
    let descriptor = &plan.descriptors()[0];

    assert_eq!(descriptor.kind(), CollectiveKind::AllGather);
    assert_eq!(descriptor.input_bytes(), 48);
    assert_eq!(descriptor.output_bytes(), 96);

    let mut unsupported = CollectivePlanBuilder::new(&bound);
    assert_eq!(
        unsupported
            .push_dyn(
                MeshAxis::Tensor,
                0,
                32,
                DTypeId::Q8_0,
                PlacementKind::Sharded { axis: 1 },
                PlacementKind::Replicated,
                StreamId::default(),
                None,
            )
            .unwrap_err(),
        PlanError::Collective(CollectiveError::UnsupportedDType {
            dtype: DTypeId::Q8_0
        })
    );

    let mut illegal = CollectivePlanBuilder::new(&bound);
    assert!(matches!(
        illegal.push_dyn(
            MeshAxis::Tensor,
            0,
            8,
            DTypeId::F32,
            PlacementKind::Sharded { axis: 0 },
            PlacementKind::Sharded { axis: 1 },
            StreamId::default(),
            None,
        ),
        Err(PlanError::Distributed(_))
    ));

    let mut integer_mean = CollectivePlanBuilder::new(&bound);
    assert_eq!(
        integer_mean
            .push_dyn(
                MeshAxis::Tensor,
                0,
                4,
                DTypeId::U32,
                PlacementKind::Partial {
                    reduction: ReduceOp::Mean,
                },
                PlacementKind::Replicated,
                StreamId::default(),
                None,
            )
            .unwrap_err(),
        PlanError::Collective(CollectiveError::UnsupportedReduction {
            dtype: DTypeId::U32,
            op: ReduceOp::Mean,
        })
    );

    let mut wrong_group = CollectivePlanBuilder::new(&bound);
    assert!(matches!(
        wrong_group.push_dyn(
            MeshAxis::Data,
            0,
            8,
            DTypeId::F32,
            PlacementKind::Sharded { axis: 1 },
            PlacementKind::Replicated,
            StreamId::default(),
            None,
        ),
        Err(PlanError::WrongAxis {
            expected: MeshAxis::Tensor,
            found: MeshAxis::Data,
            ..
        })
    ));
}

#[test]
fn reduce_scatter_derives_rank_local_output_and_rejects_a_remainder() {
    let bound = mesh("a");
    let mut builder = CollectivePlanBuilder::new(&bound);
    builder
        .push_static::<f32, PartialSum, Shard>(MeshAxis::Tensor, 0, 16, StreamId::default(), None)
        .unwrap();
    let plan = builder.finish();
    assert_eq!(
        plan.descriptors()[0].kind(),
        CollectiveKind::ReduceScatter(ReduceOp::Sum)
    );
    assert_eq!(plan.descriptors()[0].output_elements(), 8);

    let mut rejected = CollectivePlanBuilder::new(&bound);
    assert_eq!(
        rejected
            .push_static::<f32, PartialSum, Shard>(
                MeshAxis::Tensor,
                0,
                15,
                StreamId::default(),
                None,
            )
            .unwrap_err(),
        PlanError::Collective(CollectiveError::NonDivisible {
            elements: 15,
            ranks: 2
        })
    );
}

#[test]
fn builder_rejects_rank_dependency_and_non_collective_mistakes() {
    let bound = mesh("a");
    let mut bad_rank = CollectivePlanBuilder::new(&bound);
    assert_eq!(
        bad_rank
            .push_static::<f32, Shard, Replica>(MeshAxis::Tensor, 2, 8, StreamId::default(), None,)
            .unwrap_err(),
        PlanError::RankOutOfRange { rank: 2, world: 2 }
    );

    let mut bad_dependency = CollectivePlanBuilder::new(&bound);
    let foreign = {
        let mut other = CollectivePlanBuilder::new(&bound);
        other
            .push_static::<f32, Shard, Replica>(MeshAxis::Tensor, 0, 8, StreamId::default(), None)
            .unwrap()
    };
    assert!(matches!(
        bad_dependency.push_static::<f32, Shard, Replica>(
            MeshAxis::Tensor,
            0,
            8,
            StreamId::default(),
            Some(foreign),
        ),
        Err(PlanError::UnknownDependency { .. })
    ));
}

#[test]
fn equal_plans_hash_identically_and_preflight_mints_agreement() {
    let bound = mesh("a");
    let rank_zero = two_collective_plan(&bound);
    let rank_one = two_collective_plan(&bound);

    assert_eq!(rank_zero.hash(), rank_one.hash());
    let agreed = preflight(2, &[rank_zero.summary(), rank_one.summary()]).unwrap();
    assert_eq!(agreed.ranks(), 2);
    assert_eq!(agreed.summary(), rank_zero.summary());
}

#[test]
fn two_network_processes_derive_one_mesh_and_one_plan_identity() {
    let rank_zero_mesh = network_mesh(0);
    let rank_one_mesh = network_mesh(1);
    assert_eq!(rank_zero_mesh.id(), rank_one_mesh.id());

    let rank_zero_plan = two_collective_plan(&rank_zero_mesh);
    let rank_one_plan = two_collective_plan(&rank_one_mesh);
    assert_eq!(rank_zero_plan.hash(), rank_one_plan.hash());
    preflight(2, &[rank_zero_plan.summary(), rank_one_plan.summary()]).unwrap();
}

#[test]
fn preflight_names_count_hash_and_mesh_divergence_before_launch() {
    let bound = mesh("a");
    let expected = two_collective_plan(&bound);

    let mut shorter_builder = CollectivePlanBuilder::new(&bound);
    shorter_builder
        .push_static::<f32, Shard, Replica>(MeshAxis::Tensor, 0, 8, StreamId::default(), None)
        .unwrap();
    let shorter = shorter_builder.finish();
    assert!(matches!(
        preflight(2, &[expected.summary(), shorter.summary()]),
        Err(PlanError::CollectiveCountMismatch { rank: 1, .. })
    ));

    let mut different_builder = CollectivePlanBuilder::new(&bound);
    let different_first = different_builder
        .push_static::<f64, Shard, Replica>(MeshAxis::Tensor, 0, 8, StreamId::new(2), None)
        .unwrap();
    different_builder
        .push_static::<f32, PartialSum, Replica>(
            MeshAxis::Tensor,
            0,
            16,
            StreamId::new(3),
            Some(different_first),
        )
        .unwrap();
    let different = different_builder.finish();
    assert!(matches!(
        preflight(2, &[expected.summary(), different.summary()]),
        Err(PlanError::PlanHashMismatch { rank: 1, .. })
    ));

    let other_mesh = mesh("b");
    let other_plan = two_collective_plan(&other_mesh);
    assert!(matches!(
        preflight(2, &[expected.summary(), other_plan.summary()]),
        Err(PlanError::MeshMismatch { rank: 1, .. })
    ));
}

#[test]
fn static_plan_rejections_are_compile_errors() {
    if std::fs::read("/home/xupremix/.cargo/config.toml").is_err() {
        return;
    }
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/collective_plan_compile_fail/*.rs");
    if std::env::var_os("TRYBUILD").as_deref() != Some(std::ffi::OsStr::new("overwrite")) {
        support::compile_fail_cases_name_their_reason(
            Path::new("tests/collective_plan_compile_fail"),
            &BTreeMap::from([
                ("cross_mesh_placement", "E0277"),
                ("illegal_static_transition", "E0277"),
                ("integer_mean", "E0277"),
                ("non_collective_static_transition", "E0277"),
                ("q8_collective_plan", "E0277"),
            ]),
        );
    }
}
