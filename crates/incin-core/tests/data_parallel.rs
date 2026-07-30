//! `DST-008`: the local, transport-neutral half of DP=2 training.
//!
//! The two-host NCCL case lives in the facade because it needs CUDA, NCCL, and
//! sockets. These cases pin the same plan and arithmetic on the deterministic
//! transport, including static rejection and `Dyn` runtime guards.

#![cfg(feature = "distributed")]

mod support;

use std::collections::BTreeMap;
use std::path::Path;

use incin_core::dist::mesh::{
    DeviceIdentity, DeviceMesh, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin_core::dist::{
    CollectiveKind, DataParallelError, DataParallelPlan, DataParallelPlanBuilder, GradientId,
    PlacementKind, StreamId, TwoRankDataParallel, preflight,
};
use incin_core::exec::ReduceOp;
use incin_core::prelude::{DTypeId, DeviceId};

#[derive(Clone)]
struct TwoNetworkCuda {
    rank: usize,
}

impl TopologyProbe for TwoNetworkCuda {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        (device.kind() == incin_core::prelude::DeviceKind::Cuda && device.ordinal() < 2).then(
            || {
                DeviceIdentity::new(
                    device,
                    format!("GPU-DP2-{}", device.ordinal()),
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
        ProcessLayout::ProcessPerRank {
            rank: self.rank,
            world: 2,
        }
    }
}

fn mesh(rank: usize) -> DeviceMesh<TwoRankDataParallel> {
    DeviceMesh::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &TwoNetworkCuda { rank },
    )
    .unwrap()
}

fn static_plan(rank: usize, order: [u64; 2]) -> DataParallelPlan {
    let bound = mesh(rank);
    let mut builder = DataParallelPlanBuilder::new(&bound, rank);
    builder
        .push_static::<f32>(GradientId::new(order[0]).unwrap(), 2, StreamId::new(4))
        .unwrap();
    builder
        .push_static::<f64>(GradientId::new(order[1]).unwrap(), 3, StreamId::new(5))
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn static_dp2_gradients_are_ordered_mean_all_reduces() {
    let plan = static_plan(0, [11, 22]);
    let gradients = plan.gradients();
    let descriptors = plan.collective_plan().descriptors();

    assert_eq!(gradients.len(), 2);
    assert_eq!(gradients[0].id(), GradientId::new(11).unwrap());
    assert_eq!(gradients[0].dtype(), DTypeId::F32);
    assert_eq!(gradients[0].elements(), 2);
    assert_eq!(gradients[1].dtype(), DTypeId::F64);

    assert_eq!(
        descriptors[0].kind(),
        CollectiveKind::AllReduce(ReduceOp::Mean)
    );
    assert_eq!(
        descriptors[0].source(),
        PlacementKind::Partial {
            reduction: ReduceOp::Mean
        }
    );
    assert_eq!(descriptors[0].destination(), PlacementKind::Replicated);
    assert_eq!(descriptors[0].group().ranks(), 2);
    assert_eq!(descriptors[0].tag().get(), 11);
    assert_eq!(descriptors[0].depends_on(), None);
    assert_eq!(descriptors[1].tag().get(), 22);
    assert_eq!(descriptors[1].depends_on(), Some(descriptors[0].sequence()));
}

#[test]
fn equal_network_ranks_preflight_and_same_shaped_reordering_does_not() {
    let rank_zero = static_plan(0, [11, 22]);
    let rank_one = static_plan(1, [11, 22]);
    preflight(
        2,
        &[
            rank_zero.collective_plan().summary(),
            rank_one.collective_plan().summary(),
        ],
    )
    .unwrap();

    let first = {
        let bound = mesh(0);
        let mut builder = DataParallelPlanBuilder::new(&bound, 0);
        builder
            .push_static::<f32>(GradientId::new(11).unwrap(), 2, StreamId::default())
            .unwrap();
        builder
            .push_static::<f32>(GradientId::new(22).unwrap(), 2, StreamId::default())
            .unwrap();
        builder.finish().unwrap()
    };
    let swapped = {
        let bound = mesh(1);
        let mut builder = DataParallelPlanBuilder::new(&bound, 1);
        builder
            .push_static::<f32>(GradientId::new(22).unwrap(), 2, StreamId::default())
            .unwrap();
        builder
            .push_static::<f32>(GradientId::new(11).unwrap(), 2, StreamId::default())
            .unwrap();
        builder.finish().unwrap()
    };
    assert_ne!(
        first.collective_plan().hash(),
        swapped.collective_plan().hash(),
        "parameter identity must distinguish same-shaped gradient reordering"
    );
}

#[test]
fn dyn_accepts_floats_and_rejects_integer_and_quantized_gradients() {
    let bound = mesh(0);
    let mut builder = DataParallelPlanBuilder::new(&bound, 0);
    builder
        .push_dyn(
            GradientId::new(1).unwrap(),
            4,
            DTypeId::F64,
            StreamId::default(),
        )
        .unwrap();
    let plan = builder.finish().unwrap();
    assert_eq!(plan.gradients()[0].dtype(), DTypeId::F64);

    for (offset, dtype) in [DTypeId::U8, DTypeId::U32, DTypeId::I64, DTypeId::Q8_0]
        .into_iter()
        .enumerate()
    {
        let bound = mesh(0);
        let mut rejected = DataParallelPlanBuilder::new(&bound, 0);
        assert_eq!(
            rejected
                .push_dyn(
                    GradientId::new(10 + offset as u64).unwrap(),
                    32,
                    dtype,
                    StreamId::default(),
                )
                .unwrap_err(),
            DataParallelError::UnsupportedGradientDType { dtype }
        );
    }
}

#[test]
fn duplicate_reserved_empty_and_bad_rank_plans_are_rejected() {
    assert_eq!(
        GradientId::new(0),
        Err(DataParallelError::ReservedGradientId)
    );

    let bound = mesh(0);
    assert_eq!(
        DataParallelPlanBuilder::new(&bound, 0).finish(),
        Err(DataParallelError::NoGradients)
    );

    let mut duplicate = DataParallelPlanBuilder::new(&bound, 0);
    let id = GradientId::new(9).unwrap();
    duplicate
        .push_static::<f32>(id, 2, StreamId::default())
        .unwrap();
    assert_eq!(
        duplicate
            .push_static::<f32>(id, 2, StreamId::default())
            .unwrap_err(),
        DataParallelError::DuplicateGradient { id }
    );

    let mut bad_rank = DataParallelPlanBuilder::new(&bound, 2);
    assert!(matches!(
        bad_rank.push_static::<f32>(GradientId::new(10).unwrap(), 2, StreamId::default()),
        Err(DataParallelError::Plan(_))
    ));
}

#[test]
fn static_dp_contract_rejections_are_compile_errors() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/data_parallel_compile_fail/*.rs");
    if std::env::var_os("TRYBUILD").as_deref() != Some(std::ffi::OsStr::new("overwrite")) {
        support::compile_fail_cases_name_their_reason(
            Path::new("tests/data_parallel_compile_fail"),
            &BTreeMap::from([
                ("integer_gradient", "E0277"),
                ("q8_gradient", "E0277"),
                ("wrong_mesh", "E0308"),
            ]),
        );
    }
}
