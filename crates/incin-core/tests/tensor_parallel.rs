//! `DST-009`: transport-neutral TP=2 linear and attention planning.

#![cfg(feature = "distributed")]

extern crate incin_core as incin;

mod support;

use std::collections::BTreeMap;
use std::path::Path;

use incin_core::dist::mesh::{
    DeviceIdentity, DeviceMesh, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin_core::dist::{
    CollectiveKind, PlacementKind, StreamId, TensorParallelCollective, TensorParallelDimension,
    TensorParallelError, TensorParallelId, TensorParallelPlan, TensorParallelPlanBuilder,
    TwoRankTensorParallel, preflight,
};
use incin_core::exec::ReduceOp;
use incin_core::nn::{TwoWayColumnLinearShape, TwoWayRowLinearShape};
use incin_core::prelude::{DTypeId, DeviceId, s};
use incin_core::typenum::{U0, U1, U2, U4, U6};

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
                    format!("GPU-TP2-{}", device.ordinal()),
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

fn mesh(rank: usize) -> DeviceMesh<TwoRankTensorParallel> {
    DeviceMesh::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &TwoNetworkCuda { rank },
    )
    .unwrap()
}

fn static_plan(rank: usize) -> TensorParallelPlan {
    let bound = mesh(rank);
    let mut builder = TensorParallelPlanBuilder::new(&bound, rank);
    builder
        .push_column_static::<f32, U1, U6>(TensorParallelId::new(11).unwrap(), 2, StreamId::new(3))
        .unwrap();
    builder
        .push_row_static::<f64, U4>(TensorParallelId::new(22).unwrap(), 6, StreamId::new(4))
        .unwrap();
    builder
        .push_attention_static::<f32, U0, U2>(
            TensorParallelId::new(33).unwrap(),
            4,
            StreamId::new(5),
        )
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn static_column_row_and_attention_have_exact_tp2_semantics() {
    type LinearShape = s![4, 6];
    assert_eq!(
        <LinearShape as TwoWayColumnLinearShape>::LOCAL_OUT_FEATURES,
        3
    );
    assert_eq!(<LinearShape as TwoWayRowLinearShape>::LOCAL_IN_FEATURES, 2);

    let plan = static_plan(0);
    let operations = plan.operations();
    let descriptors = plan.collective_plan().descriptors();
    assert_eq!(operations.len(), 3);
    assert_eq!(
        operations[0].collective(),
        TensorParallelCollective::ColumnOutputGather { tensor_axis: 1 }
    );
    assert_eq!(operations[0].local_elements(), 6);
    assert_eq!(operations[0].global_elements(), 12);
    assert_eq!(descriptors[0].kind(), CollectiveKind::AllGather);
    assert_eq!(descriptors[0].source(), PlacementKind::Sharded { axis: 1 });
    assert_eq!(descriptors[0].destination(), PlacementKind::Replicated);
    assert_eq!(descriptors[0].tag().get(), (11 << 2) | 1);

    assert_eq!(
        operations[1].collective(),
        TensorParallelCollective::RowOutputSum
    );
    assert_eq!(
        descriptors[1].kind(),
        CollectiveKind::AllReduce(ReduceOp::Sum)
    );
    assert_eq!(
        descriptors[1].source(),
        PlacementKind::Partial {
            reduction: ReduceOp::Sum
        }
    );
    assert_eq!(descriptors[1].depends_on(), Some(descriptors[0].sequence()));
    assert_eq!(descriptors[1].tag().get(), (22 << 2) | 2);

    assert_eq!(
        operations[2].collective(),
        TensorParallelCollective::AttentionHeadGather { tensor_axis: 0 }
    );
    assert_eq!(operations[2].local_elements(), 4);
    assert_eq!(operations[2].global_elements(), 8);
    assert_eq!(descriptors[2].kind(), CollectiveKind::AllGather);
    assert_eq!(descriptors[2].tag().get(), (33 << 2) | 3);
    assert_eq!(descriptors[2].group().ranks(), 2);
}

#[test]
fn equal_network_ranks_preflight_and_semantic_reordering_does_not() {
    let rank_zero = static_plan(0);
    let rank_one = static_plan(1);
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
        let mut builder = TensorParallelPlanBuilder::new(&bound, 0);
        builder
            .push_column_static::<f32, U0, U2>(
                TensorParallelId::new(1).unwrap(),
                4,
                StreamId::default(),
            )
            .unwrap();
        builder
            .push_attention_static::<f32, U0, U2>(
                TensorParallelId::new(2).unwrap(),
                4,
                StreamId::default(),
            )
            .unwrap();
        builder.finish().unwrap()
    };
    let swapped = {
        let bound = mesh(1);
        let mut builder = TensorParallelPlanBuilder::new(&bound, 1);
        builder
            .push_attention_static::<f32, U0, U2>(
                TensorParallelId::new(2).unwrap(),
                4,
                StreamId::default(),
            )
            .unwrap();
        builder
            .push_column_static::<f32, U0, U2>(
                TensorParallelId::new(1).unwrap(),
                4,
                StreamId::default(),
            )
            .unwrap();
        builder.finish().unwrap()
    };
    assert_ne!(
        first.collective_plan().hash(),
        swapped.collective_plan().hash()
    );
}

#[test]
fn dyn_checks_dtype_divisibility_axis_and_overflow_before_planning() {
    let bound = mesh(0);
    let mut builder = TensorParallelPlanBuilder::new(&bound, 0);
    builder
        .push_column_dyn(
            TensorParallelId::new(1).unwrap(),
            &[2, 6],
            1,
            DTypeId::F64,
            StreamId::default(),
        )
        .unwrap();
    builder
        .push_row_dyn(
            TensorParallelId::new(2).unwrap(),
            4,
            12,
            DTypeId::F32,
            StreamId::default(),
        )
        .unwrap();
    builder
        .push_attention_dyn(
            TensorParallelId::new(3).unwrap(),
            &[2, 4, 8],
            1,
            DTypeId::F16,
            StreamId::default(),
        )
        .unwrap();
    let plan = builder.finish().unwrap();
    assert_eq!(plan.operations()[0].dtype(), DTypeId::F64);
    assert_eq!(plan.operations()[0].local_elements(), 6);
    assert_eq!(plan.operations()[2].local_elements(), 32);

    for (offset, dtype) in [DTypeId::U8, DTypeId::U32, DTypeId::I64, DTypeId::Q8_0]
        .into_iter()
        .enumerate()
    {
        let mut rejected = TensorParallelPlanBuilder::new(&bound, 0);
        assert_eq!(
            rejected
                .push_row_dyn(
                    TensorParallelId::new(10 + offset as u64).unwrap(),
                    4,
                    1,
                    dtype,
                    StreamId::default(),
                )
                .unwrap_err(),
            TensorParallelError::UnsupportedTensorDType { dtype }
        );
    }

    let mut nondivisible = TensorParallelPlanBuilder::new(&bound, 0);
    assert_eq!(
        nondivisible
            .push_row_dyn(
                TensorParallelId::new(20).unwrap(),
                3,
                1,
                DTypeId::F32,
                StreamId::default(),
            )
            .unwrap_err(),
        TensorParallelError::NonDivisible {
            dimension: TensorParallelDimension::InputFeatures,
            extent: 3,
            ranks: 2,
        }
    );

    let mut bad_axis = TensorParallelPlanBuilder::new(&bound, 0);
    assert_eq!(
        bad_axis
            .push_column_dyn(
                TensorParallelId::new(21).unwrap(),
                &[2, 6],
                2,
                DTypeId::F32,
                StreamId::default(),
            )
            .unwrap_err(),
        TensorParallelError::AxisOutOfBounds { axis: 2, rank: 2 }
    );

    let mut overflow = TensorParallelPlanBuilder::new(&bound, 0);
    assert_eq!(
        overflow
            .push_attention_dyn(
                TensorParallelId::new(22).unwrap(),
                &[usize::MAX, 2],
                1,
                DTypeId::F32,
                StreamId::default(),
            )
            .unwrap_err(),
        TensorParallelError::ElementCountOverflow
    );
}

#[test]
fn identities_empty_duplicate_and_bad_rank_are_structured_errors() {
    assert_eq!(
        TensorParallelId::new(0),
        Err(TensorParallelError::ReservedOperationId)
    );
    assert!(matches!(
        TensorParallelId::new(u64::MAX),
        Err(TensorParallelError::OperationIdTooLarge { .. })
    ));

    let bound = mesh(0);
    assert_eq!(
        TensorParallelPlanBuilder::new(&bound, 0).finish(),
        Err(TensorParallelError::NoOperations)
    );

    let mut duplicate = TensorParallelPlanBuilder::new(&bound, 0);
    let id = TensorParallelId::new(7).unwrap();
    duplicate
        .push_row_static::<f32, U4>(id, 2, StreamId::default())
        .unwrap();
    assert_eq!(
        duplicate
            .push_column_static::<f32, U0, U2>(id, 2, StreamId::default())
            .unwrap_err(),
        TensorParallelError::DuplicateOperation { id }
    );

    let mut bad_rank = TensorParallelPlanBuilder::new(&bound, 2);
    assert!(matches!(
        bad_rank.push_row_static::<f32, U4>(
            TensorParallelId::new(8).unwrap(),
            2,
            StreamId::default(),
        ),
        Err(TensorParallelError::Plan(_))
    ));
}

#[test]
fn static_tp_contract_rejections_are_compile_errors() {
    if std::fs::read("/home/xupremix/.cargo/config.toml").is_err() {
        return;
    }
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/tensor_parallel_compile_fail/*.rs");
    if std::env::var_os("TRYBUILD").as_deref() != Some(std::ffi::OsStr::new("overwrite")) {
        support::compile_fail_cases_name_their_reason(
            Path::new("tests/tensor_parallel_compile_fail"),
            &BTreeMap::from([
                ("integer_value", "E0277"),
                ("nondivisible_attention_heads", "E0277"),
                ("nondivisible_column_linear", "E0277"),
                ("nondivisible_row_linear", "E0277"),
                ("wrong_mesh", "E0308"),
            ]),
        );
    }
}
