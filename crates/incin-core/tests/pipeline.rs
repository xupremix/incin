//! `DST-010`: transport-neutral PP=2 planning and schedule evidence.

#![cfg(feature = "distributed")]

mod support;

use std::collections::BTreeMap;
use std::path::Path;

use incin_core::dist::mesh::{
    DeviceIdentity, DeviceMesh, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin_core::dist::{
    ActivationCheckpoint, CollectiveKind, GPipe, OneForwardOneBackward, PipelineAction,
    PipelineBoundaryId, PipelineError, PipelinePhase, PipelinePlan, PipelinePlanBuilder,
    PipelineSchedule, PipelineTransfer, PlacementKind, StreamId, TwoRankPipeline, preflight,
};
use incin_core::prelude::{DTypeId, DeviceId};
use incin_core::typenum::{U2, U3, U4};

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
                    format!("GPU-PP2-{}", device.ordinal()),
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

fn mesh(rank: usize) -> DeviceMesh<TwoRankPipeline> {
    DeviceMesh::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &TwoNetworkCuda { rank },
    )
    .unwrap()
}

fn static_gpipe(rank: usize) -> PipelinePlan {
    PipelinePlanBuilder::build_static::<f32, (U2, U3), U4, GPipe>(
        &mesh(rank),
        rank,
        PipelineBoundaryId::new(7).unwrap(),
        ActivationCheckpoint::Keep,
        StreamId::new(9),
    )
    .unwrap()
}

#[test]
fn static_gpipe_has_global_send_recv_descriptors_and_exact_bubbles() {
    let plan = static_gpipe(0);
    let schedule = plan.schedule();
    assert_eq!(schedule.schedule(), PipelineSchedule::GPipe);
    assert_eq!(schedule.microbatches(), 4);
    assert_eq!(schedule.warmup_steps(), 5);
    assert_eq!(schedule.steady_steps(), 0);
    assert_eq!(schedule.cooldown_steps(), 5);
    assert_eq!(schedule.clocks().len(), 10);
    assert_eq!(schedule.bubble_slots(), 4);
    assert_eq!(schedule.max_live_activations(), [4, 4]);
    assert_eq!(
        schedule.clocks()[0].stage(0),
        Some(PipelineAction::Forward { microbatch: 0 })
    );
    assert_eq!(schedule.clocks()[0].stage(1), None);
    assert_eq!(schedule.clocks()[0].phase(), PipelinePhase::Warmup);

    let transfers = plan.transfers();
    let descriptors = plan.collective_plan().descriptors();
    assert_eq!(transfers.len(), 8);
    assert_eq!(transfers[0].transfer(), PipelineTransfer::ForwardActivation);
    assert_eq!(transfers[0].microbatch(), 0);
    assert_eq!(transfers[0].elements(), 6);
    assert_eq!(transfers[4].transfer(), PipelineTransfer::BackwardGradient);
    assert_eq!(transfers[4].microbatch(), 3);
    assert_eq!(
        descriptors[0].kind(),
        CollectiveKind::SendRecv {
            source: 0,
            destination: 1
        }
    );
    assert_eq!(
        descriptors[0].source(),
        PlacementKind::PipelineStage { index: 0 }
    );
    assert_eq!(
        descriptors[0].destination(),
        PlacementKind::PipelineStage { index: 1 }
    );
    assert_eq!(descriptors[0].group().ranks(), 2);
    assert_eq!(descriptors[0].input_elements(), 6);
    assert_eq!(descriptors[0].output_elements(), 6);
    assert_eq!(descriptors[0].tag().get(), 7 << 33);
    for pair in descriptors.windows(2) {
        assert_eq!(pair[1].depends_on(), Some(pair[0].sequence()));
    }
}

#[test]
fn static_one_f_one_b_reduces_peak_activation_residency() {
    let plan = PipelinePlanBuilder::build_static::<f64, (U2, U3), U4, OneForwardOneBackward>(
        &mesh(0),
        0,
        PipelineBoundaryId::new(8).unwrap(),
        ActivationCheckpoint::Recompute,
        StreamId::new(10),
    )
    .unwrap();
    let schedule = plan.schedule();
    assert_eq!(schedule.schedule(), PipelineSchedule::OneForwardOneBackward);
    assert_eq!(schedule.warmup_steps(), 2);
    assert_eq!(schedule.steady_steps(), 6);
    assert_eq!(schedule.cooldown_steps(), 2);
    assert_eq!(schedule.clocks().len(), 10);
    assert_eq!(schedule.bubble_slots(), 4);
    assert_eq!(schedule.max_live_activations(), [2, 1]);

    assert_eq!(
        plan.transfers()
            .iter()
            .map(|transfer| (transfer.transfer(), transfer.microbatch()))
            .collect::<Vec<_>>(),
        vec![
            (PipelineTransfer::ForwardActivation, 0),
            (PipelineTransfer::ForwardActivation, 1),
            (PipelineTransfer::BackwardGradient, 0),
            (PipelineTransfer::ForwardActivation, 2),
            (PipelineTransfer::BackwardGradient, 1),
            (PipelineTransfer::ForwardActivation, 3),
            (PipelineTransfer::BackwardGradient, 2),
            (PipelineTransfer::BackwardGradient, 3),
        ]
    );
}

#[test]
fn static_and_dyn_rank_plans_agree_but_semantic_drift_does_not() {
    let rank_zero = static_gpipe(0);
    let rank_one = static_gpipe(1);
    preflight(
        2,
        &[
            rank_zero.collective_plan().summary(),
            rank_one.collective_plan().summary(),
        ],
    )
    .unwrap();

    let dynamic = PipelinePlanBuilder::build_dyn(
        &mesh(1),
        1,
        PipelineBoundaryId::new(7).unwrap(),
        &[2, 3],
        DTypeId::F32,
        4,
        PipelineSchedule::GPipe,
        ActivationCheckpoint::Keep,
        StreamId::new(9),
    )
    .unwrap();
    assert_eq!(
        rank_zero.collective_plan().hash(),
        dynamic.collective_plan().hash()
    );

    let different_schedule = PipelinePlanBuilder::build_dyn(
        &mesh(1),
        1,
        PipelineBoundaryId::new(7).unwrap(),
        &[2, 3],
        DTypeId::F32,
        4,
        PipelineSchedule::OneForwardOneBackward,
        ActivationCheckpoint::Keep,
        StreamId::new(9),
    )
    .unwrap();
    assert_ne!(
        rank_zero.collective_plan().hash(),
        different_schedule.collective_plan().hash()
    );
}

#[test]
fn dyn_rejects_every_static_dtype_violation_and_runtime_geometry_error() {
    let bound = mesh(0);
    for dtype in [DTypeId::U8, DTypeId::U32, DTypeId::I64, DTypeId::Q8_0] {
        assert_eq!(
            PipelinePlanBuilder::build_dyn(
                &bound,
                0,
                PipelineBoundaryId::new(1).unwrap(),
                &[2, 3],
                dtype,
                2,
                PipelineSchedule::GPipe,
                ActivationCheckpoint::Keep,
                StreamId::default(),
            )
            .unwrap_err(),
            PipelineError::UnsupportedDType { dtype }
        );
    }

    assert_eq!(
        PipelinePlanBuilder::build_dyn(
            &bound,
            0,
            PipelineBoundaryId::new(1).unwrap(),
            &[2, 3],
            DTypeId::F32,
            0,
            PipelineSchedule::GPipe,
            ActivationCheckpoint::Keep,
            StreamId::default(),
        )
        .unwrap_err(),
        PipelineError::ZeroMicrobatches
    );
    assert!(matches!(
        PipelinePlanBuilder::build_dyn(
            &bound,
            0,
            PipelineBoundaryId::new(1).unwrap(),
            &[usize::MAX, 2],
            DTypeId::F32,
            1,
            PipelineSchedule::GPipe,
            ActivationCheckpoint::Keep,
            StreamId::default(),
        ),
        Err(PipelineError::Shape(_))
    ));
    assert!(matches!(
        PipelinePlanBuilder::build_dyn(
            &bound,
            2,
            PipelineBoundaryId::new(1).unwrap(),
            &[2, 3],
            DTypeId::F32,
            1,
            PipelineSchedule::GPipe,
            ActivationCheckpoint::Keep,
            StreamId::default(),
        ),
        Err(PipelineError::Plan(_))
    ));
    assert_eq!(
        PipelineBoundaryId::new(0),
        Err(PipelineError::ReservedBoundaryId)
    );
    assert!(matches!(
        PipelineBoundaryId::new(u64::MAX),
        Err(PipelineError::BoundaryIdTooLarge { .. })
    ));
}

#[test]
fn static_pipeline_contract_rejections_are_compile_errors() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/pipeline_compile_fail/*.rs");
    if std::env::var_os("TRYBUILD").as_deref() != Some(std::ffi::OsStr::new("overwrite")) {
        support::compile_fail_cases_name_their_reason(
            Path::new("tests/pipeline_compile_fail"),
            &BTreeMap::from([
                ("integer_dtype", "E0277"),
                ("too_many_static_microbatches", "E0271"),
                ("wrong_mesh", "E0308"),
                ("zero_microbatches", "E0277"),
            ]),
        );
    }
}
