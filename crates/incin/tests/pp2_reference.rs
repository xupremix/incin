//! Microbatched 2-stage pipeline over the reference transport
//! (issue #99, in-process slice).
//!
//! A [`PipelinePlan`](incin::experimental::distributed::PipelinePlan) fixes
//! the microbatch count, the stage-boundary transfers, and the
//! [`OneForwardOneBackward`] clock timeline - including its bubble slots.
//! This file drives that exact timeline with real CPU modules: stage 0 is
//! a `Linear + ReLU`, stage 1 a `Linear`, and every forward handoff moves
//! through the reference transport's `send_recv` under the plan's own
//! transfer descriptor (group, stream, source, destination). The
//! per-microbatch outputs must equal the single-device sequential run,
//! and the idle slots the driver observes must equal the plan's
//! `bubble_slots`.
//!
//! The backward half of the timeline is walked but not executed: the
//! autograd has no cross-graph handoff (detach plus external-gradient
//! backward), so there is no honest way to run a stage-1 gradient into a
//! stage-0 backward in-process. The driver counts those slots as
//! skipped-by-design against the same clocks, and the planning layer's
//! `validate_clocks` proof - every forward precedes its backward, every
//! stage covers every microbatch - is what keeps the unexecuted half
//! honest. GPipe is the same story one schedule over: its numerics cross
//! the reference transport in `incin-backends`' `pipeline_reference`
//! tests, and only 1F1B is executed here.
//!
//! Everything runs on the CPU with no hardware: the mesh binds two CUDA
//! stand-ins for planning while every tensor stays on the CPU backend.
//! The two-host NCCL leg stays hardware-gated (issue #82). Bubble numbers
//! below are CPU-only slot counts from the plan's logical clocks, not
//! device measurements.

#![cfg(all(feature = "cpu", feature = "distributed-reference"))]

use incin::backend_authoring::HostReadback;
use incin::experimental::distributed::mesh::{
    DeviceIdentity, DeviceMesh, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin::experimental::distributed::{
    ActivationCheckpoint, CollectiveKind, OneForwardOneBackward, PipelineAction,
    PipelineBoundaryId, PipelineError, PipelinePlanBuilder, PipelineTransfer, StreamId,
    TwoRankPipeline,
};
use incin::prelude::*;
use incin::state::{collect_state, load_state};
use incin_backends::dist::{
    CollectiveBackend, ReferenceBuffer, ReferenceTransport, ReferenceValues,
};
use std::collections::BTreeMap;

type Backend = incin::DefaultBackend;

/// In-process planning probe: two reachable CUDA stand-ins, no hardware.
struct ReferencePp2;

impl TopologyProbe for ReferencePp2 {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        (device.kind() == DeviceKind::Cuda && device.ordinal() < 2).then(|| {
            DeviceIdentity::new(
                device,
                format!("reference-pp2-exec-{}", device.ordinal()),
                "sm_reference".to_string(),
            )
        })
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
        ProcessLayout::SingleProcess
    }
}

fn mesh() -> DeviceMesh<TwoRankPipeline> {
    DeviceMesh::bind(&[DeviceId::cuda(0), DeviceId::cuda(1)], &ReferencePp2).unwrap()
}

fn f32_buffer(values: Vec<f32>) -> ReferenceBuffer<f32> {
    ReferenceBuffer::try_new(ReferenceValues::F32(values), Default::default()).unwrap()
}

fn read_f32<K: DType>(
    storage: &<Backend as incin::backend_authoring::StorageBackend>::Storage<K>,
) -> Vec<f64> {
    Backend::float_to_vec1::<K>(storage).expect("CPU readback works")
}

/// Four microbatches of `[2, 2]` inputs with fixed values.
fn microbatch_inputs() -> Result<Vec<Tensor<Dyn, Backend>>> {
    let mut inputs = Vec::new();
    for microbatch in 0..4 {
        let shift = microbatch as f32 * 0.25;
        let data: Vec<f32> = (0..4).map(|i| (i as f32) * 0.2 - 0.4 + shift).collect();
        inputs.push(Tensor::<Dyn, Backend>::from_slice(&data, vec![2, 2])?);
    }
    Ok(inputs)
}

/// A 1F1B schedule over 2 stages and 4 microbatches, driven clock by
/// clock from the plan's own timeline, reproduces the single-device
/// sequential outputs - and the idle slots it leaves behind are exactly
/// the plan's declared bubble.
#[test]
fn one_f_one_b_clocks_drive_matching_outputs_with_the_planned_bubble() -> Result<()> {
    const MICROBATCHES: usize = 4;

    let bound = mesh();
    let boundary = PipelineBoundaryId::new(7).unwrap();
    let plan = PipelinePlanBuilder::build_dyn(
        &bound,
        0,
        boundary,
        &[2, 4],
        DTypeId::F32,
        MICROBATCHES,
        incin::experimental::distributed::PipelineSchedule::OneForwardOneBackward,
        ActivationCheckpoint::Keep,
        StreamId::new(5),
    )
    .expect("a 1F1B plan builds");
    let schedule = plan.schedule();
    assert_eq!(schedule.microbatches(), MICROBATCHES);
    assert_eq!(
        plan.transfers().len(),
        MICROBATCHES * 2,
        "one forward plus one backward transfer per microbatch"
    );

    // Stages share one init with the single-device reference below.
    let stage0 = seq![Linear::<Dyn, Backend>::build((2, 4))?, ReLU];
    let stage1 = seq![Linear::<Dyn, Backend>::build((4, 2))?];
    let stage0_init = collect_state::<Backend, _>(&stage0)?;
    let stage1_init = collect_state::<Backend, _>(&stage1)?;
    let mut reference0 = seq![Linear::<Dyn, Backend>::build((2, 4))?, ReLU];
    let mut reference1 = seq![Linear::<Dyn, Backend>::build((4, 2))?];
    load_state::<Backend, _>(&mut reference0, &stage0_init)?;
    load_state::<Backend, _>(&mut reference1, &stage1_init)?;

    let inputs = microbatch_inputs()?;

    // Drive the plan's clocks. Forward actions execute; backward actions
    // are counted as skipped-by-design (no autograd handoff in-process);
    // empty slots are the bubble.
    let mut activations: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    let mut outputs: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    let mut executed_forward = 0;
    let mut skipped_backward = 0;
    let mut bubble_slots = 0;
    for clock in schedule.clocks() {
        for stage in 0..2 {
            match clock.stage(stage) {
                Some(PipelineAction::Forward { microbatch }) => {
                    executed_forward += 1;
                    if stage == 0 {
                        let activation = stage0.forward(inputs[microbatch].clone())?;
                        let values = read_f32::<f32>(activation.inner())
                            .iter()
                            .map(|value| *value as f32)
                            .collect::<Vec<f32>>();
                        assert_eq!(
                            values.len(),
                            8,
                            "stage-0 activation is [2, 4] per microbatch"
                        );
                        activations.insert(microbatch, values);
                    } else {
                        let handed = handoff_through_transport(
                            &plan,
                            boundary,
                            microbatch,
                            activations
                                .get(&microbatch)
                                .expect("stage 0 forwarded this microbatch first"),
                        )
                        .expect("the forward handoff moves through the transport");
                        let input = Tensor::<Dyn, Backend>::from_slice(&handed, vec![2, 4])?;
                        let output = stage1.forward(input)?;
                        let values = read_f32::<f32>(output.inner())
                            .iter()
                            .map(|value| *value as f32)
                            .collect::<Vec<f32>>();
                        outputs.insert(microbatch, values);
                    }
                }
                Some(PipelineAction::Backward { .. }) => {
                    // No cross-graph handoff exists in-process: counted,
                    // not executed, against the same clocks.
                    skipped_backward += 1;
                }
                None => bubble_slots += 1,
            }
        }
    }

    assert_eq!(
        executed_forward,
        MICROBATCHES * 2,
        "every microbatch forwards once per stage"
    );
    assert_eq!(
        skipped_backward,
        MICROBATCHES * 2,
        "the driver walked the full timeline including the backward half"
    );
    assert_eq!(
        outputs.len(),
        MICROBATCHES,
        "every microbatch leaves stage 1 with an output"
    );
    assert_eq!(
        bubble_slots,
        schedule.bubble_slots(),
        "the observed idle slots are exactly the planned bubble"
    );
    assert!(
        bubble_slots > 0,
        "a 2-stage pipeline over {MICROBATCHES} microbatches must idle somewhere"
    );
    eprintln!(
        "pp2 1f1b: {} clocks, {} bubble slots, {} skipped backward slots (CPU-only slot counts)",
        schedule.clocks().len(),
        bubble_slots,
        skipped_backward
    );

    // The single-device reference: stage 1 over stage 0, microbatch by
    // microbatch, no pipeline. Same weights, same kernels, same order
    // per microbatch - so the outputs agree bit-exactly.
    let mut max_abs = 0.0f64;
    for (microbatch, input) in inputs.iter().enumerate() {
        let expected = reference1.forward(reference0.forward(input.clone())?)?;
        let want = read_f32::<f32>(expected.inner());
        let got = outputs
            .get(&microbatch)
            .expect("every microbatch produced an output");
        assert_eq!(got.len(), want.len());
        for (index, (got, want)) in got.iter().zip(&want).enumerate() {
            max_abs = max_abs.max((*got as f64 - want).abs());
            assert_eq!(
                *got as f64, *want,
                "microbatch {microbatch} output {index} must match single-device"
            );
        }
    }
    eprintln!("pp2 1f1b: max abs output diff vs single-device {max_abs}");

    // Keep the marker type honest: the static schedule spelling selects
    // the same runtime schedule the driver executed.
    assert_eq!(
        <OneForwardOneBackward as incin::experimental::distributed::StaticPipelineSchedule>::SCHEDULE,
        incin::experimental::distributed::PipelineSchedule::OneForwardOneBackward
    );
    Ok(())
}

/// Moves one stage-0 activation to stage 1 through the reference
/// transport under the plan's own forward-transfer descriptor: the group,
/// stream, source, and destination the collective plan validated, looked
/// up by the transfer's stable plan tag.
fn handoff_through_transport(
    plan: &incin::experimental::distributed::PipelinePlan,
    boundary: PipelineBoundaryId,
    microbatch: usize,
    activation: &[f32],
) -> Result<Vec<f32>> {
    let tag = PipelineTransfer::ForwardActivation.plan_tag(boundary, microbatch);
    let descriptor = plan
        .collective_plan()
        .descriptors()
        .iter()
        .find(|descriptor| {
            descriptor.tag() == tag
                && descriptor.kind()
                    == CollectiveKind::SendRecv {
                        source: 0,
                        destination: 1,
                    }
        })
        .expect("the plan carries this microbatch's forward transfer");
    let zeros = vec![0.0f32; activation.len()];
    let moved = ReferenceTransport
        .send_recv::<f32>(
            descriptor.group(),
            &[f32_buffer(activation.to_vec()), f32_buffer(zeros)],
            PipelineTransfer::ForwardActivation.source_rank(),
            PipelineTransfer::ForwardActivation.destination_rank(),
            descriptor.stream(),
        )
        .expect("the reference send_recv moves the activation");
    let (buffers, _) = moved.into_parts();
    assert_eq!(buffers.len(), 2, "one output buffer per rank");
    let ReferenceValues::F32(stage0_slot) = buffers[0].values() else {
        panic!("the source slot keeps its own bytes");
    };
    let ReferenceValues::F32(stage1_slot) = buffers[1].values() else {
        panic!("the destination slot receives f32");
    };
    assert_eq!(
        stage0_slot,
        &activation.to_vec(),
        "the source rank's slot is untouched"
    );
    assert_eq!(
        stage1_slot,
        &activation.to_vec(),
        "the destination rank receives the activation bytes unchanged"
    );
    Ok(stage1_slot.clone())
}

/// Degenerate pipeline requests are refused at plan build: zero
/// microbatches cannot make progress, and boundary zero is reserved.
#[test]
fn degenerate_schedules_are_refused() {
    let bound = mesh();
    match PipelinePlanBuilder::build_dyn(
        &bound,
        0,
        PipelineBoundaryId::new(9).unwrap(),
        &[2, 4],
        DTypeId::F32,
        0,
        incin::experimental::distributed::PipelineSchedule::OneForwardOneBackward,
        ActivationCheckpoint::Keep,
        StreamId::new(5),
    ) {
        Err(PipelineError::ZeroMicrobatches) => {}
        other => panic!("expected ZeroMicrobatches, got {other:?}"),
    }
    match PipelineBoundaryId::new(0) {
        Err(PipelineError::ReservedBoundaryId) => {}
        other => panic!("expected ReservedBoundaryId, got {other:?}"),
    }
}
