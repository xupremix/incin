//! CPU-runnable numerical evidence for PP=2 GPipe and 1F1B semantics.

#![cfg(all(feature = "distributed-reference", feature = "cpu"))]

use incin_backends::dist::{
    CollectiveBackend, ReferenceBuffer, ReferenceTransport, ReferenceValues,
};
use incin_core::dist::mesh::{
    DeviceIdentity, DeviceMesh, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin_core::dist::{
    ActivationCheckpoint, GPipe, OneForwardOneBackward, PipelineBoundaryId, PipelinePlan,
    PipelinePlanBuilder, PipelineTransfer, StreamId, TwoRankPipeline,
};
use incin_core::prelude::{DTypeId, DeviceId, Dyn};
use incin_core::typenum::{U2, U3};

struct ReferencePp2;

impl TopologyProbe for ReferencePp2 {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        (device.kind() == incin_core::prelude::DeviceKind::Cuda && device.ordinal() < 2).then(
            || {
                DeviceIdentity::new(
                    device,
                    format!("reference-pp2-{}", device.ordinal()),
                    "sm_reference".to_string(),
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
        ProcessLayout::SingleProcess
    }
}

fn mesh() -> DeviceMesh<TwoRankPipeline> {
    DeviceMesh::bind(&[DeviceId::cuda(0), DeviceId::cuda(1)], &ReferencePp2).unwrap()
}

#[test]
fn static_f32_gpipe_forward_and_gradients_match_one_device() {
    let plan = PipelinePlanBuilder::build_static::<
        f32,
        incin_core::shapes::DimCons<incin_core::shapes::Static<U2>, incin_core::shapes::Nil>,
        U3,
        GPipe,
    >(
        &mesh(),
        0,
        PipelineBoundaryId::new(1).unwrap(),
        ActivationCheckpoint::Keep,
        StreamId::new(1),
    )
    .unwrap();
    let inputs = [[1.0_f32, 2.0], [3.0, 4.0], [-1.0, 2.0]];
    let expected = reference_model_f32(&inputs);
    let found = execute_static_f32(&plan, &inputs);
    assert_eq!(found, expected);
}

#[test]
fn dyn_f64_one_f_one_b_uses_the_same_transfers_and_matches_one_device() {
    let plan = PipelinePlanBuilder::build_dyn(
        &mesh(),
        0,
        PipelineBoundaryId::new(2).unwrap(),
        &[2],
        DTypeId::F64,
        3,
        incin_core::dist::PipelineSchedule::OneForwardOneBackward,
        ActivationCheckpoint::Recompute,
        StreamId::new(2),
    )
    .unwrap();
    let inputs = [[1.0_f64, 2.0], [3.0, 4.0], [-1.0, 2.0]];
    let expected = reference_model_f64(&inputs);
    let found = execute_dyn_f64(&plan, &inputs);
    assert_eq!(found, expected);

    let static_plan = PipelinePlanBuilder::build_static::<
        f64,
        incin_core::shapes::DimCons<incin_core::shapes::Static<U2>, incin_core::shapes::Nil>,
        U3,
        OneForwardOneBackward,
    >(
        &mesh(),
        0,
        PipelineBoundaryId::new(2).unwrap(),
        ActivationCheckpoint::Recompute,
        StreamId::new(2),
    )
    .unwrap();
    assert_eq!(
        plan.collective_plan().hash(),
        static_plan.collective_plan().hash()
    );
}

fn execute_static_f32(
    plan: &PipelinePlan,
    inputs: &[[f32; 2]; 3],
) -> (Vec<f32>, [f32; 4], [f32; 2]) {
    let mut activations = [[0.0; 2]; 3];
    let mut activation_grads = [[0.0; 2]; 3];
    let mut outputs = vec![0.0; 3];
    let mut grad_w0 = [0.0; 4];
    let mut grad_w1 = [0.0; 2];
    for (semantic, descriptor) in plan
        .transfers()
        .iter()
        .zip(plan.collective_plan().descriptors())
    {
        let microbatch = semantic.microbatch();
        match semantic.transfer() {
            PipelineTransfer::ForwardActivation => {
                let activation = stage_zero_f32(inputs[microbatch]);
                let moved = ReferenceTransport
                    .send_recv(
                        descriptor.group(),
                        &[f32_buffer(activation.to_vec()), f32_buffer(vec![0.0; 2])],
                        0,
                        1,
                        descriptor.stream(),
                    )
                    .unwrap();
                let ReferenceValues::F32(values) = moved.buffers()[1].values() else {
                    unreachable!()
                };
                activations[microbatch].copy_from_slice(values);
                outputs[microbatch] = stage_one_f32(activations[microbatch]);
                grad_w1[0] += activations[microbatch][0];
                grad_w1[1] += activations[microbatch][1];
                activation_grads[microbatch] = [2.0, -1.0];
            }
            PipelineTransfer::BackwardGradient => {
                let moved = ReferenceTransport
                    .send_recv(
                        descriptor.group(),
                        &[
                            f32_buffer(vec![0.0; 2]),
                            f32_buffer(activation_grads[microbatch].to_vec()),
                        ],
                        1,
                        0,
                        descriptor.stream(),
                    )
                    .unwrap();
                let ReferenceValues::F32(values) = moved.buffers()[0].values() else {
                    unreachable!()
                };
                accumulate_w0_f32(&mut grad_w0, inputs[microbatch], [values[0], values[1]]);
            }
        }
    }
    (outputs, grad_w0, grad_w1)
}

fn execute_dyn_f64(plan: &PipelinePlan, inputs: &[[f64; 2]; 3]) -> (Vec<f64>, [f64; 4], [f64; 2]) {
    let mut activations = [[0.0; 2]; 3];
    let mut activation_grads = [[0.0; 2]; 3];
    let mut outputs = vec![0.0; 3];
    let mut grad_w0 = [0.0; 4];
    let mut grad_w1 = [0.0; 2];
    for (semantic, descriptor) in plan
        .transfers()
        .iter()
        .zip(plan.collective_plan().descriptors())
    {
        let microbatch = semantic.microbatch();
        let buffer = |values| {
            ReferenceBuffer::<Dyn>::try_new(ReferenceValues::F64(values), DTypeId::F64.into())
                .unwrap()
        };
        match semantic.transfer() {
            PipelineTransfer::ForwardActivation => {
                let activation = stage_zero_f64(inputs[microbatch]);
                let moved = ReferenceTransport
                    .send_recv(
                        descriptor.group(),
                        &[buffer(activation.to_vec()), buffer(vec![0.0; 2])],
                        0,
                        1,
                        descriptor.stream(),
                    )
                    .unwrap();
                let ReferenceValues::F64(values) = moved.buffers()[1].values() else {
                    unreachable!()
                };
                activations[microbatch].copy_from_slice(values);
                outputs[microbatch] = stage_one_f64(activations[microbatch]);
                grad_w1[0] += activations[microbatch][0];
                grad_w1[1] += activations[microbatch][1];
                activation_grads[microbatch] = [2.0, -1.0];
            }
            PipelineTransfer::BackwardGradient => {
                let moved = ReferenceTransport
                    .send_recv(
                        descriptor.group(),
                        &[
                            buffer(vec![0.0; 2]),
                            buffer(activation_grads[microbatch].to_vec()),
                        ],
                        1,
                        0,
                        descriptor.stream(),
                    )
                    .unwrap();
                let ReferenceValues::F64(values) = moved.buffers()[0].values() else {
                    unreachable!()
                };
                accumulate_w0_f64(&mut grad_w0, inputs[microbatch], [values[0], values[1]]);
            }
        }
    }
    (outputs, grad_w0, grad_w1)
}

fn reference_model_f32(inputs: &[[f32; 2]; 3]) -> (Vec<f32>, [f32; 4], [f32; 2]) {
    let mut outputs = Vec::new();
    let mut grad_w0 = [0.0; 4];
    let mut grad_w1 = [0.0; 2];
    for &input in inputs {
        let activation = stage_zero_f32(input);
        outputs.push(stage_one_f32(activation));
        grad_w1[0] += activation[0];
        grad_w1[1] += activation[1];
        accumulate_w0_f32(&mut grad_w0, input, [2.0, -1.0]);
    }
    (outputs, grad_w0, grad_w1)
}

fn reference_model_f64(inputs: &[[f64; 2]; 3]) -> (Vec<f64>, [f64; 4], [f64; 2]) {
    let mut outputs = Vec::new();
    let mut grad_w0 = [0.0; 4];
    let mut grad_w1 = [0.0; 2];
    for &input in inputs {
        let activation = stage_zero_f64(input);
        outputs.push(stage_one_f64(activation));
        grad_w1[0] += activation[0];
        grad_w1[1] += activation[1];
        accumulate_w0_f64(&mut grad_w0, input, [2.0, -1.0]);
    }
    (outputs, grad_w0, grad_w1)
}

fn stage_zero_f32([x0, x1]: [f32; 2]) -> [f32; 2] {
    [x0 + 2.0 * x1, 3.0 * x0 - x1]
}

fn stage_one_f32([a0, a1]: [f32; 2]) -> f32 {
    2.0 * a0 - a1
}

fn accumulate_w0_f32(grad: &mut [f32; 4], [x0, x1]: [f32; 2], [g0, g1]: [f32; 2]) {
    grad[0] += g0 * x0;
    grad[1] += g0 * x1;
    grad[2] += g1 * x0;
    grad[3] += g1 * x1;
}

fn stage_zero_f64([x0, x1]: [f64; 2]) -> [f64; 2] {
    [x0 + 2.0 * x1, 3.0 * x0 - x1]
}

fn stage_one_f64([a0, a1]: [f64; 2]) -> f64 {
    2.0 * a0 - a1
}

fn accumulate_w0_f64(grad: &mut [f64; 4], [x0, x1]: [f64; 2], [g0, g1]: [f64; 2]) {
    grad[0] += g0 * x0;
    grad[1] += g0 * x1;
    grad[2] += g1 * x0;
    grad[3] += g1 * x1;
}

fn f32_buffer(values: Vec<f32>) -> ReferenceBuffer<f32> {
    ReferenceBuffer::try_new(ReferenceValues::F32(values), Default::default()).unwrap()
}
