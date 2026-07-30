//! CPU-runnable arithmetic evidence for the DP=2 gradient contract.

#![cfg(all(feature = "distributed-reference", feature = "cpu"))]

use incin_backends::dist::{
    CollectiveBackend, ReferenceBuffer, ReferenceTransport, ReferenceValues,
};
use incin_core::dist::mesh::{
    Data, DeviceIdentity, DeviceMesh, LinkClass, MeshSpec, ProcessLayout, TopologyProbe,
    TransportVersion,
};
use incin_core::dist::{DataParallelPlanBuilder, GradientId, StreamId, TwoRankDataParallel};
use incin_core::exec::ReduceOp;
use incin_core::prelude::{DTypeId, DeviceId, Dyn};

struct ReferenceDp2;

impl TopologyProbe for ReferenceDp2 {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        (device.kind() == incin_core::prelude::DeviceKind::Cuda && device.ordinal() < 2).then(
            || {
                DeviceIdentity::new(
                    device,
                    format!("reference-dp2-{}", device.ordinal()),
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

fn mesh() -> DeviceMesh<TwoRankDataParallel> {
    DeviceMesh::bind(&[DeviceId::cuda(0), DeviceId::cuda(1)], &ReferenceDp2).unwrap()
}

#[test]
fn mean_gradient_and_one_sgd_step_match_the_full_batch() {
    let bound = mesh();
    let mut builder = DataParallelPlanBuilder::new(&bound, 0);
    builder
        .push_static::<f32>(GradientId::new(7).unwrap(), 2, StreamId::new(1))
        .unwrap();
    let plan = builder.finish().unwrap();
    let descriptor = &plan.collective_plan().descriptors()[0];
    let local = [
        ReferenceBuffer::<f32>::try_new(ReferenceValues::F32(vec![2.0, 4.0]), Default::default())
            .unwrap(),
        ReferenceBuffer::<f32>::try_new(ReferenceValues::F32(vec![6.0, 8.0]), Default::default())
            .unwrap(),
    ];
    let output = ReferenceTransport
        .all_reduce(
            descriptor.group(),
            &local,
            ReduceOp::Mean,
            descriptor.stream(),
        )
        .unwrap();
    let ReferenceValues::F32(average) = output.buffers()[0].values() else {
        panic!("f32 plan produced another dtype");
    };

    let initial = [10.0, 20.0];
    let learning_rate = 0.25;
    let distributed = [
        initial[0] - learning_rate * average[0],
        initial[1] - learning_rate * average[1],
    ];
    let full_batch_gradient = [(2.0 + 6.0) / 2.0, (4.0 + 8.0) / 2.0];
    let single_device = [
        initial[0] - learning_rate * full_batch_gradient[0],
        initial[1] - learning_rate * full_batch_gradient[1],
    ];
    assert_eq!(distributed, single_device);
}

#[test]
fn dyn_f64_mean_uses_the_same_reference_transport_path() {
    let bound = mesh();
    let mut builder = DataParallelPlanBuilder::new(&bound, 0);
    builder
        .push_dyn(
            GradientId::new(8).unwrap(),
            1,
            DTypeId::F64,
            StreamId::default(),
        )
        .unwrap();
    let plan = builder.finish().unwrap();
    let descriptor = &plan.collective_plan().descriptors()[0];
    let buffers = [
        ReferenceBuffer::<Dyn>::try_new(ReferenceValues::F64(vec![1.0]), DTypeId::F64).unwrap(),
        ReferenceBuffer::<Dyn>::try_new(ReferenceValues::F64(vec![3.0]), DTypeId::F64).unwrap(),
    ];
    let reduced = ReferenceTransport
        .all_reduce(
            descriptor.group(),
            &buffers,
            ReduceOp::Mean,
            descriptor.stream(),
        )
        .unwrap();
    assert_eq!(
        reduced.buffers()[0].values(),
        &ReferenceValues::F64(vec![2.0])
    );
}

// Keep these imports/type constructions compiling beside the fixed alias. A
// mesh with tensor parallelism instead is rejected by the core trybuild suite.
#[allow(dead_code)]
type ExplicitDp2 = MeshSpec<Data<incin_core::typenum::U2>>;
