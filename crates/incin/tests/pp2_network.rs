//! Two-host NCCL connectivity evidence for `DST-010`.
//!
//! These ignored tests validate pipeline plan construction and communicator
//! setup. They require two network-accessible CUDA hosts with NCCL. Tensor
//! kernel parity remains covered by backend-specific tests.

#![cfg(all(feature = "distributed-nccl", feature = "hardware-tests"))]

use incin::experimental::distributed::{
    ActivationCheckpoint, DistributedContext, GPipe, NcclTopology, NcclTransport,
    PipelineBoundaryId, PipelinePlanBuilder, StreamId, TwoRankPipeline,
};
use incin::prelude::*;
use incin::typenum::{U2, U4};
use incin::types::{DimCons, Nil};

#[test]
#[ignore = "requires two network-accessible CUDA hosts with NCCL"]
fn pp2_plan_and_communicator_initialize() {
    let context = DistributedContext::<Dyn, Dyn>::from_env().expect("two-rank rendezvous");
    let rank = context.rank();
    let topology = NcclTopology::discover_context(&context).expect("discover CUDA identities");
    let mesh = incin::experimental::distributed::mesh::DeviceMesh::<TwoRankPipeline>::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &topology,
    )
    .expect("bind PP=2 network topology");
    let plan = PipelinePlanBuilder::build_static::<f32, DimCons<U2, Nil>, U4, GPipe>(
        &mesh,
        rank,
        PipelineBoundaryId::new(301).unwrap(),
        ActivationCheckpoint::Keep,
        StreamId::new(0),
    )
    .expect("build PP=2 GPipe plan");
    let transport = NcclTransport::connect_context(&context, plan.into_collective_plan())
        .expect("initialize PP NCCL communicator");
    assert_eq!(transport.cursor(), 0);
    drop(transport);
    context.shutdown().expect("coordinated PP shutdown");
}
