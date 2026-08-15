//! Two-host NCCL connectivity evidence for `DST-008`.
//!
//! These ignored tests validate distributed plan construction and communicator
//! setup. They require two network-accessible CUDA hosts with NCCL. Tensor
//! kernel parity remains covered by the backend-specific capability tests.

#![cfg(all(feature = "distributed-nccl", feature = "hardware-tests"))]

use incin::experimental::distributed::{
    DataParallelPlanBuilder, DistributedContext, GradientId, NcclTopology, NcclTransport, StreamId,
    TwoRankDataParallel,
};
use incin::prelude::*;

#[test]
#[ignore = "requires two network-accessible CUDA hosts with NCCL"]
fn dp2_plan_and_communicator_initialize() {
    let context = DistributedContext::<Dyn, Dyn>::from_env().expect("two-rank rendezvous");
    let rank = context.rank();
    let topology = NcclTopology::discover_context(&context).expect("discover CUDA identities");
    let mesh = incin::experimental::distributed::mesh::DeviceMesh::<TwoRankDataParallel>::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &topology,
    )
    .expect("bind DP=2 network topology");

    let mut builder = DataParallelPlanBuilder::new(&mesh, rank);
    builder
        .push_static::<f32>(GradientId::new(101).unwrap(), 2, StreamId::new(0))
        .expect("static f32 gradient");
    builder
        .push_dyn(
            GradientId::new(202).unwrap(),
            2,
            DTypeId::F32,
            StreamId::new(1),
        )
        .expect("Dyn f32 gradient");
    let plan = builder.finish().expect("non-empty DP plan");
    let transport = NcclTransport::connect_context(&context, plan.into_collective_plan())
        .expect("initialize DP NCCL communicator");
    assert_eq!(transport.cursor(), 0);
    drop(transport);
    context.shutdown().expect("coordinated DP shutdown");
}
