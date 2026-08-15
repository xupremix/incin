//! Two-host NCCL connectivity evidence for `DST-009`.
//!
//! These ignored tests validate tensor-parallel plan construction and
//! communicator setup. They require two network-accessible CUDA hosts with
//! NCCL. Tensor kernel parity remains covered by backend-specific tests.

#![cfg(all(feature = "distributed-nccl", feature = "hardware-tests"))]

use incin::experimental::distributed::{
    DistributedContext, NcclTopology, NcclTransport, StreamId, TensorParallelPlanBuilder,
    TwoRankTensorParallel,
};
use incin::prelude::*;

#[test]
#[ignore = "requires two network-accessible CUDA hosts with NCCL"]
fn tp2_plan_and_communicator_initialize() {
    let context = DistributedContext::<Dyn, Dyn>::from_env().expect("two-rank rendezvous");
    let rank = context.rank();
    let topology = NcclTopology::discover_context(&context).expect("discover CUDA identities");
    let mesh = incin::experimental::distributed::mesh::DeviceMesh::<TwoRankTensorParallel>::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &topology,
    )
    .expect("bind TP=2 network topology");

    let mut builder = TensorParallelPlanBuilder::new(&mesh, rank);
    builder
        .push_column_static::<f32, incin::typenum::U0, incin::typenum::U2>(
            incin::experimental::distributed::TensorParallelId::new(201).unwrap(),
            2,
            StreamId::new(0),
        )
        .expect("static TP collective");
    let plan = builder.finish().expect("non-empty TP plan");
    let transport = NcclTransport::connect_context(&context, plan.into_collective_plan())
        .expect("initialize TP NCCL communicator");
    assert_eq!(transport.cursor(), 0);
    drop(transport);
    context.shutdown().expect("coordinated TP shutdown");
}
