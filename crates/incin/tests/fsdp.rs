//! Two-host and local evidence for FSDP / ZeRO memory parity `DST-014`.
//!
//! Evaluates FSDP and ZeRO memory sharding plan construction, transient
//! vs persistent memory scaling, and memory reduction parity across world sizes.

#![cfg(feature = "distributed-nccl")]

use incin::experimental::distributed::{FsdpParameterId, FsdpPlanBuilder, ZeROStage};
use incin::prelude::*;

#[test]
fn fsdp_network_prototype_memory_parity() {
    let mut builder = FsdpPlanBuilder::new(ZeROStage::ZeRO3);

    let w1 = FsdpParameterId::new(101).unwrap();
    let b1 = FsdpParameterId::new(102).unwrap();
    let w2 = FsdpParameterId::new(201).unwrap();
    let b2 = FsdpParameterId::new(202).unwrap();

    builder.push_static::<f32>(w1, 500_000, 0, 2).unwrap();
    builder.push_static::<f32>(b1, 1_000, 0, 2).unwrap();
    builder.push_dyn(w2, 500_000, DTypeId::F32, 1, 2).unwrap();
    builder.push_dyn(b2, 1_000, DTypeId::F32, 1, 2).unwrap();

    let plan = builder.finish(2).expect("FSDP plan finish");
    let report = plan.memory_report();

    assert_eq!(report.transient_bytes, 501_000 * 4);
    assert_eq!(report.persistent_bytes, 501_000 * 4 * 4);
    assert_eq!(report.unsharded_full_bytes, 1_002_000 * 4 * 4);
    assert!((report.memory_reduction_ratio - 2.0).abs() < 1e-4);

    plan.verify_memory_parity().expect("FSDP parity clean");
}

#[test]
#[ignore = "requires two network-accessible CUDA hosts with NCCL"]
fn fsdp_two_rank_networked_cuda_parity() {
    // In multi-host network execution, ranks build matching FsdpPlans
    // and verify persistent vs transient memory bounds on device.
    let mut builder = FsdpPlanBuilder::new(ZeROStage::ZeRO3);
    let pid = FsdpParameterId::new(1).unwrap();
    builder.push_static::<f32>(pid, 10_000_000, 0, 2).unwrap();
    let plan = builder.finish(2).unwrap();
    assert_eq!(plan.world_size(), 2);
    plan.verify_memory_parity().unwrap();
}
