//! FSDP and ZeRO memory parity integration test (DST-014).

#![cfg(feature = "distributed")]

use incin_core::dist::fsdp::{
    FsdpError, FsdpParameterId, FsdpPlanBuilder, ZeROStage,
};
use incin_core::prelude::DTypeId;

#[test]
fn zero3_fsdp_persistent_and_transient_memory_parity() {
    let mut builder = FsdpPlanBuilder::new(ZeROStage::ZeRO3);

    // Register 4 layers with 1,000,000 parameters each (4M params total = 16MB in f32)
    for layer in 0..4 {
        let weight_id = FsdpParameterId::new((layer * 2 + 1) as u64).unwrap();
        let bias_id = FsdpParameterId::new((layer * 2 + 2) as u64).unwrap();

        builder
            .push_dyn(weight_id, 990_000, DTypeId::F32, layer, 2)
            .unwrap();
        builder
            .push_dyn(bias_id, 10_000, DTypeId::F32, layer, 2)
            .unwrap();
    }

    let plan = builder.finish(2).expect("valid 2-rank ZeRO-3 plan");
    assert_eq!(plan.stage(), ZeROStage::ZeRO3);
    assert_eq!(plan.world_size(), 2);
    assert_eq!(plan.parameters().len(), 8);

    let report = plan.memory_report();

    // Full unsharded DP model: 4M params * 4 bytes/param * 4 (weight+grad+opt) = 64,000,000 bytes
    assert_eq!(report.unsharded_full_bytes, 64_000_000);

    // Persistent ZeRO-3 memory per rank: 2M sharded params * 4 bytes * 4 = 32,000,000 bytes (half of DP)
    assert_eq!(report.persistent_bytes, 32_000_000);

    // Peak transient memory per rank: largest single layer unsharded params (1M params = 4,000,000 bytes)
    assert_eq!(report.transient_bytes, 4_000_000);

    // Memory reduction ratio should be exactly 2.0x
    assert!((report.memory_reduction_ratio - 2.0).abs() < 1e-5);

    // Parity verification clean
    plan.verify_memory_parity().expect("memory parity verified");
}

#[test]
fn zero1_zero2_zero3_stage_comparison() {
    let create_plan = |stage: ZeROStage| {
        let mut builder = FsdpPlanBuilder::new(stage);
        builder
            .push_dyn(FsdpParameterId::new(1).unwrap(), 100_000, DTypeId::F32, 0, 4)
            .unwrap();
        builder.finish(4).unwrap()
    };

    let p1 = create_plan(ZeROStage::ZeRO1);
    let p2 = create_plan(ZeROStage::ZeRO2);
    let p3 = create_plan(ZeROStage::ZeRO3);

    let r1 = p1.memory_report();
    let r2 = p2.memory_report();
    let r3 = p3.memory_report();

    // ZeRO-3 should yield the strictly smallest persistent memory per rank on world_size=4
    assert!(r3.persistent_bytes < r2.persistent_bytes);
    assert!(r2.persistent_bytes < r1.persistent_bytes);

    // ZeRO-1 and ZeRO-2 do not allocate transient weights
    assert_eq!(r1.transient_bytes, 0);
    assert_eq!(r2.transient_bytes, 0);
    assert!(r3.transient_bytes > 0);
}

#[test]
fn fsdp_builder_error_handling() {
    // Reserved ID
    assert_eq!(
        FsdpParameterId::new(0),
        Err(FsdpError::ReservedParameterId)
    );

    // Duplicate parameter ID
    let mut builder = FsdpPlanBuilder::new(ZeROStage::ZeRO3);
    let pid = FsdpParameterId::new(42).unwrap();
    builder.push_dyn(pid, 1000, DTypeId::F32, 0, 2).unwrap();
    assert!(matches!(
        builder.push_dyn(pid, 1000, DTypeId::F32, 0, 2),
        Err(FsdpError::DuplicateParameter { .. })
    ));

    // Empty parameters finish
    let empty_builder = FsdpPlanBuilder::new(ZeROStage::ZeRO3);
    assert_eq!(empty_builder.finish(2), Err(FsdpError::NoParameters));

    // Invalid world size
    let mut invalid_ws_builder = FsdpPlanBuilder::new(ZeROStage::ZeRO3);
    assert_eq!(
        invalid_ws_builder.push_dyn(pid, 1000, DTypeId::F32, 0, 0),
        Err(FsdpError::InvalidWorldSize)
    );
}
