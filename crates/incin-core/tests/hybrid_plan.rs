//! `DST-011`: two-rank hybrid feasibility, memory, and report evidence.

#![cfg(feature = "distributed")]

mod support;

use std::collections::BTreeMap;
use std::path::Path;

use incin_core::dist::mesh::{
    Data, DeviceIdentity, DeviceMesh, LinkClass, MeshSpec, Pipeline, ProcessLayout, TensorParallel,
    TopologyProbe, TransportVersion,
};
use incin_core::dist::{
    HybridPlanError, HybridPlanner, HybridWorkload, MemoryLimit, OneForwardOneBackward,
    ParallelOptions, ParallelStrategy, ParallelStrategyKind, PipelineSchedule, PlanObjective,
    PlanningCollectiveKind, ShardRemainderPolicy, StaticParallelOptions, StrategyRejection,
    StrategySet, TwoRankDataParallel, TwoRankPlanningTopology, WorkloadField,
};
use incin_core::prelude::{DTypeId, DeviceId};
use incin_core::typenum::{U1, U4, U8, U16};

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
                    format!("GPU-HYBRID-{}", device.ordinal()),
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

struct OneCuda;

impl TopologyProbe for OneCuda {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        (device == DeviceId::cuda(0))
            .then(|| DeviceIdentity::new(device, "GPU-ONE".to_string(), "sm_90".to_string()))
    }

    fn link(&self, from: DeviceId, to: DeviceId) -> LinkClass {
        if from == to {
            LinkClass::SameDevice
        } else {
            LinkClass::Unreachable
        }
    }

    fn transport(&self) -> TransportVersion {
        TransportVersion::new("reference".to_string(), 1, 0, 0)
    }

    fn layout(&self) -> ProcessLayout {
        ProcessLayout::SingleProcess
    }
}

fn topology() -> TwoRankPlanningTopology {
    let mesh: DeviceMesh<TwoRankDataParallel> = DeviceMesh::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &TwoNetworkCuda { rank: 0 },
    )
    .unwrap();
    TwoRankPlanningTopology::from_static_mesh(&mesh).unwrap()
}

fn policy(objective: PlanObjective, limit: MemoryLimit) -> StaticParallelOptions {
    StaticParallelOptions {
        memory_limit: limit,
        remainder: ShardRemainderPolicy::Reject,
        objective,
    }
}

fn dyn_options(objective: PlanObjective, limit: MemoryLimit) -> ParallelOptions {
    ParallelOptions {
        strategy: ParallelStrategy::Auto {
            allowed: StrategySet::ALL,
        },
        memory_limit: limit,
        remainder: ShardRemainderPolicy::Reject,
        schedule: PipelineSchedule::OneForwardOneBackward,
        objective,
    }
}

fn workload(activation_elements: usize) -> HybridWorkload {
    HybridWorkload::new(8, 8, 16, activation_elements, 4, 2, [10_000; 2]).unwrap()
}

#[test]
fn static_and_dyn_auto_plans_match_with_exact_report_evidence() {
    let topology = topology();
    let static_report =
        HybridPlanner::plan_auto_static::<f32, U8, U8, U16, U4, U4, OneForwardOneBackward>(
            &topology,
            2,
            [10_000; 2],
            StrategySet::ALL,
            policy(
                PlanObjective::MinimizeStepTime,
                MemoryLimit::PerDeviceFraction(0.85),
            ),
        )
        .unwrap();
    let dynamic_report = HybridPlanner::plan_dyn(
        &topology,
        DTypeId::F32,
        workload(4),
        dyn_options(
            PlanObjective::MinimizeStepTime,
            MemoryLimit::PerDeviceFraction(0.85),
        ),
    )
    .unwrap();

    assert_eq!(static_report, dynamic_report);
    assert_eq!(
        static_report.chosen().strategy(),
        ParallelStrategyKind::Tensor
    );
    assert_eq!(static_report.feasible_candidates().len(), 3);
    assert_eq!(
        static_report.pareto_frontier(),
        &[ParallelStrategyKind::Tensor]
    );
    assert!(static_report.rejected().is_empty());

    let tensor = static_report.chosen();
    assert_eq!(tensor.dtype(), DTypeId::F32);
    assert_eq!(tensor.per_rank_peak_memory(), [144, 144]);
    assert_eq!(tensor.communication_bytes(), 32);
    assert_eq!(tensor.link(), LinkClass::Network);
    assert_eq!(tensor.transport(), "reference");
    assert_eq!(tensor.topology_fingerprint(), topology.fingerprint());
    assert_eq!(tensor.schedule(), None);
    assert_eq!(tensor.shards().len(), 2);
    assert_eq!(tensor.shards()[0].field(), WorkloadField::TensorShardExtent);
    assert_eq!(tensor.shards()[0].global(), 8);
    assert_eq!(tensor.shards()[0].per_rank(), [4, 4]);
    assert_eq!(tensor.collectives().len(), 2);
    assert_eq!(
        tensor.collectives()[0].kind(),
        PlanningCollectiveKind::AllGather
    );
    assert_eq!(
        tensor.collectives()[1].kind(),
        PlanningCollectiveKind::ReduceScatter
    );
    assert_eq!(tensor.collectives()[0].launches(), 1);
    assert_eq!(tensor.collectives()[0].bytes(), 16);
}

#[test]
fn objectives_and_schedule_change_inspectable_choices_and_memory() {
    let topology = topology();
    let communication = HybridPlanner::plan_dyn(
        &topology,
        DTypeId::F32,
        workload(64),
        dyn_options(
            PlanObjective::MinimizeCommunication,
            MemoryLimit::PerRankBytes(10_000),
        ),
    )
    .unwrap();
    assert_eq!(
        communication.chosen().strategy(),
        ParallelStrategyKind::Data
    );

    let pipeline = HybridPlanner::plan_pipeline_static::<f32, U16, U4, U4, OneForwardOneBackward>(
        &topology,
        8,
        8,
        2,
        [10_000; 2],
        policy(
            PlanObjective::MinimizeMemory,
            MemoryLimit::PerRankBytes(10_000),
        ),
    )
    .unwrap();
    assert_eq!(pipeline.chosen().strategy(), ParallelStrategyKind::Pipeline);
    assert_eq!(
        pipeline.chosen().schedule(),
        Some(PipelineSchedule::OneForwardOneBackward)
    );
    assert_eq!(pipeline.chosen().per_rank_peak_memory(), [160, 144]);
    assert_eq!(pipeline.chosen().collectives()[0].launches(), 8);
    assert_eq!(pipeline.rejected().len(), 2);
    assert!(
        pipeline
            .rejected()
            .iter()
            .all(|rejected| rejected.reason() == &StrategyRejection::NotSelected)
    );
}

#[test]
fn dyn_filters_divisibility_and_memory_with_structured_reasons() {
    let topology = topology();
    let odd = HybridWorkload::new(7, 9, 15, 4, 4, 2, [10_000; 2]).unwrap();
    let report = HybridPlanner::plan_dyn(
        &topology,
        DTypeId::F32,
        odd,
        dyn_options(
            PlanObjective::MinimizeMemory,
            MemoryLimit::PerRankBytes(10_000),
        ),
    )
    .unwrap_err();
    let HybridPlanError::NoFeasibleStrategy { rejected } = report else {
        panic!("expected complete feasibility report");
    };
    assert_eq!(rejected.len(), 3);
    assert_eq!(
        rejected[0].reason(),
        &StrategyRejection::NonDivisible {
            field: WorkloadField::BatchSize,
            value: 7,
            degree: 2,
        }
    );
    assert_eq!(
        rejected[1].reason(),
        &StrategyRejection::NonDivisible {
            field: WorkloadField::TensorShardExtent,
            value: 9,
            degree: 2,
        }
    );
    assert_eq!(
        rejected[2].reason(),
        &StrategyRejection::NonDivisible {
            field: WorkloadField::ParameterElements,
            value: 15,
            degree: 2,
        }
    );

    let memory = HybridPlanner::plan_dyn(
        &topology,
        DTypeId::F32,
        workload(4),
        dyn_options(
            PlanObjective::MinimizeMemory,
            MemoryLimit::PerRankBytes(150),
        ),
    )
    .unwrap();
    assert_eq!(memory.feasible_candidates().len(), 1);
    assert_eq!(memory.chosen().strategy(), ParallelStrategyKind::Tensor);
    assert_eq!(memory.rejected().len(), 2);
    assert!(
        memory
            .rejected()
            .iter()
            .all(|rejected| matches!(rejected.reason(), StrategyRejection::MemoryExceeded { .. }))
    );
}

#[test]
fn dyn_rejects_every_static_contract_violation() {
    let topology = topology();
    for dtype in [DTypeId::U8, DTypeId::U32, DTypeId::I64, DTypeId::Q8_0] {
        assert_eq!(
            HybridPlanner::plan_dyn(
                &topology,
                dtype,
                workload(4),
                dyn_options(
                    PlanObjective::MinimizeMemory,
                    MemoryLimit::PerRankBytes(10_000),
                ),
            ),
            Err(HybridPlanError::UnsupportedDType { dtype })
        );
    }

    assert_eq!(
        HybridPlanner::plan_dyn(
            &topology,
            DTypeId::F32,
            workload(4),
            ParallelOptions {
                strategy: ParallelStrategy::Auto {
                    allowed: StrategySet::NONE,
                },
                memory_limit: MemoryLimit::PerRankBytes(10_000),
                remainder: ShardRemainderPolicy::Reject,
                schedule: PipelineSchedule::GPipe,
                objective: PlanObjective::MinimizeMemory,
            },
        ),
        Err(HybridPlanError::EmptyStrategySet)
    );
    assert_eq!(
        HybridPlanner::plan_dyn(
            &topology,
            DTypeId::F32,
            workload(4),
            dyn_options(
                PlanObjective::MinimizeMemory,
                MemoryLimit::PerDeviceFraction(f64::NAN),
            ),
        ),
        Err(HybridPlanError::InvalidMemoryFraction)
    );
    assert_eq!(
        HybridPlanner::plan_dyn(
            &topology,
            DTypeId::F32,
            workload(4),
            ParallelOptions {
                strategy: ParallelStrategy::Data,
                memory_limit: MemoryLimit::PerRankBytes(10_000),
                remainder: ShardRemainderPolicy::PadAndMask,
                schedule: PipelineSchedule::GPipe,
                objective: PlanObjective::MinimizeMemory,
            },
        ),
        Err(HybridPlanError::UnsupportedRemainderPolicy {
            found: ShardRemainderPolicy::PadAndMask,
        })
    );
    assert!(matches!(
        HybridWorkload::new(0, 8, 16, 4, 4, 2, [10_000; 2]),
        Err(HybridPlanError::ZeroWorkloadField {
            field: WorkloadField::BatchSize
        })
    ));

    type OneRank = MeshSpec<Data<U1>, TensorParallel<U1>, Pipeline<U1>>;
    let one: DeviceMesh<OneRank> = DeviceMesh::bind(&[DeviceId::cuda(0)], &OneCuda).unwrap();
    assert_eq!(
        TwoRankPlanningTopology::from_fingerprint(one.fingerprint()),
        Err(HybridPlanError::TopologyWorld {
            expected: 2,
            found: 1,
        })
    );

    #[cfg(target_pointer_width = "64")]
    {
        let too_many =
            HybridWorkload::new(8, 8, 16, 4, u32::MAX as usize + 1, 2, [10_000; 2]).unwrap();
        assert_eq!(
            HybridPlanner::plan_dyn(
                &topology,
                DTypeId::F32,
                too_many,
                dyn_options(
                    PlanObjective::MinimizeMemory,
                    MemoryLimit::PerRankBytes(10_000),
                ),
            ),
            Err(HybridPlanError::MicrobatchLimit {
                found: u32::MAX as usize + 1,
                maximum: u32::MAX as usize,
            })
        );
    }
}

#[test]
fn static_hybrid_contract_rejections_are_compile_errors() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/hybrid_plan_compile_fail/*.rs");
    if std::env::var_os("TRYBUILD").as_deref() != Some(std::ffi::OsStr::new("overwrite")) {
        support::compile_fail_cases_name_their_reason(
            Path::new("tests/hybrid_plan_compile_fail"),
            &BTreeMap::from([
                ("integer_dtype", "HybridPlanDType"),
                ("odd_data_batch", "ShardDivisible"),
                ("odd_tensor_extent", "ShardDivisible"),
                ("wrong_world", "type mismatch"),
                ("zero_pipeline_microbatches", "NonZero"),
            ]),
        );
    }
}
