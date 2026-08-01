//! Placement and distributed-execution contracts.
//!
//! EXE-006 introduces only the local placement foundation required by
//! `StorageBackend`. DST-001 adds the typed logical mesh; DST-003 adds logical
//! placement proofs. DST-004 attaches them to tensors, and DST-007 adds
//! transport-neutral collective descriptors and preflight agreement. DST-008
//! adds the two-rank data-parallel gradient contract; DST-009 adds the matching
//! tensor-parallel linear and attention contract.
//! DST-010 adds two-stage pipeline schedules and point-to-point transfers;
//! DST-015 adds typed process rendezvous and the fail-stop context lifecycle.

#[cfg(feature = "distributed")]
pub mod collective;
#[cfg(feature = "distributed")]
pub mod context;
#[cfg(feature = "distributed")]
pub mod data_parallel;
#[cfg(feature = "distributed")]
pub mod fsdp;
#[cfg(feature = "distributed")]
pub mod mesh;
#[cfg(feature = "distributed")]
pub mod pipeline;
pub mod placement;
#[cfg(feature = "distributed")]
pub mod plan;
#[cfg(feature = "distributed")]
pub mod rule;
#[cfg(feature = "distributed")]
pub mod tensor_parallel;

#[cfg(feature = "distributed")]
pub use collective::{
    CollectiveDType, CollectiveError, CollectiveKind, CollectiveReductionDType, GroupId, StreamId,
    validate_collective_dtype, validate_collective_reduction,
};
#[cfg(feature = "distributed")]
pub use context::{
    ContextError, ContextFailure, DistributedContext, DistributedContextHandle,
    DistributedContextState, DistributedIdentity, LOCAL_CUDA_DEVICE_ENV, RANK_ENV,
    RENDEZVOUS_ADDR_ENV, RENDEZVOUS_TIMEOUT_MS_ENV, RUN_ID_ENV, RunId, StaticTwoRank,
    TWO_RANK_WORLD, WORLD_SIZE_ENV,
};
#[cfg(all(feature = "distributed", feature = "std"))]
pub use context::{
    DynRendezvousConfig, RankLaunch, RendezvousEndpoint, StaticRendezvousConfig, TwoRankLaunchPlan,
};
#[cfg(feature = "distributed")]
pub use data_parallel::{
    DataParallelDType, DataParallelError, DataParallelPlan, DataParallelPlanBuilder,
    GradientDescriptor, GradientId, TwoRankDataParallel, validate_data_parallel_dtype,
};
#[cfg(feature = "distributed")]
pub use pipeline::{
    ActivationCheckpoint, GPipe, OneForwardOneBackward, PipelineAction, PipelineBoundaryId,
    PipelineClock, PipelineDType, PipelineError, PipelinePhase, PipelinePlan, PipelinePlanBuilder,
    PipelineSchedule, PipelineScheduleDescriptor, PipelineTransfer, PipelineTransferDescriptor,
    StaticPipelineSchedule, TwoRankPipeline, validate_microbatches, validate_pipeline_dtype,
};
pub use placement::{ConstPlacement, Local, Placement, PlacementKind};
#[cfg(feature = "distributed")]
pub use placement::{
    Max, Mean, Min, Partial, PartialReduction, PipelineStage, PlacementAxis, PlacementBuf,
    PlacementOn, Prod, Replicated, Sharded, Sum,
};
#[cfg(feature = "distributed")]
pub use plan::{
    AgreedPlan, CollectiveDescriptor, CollectivePlan, CollectivePlanBuilder, CollectiveTag,
    CommunicationEvidence, HybridPlanDType, HybridPlanError, HybridPlanReport, HybridPlanner,
    HybridWorkload, MemoryLimit, ParallelOptions, ParallelStrategy, ParallelStrategyKind,
    PlanError, PlanObjective, PlanSummary, PlannedCollectiveTransition, PlanningCollectiveKind,
    RejectedStrategy, SequenceToken, ShardEvidence, StaticParallelOptions, StrategyCandidate,
    StrategyRejection, StrategySet, TwoRankPlanningTopology, WorkloadField, preflight,
    validate_hybrid_plan_dtype,
};
#[cfg(feature = "distributed")]
pub use rule::{
    CompletePlacement, DistributedError, DistributedInputs, DistributedRule, ElementwisePlacement,
    LegalTransition, PlacementTransition, PlacementTransitionRule, ReduceShardedAxis,
    ShardDivisible, ShardRemainderPolicy, ValidatedDistributed, validate_pipeline_stage,
    validate_shard, validate_transition,
};
#[cfg(feature = "distributed")]
pub use tensor_parallel::{
    TensorParallelCollective, TensorParallelDType, TensorParallelDescriptor,
    TensorParallelDimension, TensorParallelError, TensorParallelId, TensorParallelPlan,
    TensorParallelPlanBuilder, TwoRankTensorParallel, TwoWayShard, validate_tensor_parallel_dtype,
    validate_two_way_extent,
};
#[cfg(feature = "distributed")]
pub use fsdp::{
    FsdpError, FsdpMemoryReport, FsdpParameterDescriptor, FsdpParameterId, FsdpPlan,
    FsdpPlanBuilder, ZeROStage,
};
