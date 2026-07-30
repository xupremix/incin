//! Placement and distributed-execution contracts.
//!
//! EXE-006 introduces only the local placement foundation required by
//! `StorageBackend`. DST-001 adds the typed logical mesh; DST-003 adds logical
//! placement proofs. Collective planning follows in DST-007.

#[cfg(feature = "distributed")]
pub mod mesh;
pub mod placement;
#[cfg(feature = "distributed")]
pub mod rule;

pub use placement::{Local, Placement, PlacementKind};
#[cfg(feature = "distributed")]
pub use placement::{
    Max, Mean, Min, Partial, PartialReduction, PipelineStage, PlacementAxis, PlacementBuf, Prod,
    Replicated, Sharded, Sum,
};
#[cfg(feature = "distributed")]
pub use rule::{
    CompletePlacement, DistributedError, DistributedInputs, DistributedRule, ElementwisePlacement,
    LegalTransition, PlacementTransition, PlacementTransitionRule, ReduceShardedAxis,
    ShardDivisible, ShardRemainderPolicy, ValidatedDistributed, validate_pipeline_stage,
    validate_shard,
};
