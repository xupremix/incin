//! Checked collective plans and cross-rank preflight agreement.
//!
//! A placement transition says *which* movement is legal. This module turns
//! that proof into an ordered descriptor containing every value a transport
//! must not infer independently: group, sequence, element and byte counts,
//! dtype, reduction, placements, stream, and dependency.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `collective` builds checked
//! collective descriptors, `preflight` agrees a plan's summary across
//! ranks, `digest` hashes group and plan identity, and `error` is the
//! failure type shared by both. The remaining modules are the two-rank
//! hybrid strategy planner: `topology`, `strategy`, `workload`, and
//! `evidence` are its vocabulary; `candidate` is a scored, inspectable
//! result; `planner` is the algorithm; `hybrid_error` is its failure
//! type.

pub(crate) use alloc::{borrow::ToOwned, string::String, vec, vec::Vec};

pub(crate) use half::{bf16, f16};
pub(crate) use typenum::{B1, IsLessOrEqual, NonZero, U2, U4294967295, Unsigned};

pub(crate) use crate::dist::collective::{
    CollectiveDType, CollectiveError, CollectiveKind, CollectiveReductionDType, GroupId, StreamId,
    validate_collective_dtype, validate_collective_reduction,
};
pub(crate) use crate::dist::mesh::{
    DeviceMesh, LinkClass, MeshAxis, MeshId, ProcessLayout, TopologyFingerprint, ValidMesh,
};
pub(crate) use crate::dist::pipeline::{
    PipelineDType, PipelineSchedule, StaticPipelineSchedule, validate_microbatches,
};
pub(crate) use crate::dist::placement::{
    ConstPlacement, Partial, PartialReduction, Placement, PlacementAxis, PlacementKind,
    PlacementOn, Replicated, Sharded,
};
pub(crate) use crate::dist::rule::{
    DistributedError, LegalTransition, PlacementTransition, ShardDivisible, ShardRemainderPolicy,
};
pub(crate) use crate::exec::ReduceOp;
pub(crate) use crate::shapes::Dyn;
pub(crate) use crate::shapes::error::OperationKind;
pub(crate) use crate::shapes::error::ShapeError;
pub(crate) use crate::tensor::dtype::{BuiltinDType, ConstDType, DTypeId};

mod candidate;
mod collective;
mod digest;
mod error;
mod evidence;
mod hybrid_error;
mod planner;
mod preflight;
mod strategy;
mod topology;
mod workload;

pub use candidate::{HybridPlanReport, RejectedStrategy, StrategyCandidate, StrategyRejection};
pub use collective::{
    CollectiveDescriptor, CollectivePlan, CollectivePlanBuilder, CollectiveTag,
    PlannedCollectiveTransition, SequenceToken,
};
pub(crate) use digest::{
    group_token, kind_for_transition, output_elements, plan_hash, validate_peer,
};
pub use error::PlanError;
pub use evidence::{CommunicationEvidence, PlanningCollectiveKind, ShardEvidence};
pub use hybrid_error::HybridPlanError;
pub use planner::HybridPlanner;
pub use preflight::{AgreedPlan, PlanSummary, preflight};
pub use strategy::{
    MemoryLimit, ParallelOptions, ParallelStrategy, ParallelStrategyKind, PlanObjective,
    StaticParallelOptions, StrategySet,
};
pub use topology::TwoRankPlanningTopology;
pub use workload::{HybridPlanDType, HybridWorkload, WorkloadField, validate_hybrid_plan_dtype};
