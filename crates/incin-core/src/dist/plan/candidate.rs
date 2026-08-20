//! A feasible strategy candidate, why an infeasible one was rejected, and
//! the planner's final report.

use super::*;

/// Why a strategy was absent from the feasible candidate set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrategyRejection {
    /// Automatic search set excluded the strategy.
    NotAllowed,
    /// A manual strategy was selected instead.
    NotSelected,
    /// Exact two-way sharding was impossible.
    NonDivisible {
        /// Quantity that did not divide.
        field: WorkloadField,
        /// Global value.
        value: usize,
        /// Required degree.
        degree: usize,
    },
    /// Peak memory crossed a hard rank-local limit.
    MemoryExceeded {
        /// First rank over its limit.
        rank: usize,
        /// Modeled peak.
        required: usize,
        /// Effective limit.
        limit: usize,
    },
}

/// One rejected strategy and the precise feasibility reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedStrategy {
    pub(super) strategy: ParallelStrategyKind,
    pub(super) reason: StrategyRejection,
}

impl RejectedStrategy {
    /// Rejected logical layout.
    #[must_use]
    pub const fn strategy(&self) -> ParallelStrategyKind {
        self.strategy
    }

    /// Rejection evidence.
    #[must_use]
    pub const fn reason(&self) -> &StrategyRejection {
        &self.reason
    }
}

/// Feasible strategy with inspectable analytical evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyCandidate {
    pub(super) strategy: ParallelStrategyKind,
    pub(super) dtype: DTypeId,
    pub(super) shards: Vec<ShardEvidence>,
    pub(super) collectives: Vec<CommunicationEvidence>,
    pub(super) per_rank_peak_memory: [usize; 2],
    pub(super) memory_limits: [usize; 2],
    pub(super) communication_bytes: usize,
    pub(super) estimated_step_cost: u128,
    pub(super) topology_fingerprint: u64,
    pub(super) link: LinkClass,
    pub(super) transport: String,
    pub(super) schedule: Option<PipelineSchedule>,
}

impl StrategyCandidate {
    /// Candidate logical layout.
    #[must_use]
    pub const fn strategy(&self) -> ParallelStrategyKind {
        self.strategy
    }

    /// Static projection or runtime-selected dtype.
    #[must_use]
    pub const fn dtype(&self) -> DTypeId {
        self.dtype
    }

    /// Exact logical partitions.
    #[must_use]
    pub fn shards(&self) -> &[ShardEvidence] {
        &self.shards
    }

    /// Modeled communication primitives.
    #[must_use]
    pub fn collectives(&self) -> &[CommunicationEvidence] {
        &self.collectives
    }

    /// Analytical peak bytes for ranks zero and one.
    #[must_use]
    pub const fn per_rank_peak_memory(&self) -> [usize; 2] {
        self.per_rank_peak_memory
    }

    /// Effective hard memory limit on both ranks.
    #[must_use]
    pub const fn memory_limits(&self) -> [usize; 2] {
        self.memory_limits
    }

    /// Aggregate logical payload bytes for one step.
    #[must_use]
    pub const fn communication_bytes(&self) -> usize {
        self.communication_bytes
    }

    /// Deterministic analytical score, not a measured duration.
    #[must_use]
    pub const fn estimated_step_cost(&self) -> u128 {
        self.estimated_step_cost
    }

    /// Stable physical topology assumed by this estimate.
    #[must_use]
    pub const fn topology_fingerprint(&self) -> u64 {
        self.topology_fingerprint
    }

    /// Least direct rank-to-rank path used by the estimate.
    #[must_use]
    pub const fn link(&self) -> LinkClass {
        self.link
    }

    /// Communication library assumed by the estimate.
    #[must_use]
    pub fn transport(&self) -> &str {
        &self.transport
    }

    /// Pipeline schedule for PP=2, or `None` for DP/TP.
    pub const fn schedule(&self) -> Option<PipelineSchedule> {
        self.schedule
    }
}

/// Inspectable result of hybrid feasibility filtering and objective ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridPlanReport {
    pub(super) objective: PlanObjective,
    pub(super) chosen: StrategyCandidate,
    pub(super) feasible: Vec<StrategyCandidate>,
    pub(super) pareto_frontier: Vec<ParallelStrategyKind>,
    pub(super) rejected: Vec<RejectedStrategy>,
}

impl HybridPlanReport {
    /// Objective used for selection.
    #[must_use]
    pub const fn objective(&self) -> PlanObjective {
        self.objective
    }

    /// Selected feasible strategy.
    #[must_use]
    pub const fn chosen(&self) -> &StrategyCandidate {
        &self.chosen
    }

    /// Every feasible candidate in stable strategy order.
    #[must_use]
    pub fn feasible_candidates(&self) -> &[StrategyCandidate] {
        &self.feasible
    }

    /// Non-dominated strategies over memory, communication, and step score.
    #[must_use]
    pub fn pareto_frontier(&self) -> &[ParallelStrategyKind] {
        &self.pareto_frontier
    }

    /// Excluded or infeasible alternatives.
    #[must_use]
    pub fn rejected(&self) -> &[RejectedStrategy] {
        &self.rejected
    }
}
