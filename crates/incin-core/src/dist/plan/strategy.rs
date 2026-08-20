//! Strategy selection vocabulary: which layouts the hybrid planner may
//! consider, and the runtime policy inputs that bound it.

use super::*;

/// Logical two-rank strategy considered by the hybrid planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParallelStrategyKind {
    /// Two replicas, each processing half of the batch.
    Data,
    /// One model whose selected dimensions are split across two ranks.
    Tensor,
    /// Two sequential model stages.
    Pipeline,
}

/// Set of strategies the automatic planner may consider.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrategySet(u8);

impl StrategySet {
    /// No strategy.
    pub const NONE: Self = Self(0);
    /// Data parallelism.
    pub const DATA: Self = Self(1 << 0);
    /// Tensor parallelism.
    pub const TENSOR: Self = Self(1 << 1);
    /// Pipeline parallelism.
    pub const PIPELINE: Self = Self(1 << 2);
    /// Every strategy implemented by the two-rank planner.
    pub const ALL: Self = Self(Self::DATA.0 | Self::TENSOR.0 | Self::PIPELINE.0);

    /// Whether `strategy` belongs to this set.
    #[must_use]
    pub const fn contains(self, strategy: ParallelStrategyKind) -> bool {
        let bit = match strategy {
            ParallelStrategyKind::Data => Self::DATA.0,
            ParallelStrategyKind::Tensor => Self::TENSOR.0,
            ParallelStrategyKind::Pipeline => Self::PIPELINE.0,
        };
        self.0 & bit != 0
    }

    /// Union two sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether the set contains no strategies.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for StrategySet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl core::ops::BitOrAssign for StrategySet {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

/// Manual or automatic strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParallelStrategy {
    /// Require data parallelism.
    Data,
    /// Require tensor parallelism.
    Tensor,
    /// Require pipeline parallelism.
    Pipeline,
    /// Compare every allowed feasible strategy.
    Auto {
        /// Candidate set considered before feasibility filtering.
        allowed: StrategySet,
    },
}

/// Per-rank memory ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryLimit {
    /// Same absolute byte ceiling for both ranks.
    PerRankBytes(usize),
    /// Fraction of each device's discovered capacity available to the plan.
    ///
    /// The accepted range is `(0, 1]`. Capacity remains a runtime physical
    /// fact even when all tensor dimensions are static.
    PerDeviceFraction(f64),
}

/// Objective used to order feasible candidates.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanObjective {
    /// Analytical compute, communication, and pipeline-bubble estimate.
    #[default]
    MinimizeStepTime,
    /// Lowest maximum per-rank peak memory.
    MinimizeMemory,
    /// Lowest aggregate communication volume.
    MinimizeCommunication,
}

/// Runtime strategy and policy inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParallelOptions {
    /// Manual selection or allowed automatic search space.
    pub strategy: ParallelStrategy,
    /// Hard memory ceiling.
    pub memory_limit: MemoryLimit,
    /// Sharding behavior; the initial planner supports exact rejection only.
    pub remainder: ShardRemainderPolicy,
    /// Pipeline schedule used by the PP=2 candidate.
    pub schedule: PipelineSchedule,
    /// Candidate ordering objective.
    pub objective: PlanObjective,
}

/// Policy shared by the compile-time strategy entry points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticParallelOptions {
    /// Hard memory ceiling.
    pub memory_limit: MemoryLimit,
    /// Exact sharding policy.
    pub remainder: ShardRemainderPolicy,
    /// Candidate ordering objective, recorded in the report.
    pub objective: PlanObjective,
}
