//! Inspectable evidence a strategy candidate carries: exact shards and
//! modeled communication.

use super::*;

/// One exact logical shard recorded in a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardEvidence {
    pub(super) field: WorkloadField,
    pub(super) global: usize,
    pub(super) per_rank: [usize; 2],
}

impl ShardEvidence {
    /// Sharded logical quantity.
    #[must_use]
    pub const fn field(self) -> WorkloadField {
        self.field
    }

    /// Global value before partitioning.
    #[must_use]
    pub const fn global(self) -> usize {
        self.global
    }

    /// Exact value assigned to each rank.
    #[must_use]
    pub const fn per_rank(self) -> [usize; 2] {
        self.per_rank
    }
}

/// Communication primitive modeled by a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanningCollectiveKind {
    /// Gradient all-reduce.
    AllReduce,
    /// Activation all-gather.
    AllGather,
    /// Gradient reduce-scatter paired with an all-gather.
    ReduceScatter,
    /// Pipeline point-to-point activation or gradient.
    SendRecv,
}

/// Exact aggregate communication contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommunicationEvidence {
    pub(super) kind: PlanningCollectiveKind,
    pub(super) launches: usize,
    pub(super) bytes: usize,
}

impl CommunicationEvidence {
    /// Modeled primitive.
    #[must_use]
    pub const fn kind(self) -> PlanningCollectiveKind {
        self.kind
    }

    /// Number of logical launches per step.
    #[must_use]
    pub const fn launches(self) -> usize {
        self.launches
    }

    /// Aggregate payload bytes across all launches.
    #[must_use]
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}
