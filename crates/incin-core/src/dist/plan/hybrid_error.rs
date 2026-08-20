//! Hybrid topology, workload, policy, or feasibility failures.

use super::*;

/// Hybrid topology, workload, policy, or feasibility failure.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum HybridPlanError {
    /// Runtime topology is not exactly two ranks.
    #[error("hybrid planner requires exactly {expected} devices, found {found}")]
    TopologyWorld {
        /// Required world.
        expected: usize,
        /// Discovered world.
        found: usize,
    },
    /// Process layout disagrees with the physical device count.
    #[error("hybrid planner requires process world {expected}, found {found}")]
    ProcessWorld {
        /// Required process world.
        expected: usize,
        /// Discovered process world.
        found: usize,
    },
    /// Topology fingerprint omitted an ordered rank link.
    #[error("topology has no link from rank {from_rank} to rank {to_rank}")]
    MissingLink {
        /// Sending rank.
        from_rank: usize,
        /// Receiving rank.
        to_rank: usize,
    },
    /// Topology has an explicitly unreachable ordered link.
    #[error("topology link from rank {from_rank} to rank {to_rank} is unreachable")]
    UnreachableLink {
        /// Sending rank.
        from_rank: usize,
        /// Receiving rank.
        to_rank: usize,
    },
    /// Static dtypes lack an implementation; `Dyn` reaches this variant.
    #[error("hybrid planning requires a floating dtype, found {dtype:?}")]
    UnsupportedDType {
        /// Runtime dtype.
        dtype: DTypeId,
    },
    /// Required logical workload field was zero.
    #[error("hybrid workload field {field:?} must be nonzero")]
    ZeroWorkloadField {
        /// Rejected field.
        field: WorkloadField,
    },
    /// Physical memory capacity was zero.
    #[error("rank {rank} reports zero device memory capacity")]
    ZeroDeviceCapacity {
        /// Rejected rank.
        rank: usize,
    },
    /// Absolute or resolved fractional memory limit was zero.
    #[error("memory limit must resolve to at least one byte per rank")]
    ZeroMemoryLimit,
    /// Fraction was non-finite, non-positive, or greater than one.
    #[error("per-device memory fraction must be finite and in (0, 1]")]
    InvalidMemoryFraction,
    /// Automatic selection was given no candidates.
    #[error("automatic planning requires at least one allowed strategy")]
    EmptyStrategySet,
    /// Padding and ragged sharding are not yet implemented.
    #[error("hybrid planner supports only ShardRemainderPolicy::Reject, found {found:?}")]
    UnsupportedRemainderPolicy {
        /// Requested policy.
        found: ShardRemainderPolicy,
    },
    /// Runtime microbatch count exceeds the tag/schedule representation.
    #[error("microbatch count {found} exceeds supported maximum {maximum}")]
    MicrobatchLimit {
        /// Rejected value.
        found: usize,
        /// Maximum accepted value.
        maximum: usize,
    },
    /// Checked storage sizing failed.
    #[error(transparent)]
    Shape(ShapeError),
    /// Planner-specific checked arithmetic failed.
    #[error("hybrid planning arithmetic overflow in {expression}")]
    ArithmeticOverflow {
        /// Expression that did not fit.
        expression: &'static str,
    },
    /// Every requested strategy failed feasibility.
    #[error("no feasible two-rank strategy")]
    NoFeasibleStrategy {
        /// Complete set of feasibility failures.
        rejected: Vec<RejectedStrategy>,
    },
}
