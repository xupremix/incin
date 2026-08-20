//! Failures found while constructing or agreeing on collective plans.

use super::*;

/// Failures found while constructing or agreeing on collective plans.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// Shared collective contract rejected the descriptor.
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    /// Dynamic placement transition is not legal.
    #[error(transparent)]
    Distributed(#[from] DistributedError),
    /// Element or byte count overflowed.
    #[error(transparent)]
    Shape(#[from] ShapeError),
    /// Selected rank does not exist in the mesh.
    #[error("rank {rank} is outside a mesh with {world} ranks")]
    RankOutOfRange {
        /// Requested rank.
        rank: usize,
        /// Bound mesh cardinality.
        world: usize,
    },
    /// Identity/local-shard transitions require no transport launch.
    #[error("placement transition {transition:?} does not require a collective")]
    NoCollectiveRequired {
        /// Transition that should remain local.
        transition: PlacementTransition,
    },
    /// Reduction transition did not originate from `Partial`.
    #[error("placement transition {transition:?} requires a partial source, found {placement:?}")]
    MissingReduction {
        /// Transition requiring reduction semantics.
        transition: PlacementTransition,
        /// Actual source placement.
        placement: PlacementKind,
    },
    /// Static transition projection disagrees with the runtime rule table.
    #[error("typed transition {typed:?} disagrees with runtime transition {runtime:?}")]
    TransitionMismatch {
        /// Transition selected by the trait implementation.
        typed: PlacementTransition,
        /// Transition derived from runtime placement projections.
        runtime: PlacementTransition,
    },
    /// Placement movement was assigned to the wrong mesh communicator.
    #[error("collective {kind:?} requires the {expected:?} axis, found {found:?}")]
    WrongAxis {
        /// Planned operation.
        kind: CollectiveKind,
        /// Axis implied by the placement transition.
        expected: MeshAxis,
        /// Axis supplied by the caller.
        found: MeshAxis,
    },
    /// Dependency is not an earlier token in this plan.
    #[error("dependency token {dependency:?} is not earlier than next sequence {next}")]
    UnknownDependency {
        /// Rejected dependency.
        dependency: SequenceToken,
        /// Next zero-based sequence index.
        next: usize,
    },
    /// Descriptor count no longer fits a sequence token.
    #[error("collective sequence exceeds u64")]
    SequenceOverflow,
    /// No ranks participated in preflight.
    #[error("plan preflight requires at least one rank")]
    EmptyPreflight,
    /// Submitted summaries do not cover the expected world.
    #[error("plan preflight expected {expected} ranks, found {found}")]
    PreflightRankCount {
        /// Expected world size.
        expected: usize,
        /// Submitted summary count.
        found: usize,
    },
    /// A rank built its plan for a different physical/logical mesh.
    #[error("rank {rank} has mesh {found:?}, expected {expected:?}")]
    MeshMismatch {
        /// First disagreeing rank.
        rank: usize,
        /// Rank-zero mesh.
        expected: MeshId,
        /// Disagreeing mesh.
        found: MeshId,
    },
    /// A rank plans a different number of launches.
    #[error("rank {rank} plans {found} collectives, expected {expected}")]
    CollectiveCountMismatch {
        /// First disagreeing rank.
        rank: usize,
        /// Rank-zero count.
        expected: usize,
        /// Disagreeing count.
        found: usize,
    },
    /// Descriptor contents or ordering diverge.
    #[error("rank {rank} has plan hash {found:#x}, expected {expected:#x}")]
    PlanHashMismatch {
        /// First disagreeing rank.
        rank: usize,
        /// Rank-zero hash.
        expected: u64,
        /// Disagreeing hash.
        found: u64,
    },
}
