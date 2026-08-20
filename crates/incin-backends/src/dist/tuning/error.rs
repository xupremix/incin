use super::*;

/// Distributed candidate, measurement, or commit failure.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum CollectiveTuningError {
    /// Shared collective validation failed.
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    /// Checked element-to-byte sizing failed.
    #[error("collective message byte size is not representable")]
    MessageSize,
    /// This coordination slice is exactly two ranks.
    #[error("collective tuning requires group size {expected}, found {found}")]
    GroupSize {
        /// Required group size.
        expected: usize,
        /// Supplied group size.
        found: usize,
    },
    /// Physical topology and group cardinality differ.
    #[error("topology has {found} devices, expected {expected}")]
    TopologyWorld {
        /// Required topology world.
        expected: usize,
        /// Discovered devices.
        found: usize,
    },
    /// Zero-element experiments are not representative collectives.
    #[error("collective tuning requires a nonzero element count")]
    ZeroElements,
    /// Runtime element count exceeds the static problem representation.
    #[error("collective tuning supports at most {maximum} elements, found {found}")]
    ElementLimit {
        /// Maximum supported elements.
        maximum: usize,
        /// Rejected elements.
        found: usize,
    },
    /// Dynamic all-to-all/reduce-scatter cardinality is not exact.
    #[error("{elements} elements do not divide across {ranks} ranks")]
    NonDivisible {
        /// Global or input elements.
        elements: usize,
        /// Collective ranks.
        ranks: usize,
    },
    /// Point-to-point endpoint is not in the group.
    #[error("{endpoint} rank {rank} is outside group size {ranks}")]
    PeerOutOfRange {
        /// Endpoint role.
        endpoint: &'static str,
        /// Rejected rank.
        rank: usize,
        /// Group cardinality.
        ranks: usize,
    },
    /// Send and receive endpoint are identical.
    #[error("send/receive endpoints must differ, found rank {rank} twice")]
    SamePeer {
        /// Repeated rank.
        rank: usize,
    },
    /// A future core variant has no tuning semantics yet.
    #[error("collective kind is not supported by this tuning schema")]
    UnsupportedCollectiveKind,
    /// Runtime channel count is outside the static range.
    #[error("channel count {found} is outside [{minimum}, {maximum}]")]
    ChannelCount {
        /// Minimum channels.
        minimum: usize,
        /// Maximum channels.
        maximum: usize,
        /// Rejected channels.
        found: usize,
    },
    /// Runtime chunk is zero or not a power of two.
    #[error("chunk size {found} must be a nonzero power of two")]
    ChunkBytes {
        /// Rejected bytes.
        found: usize,
    },
    /// Runtime stream priority is outside the bounded range.
    #[error("stream priority {found} is outside [{minimum}, {maximum}]")]
    StreamPriority {
        /// Minimum priority.
        minimum: i8,
        /// Maximum priority.
        maximum: i8,
        /// Rejected priority.
        found: i8,
    },
    /// Search contains no candidates.
    #[error("collective tuning requires at least one candidate")]
    EmptyCandidates,
    /// Runtime candidate budget is zero.
    #[error("collective tuning candidate budget must be nonzero")]
    ZeroCandidateBudget,
    /// Runtime sample budget is zero.
    #[error("collective tuning sample budget must be nonzero")]
    ZeroSampleBudget,
    /// Runtime budget exceeds the static search bound.
    #[error(
        "collective tuning budget supports at most {maximum} candidates and samples; found {candidates} candidates and {samples} samples"
    )]
    BudgetLimit {
        /// Maximum for either axis.
        maximum: usize,
        /// Requested candidates.
        candidates: usize,
        /// Requested samples per candidate.
        samples: usize,
    },
    /// Candidate set crosses its hard bound.
    #[error("candidate budget permits {maximum}, found {found}")]
    CandidateBudgetExceeded {
        /// Maximum candidate count.
        maximum: usize,
        /// Supplied candidates.
        found: usize,
    },
    /// Candidate was listed twice.
    #[error("candidate hash {hash:#x} appears more than once")]
    DuplicateCandidate {
        /// Repeated identity.
        hash: u64,
    },
    /// Measurement rounds do not cover the candidate set exactly.
    #[error("expected {expected} candidate rounds, found {found}")]
    RoundCount {
        /// Candidate count.
        expected: usize,
        /// Round count.
        found: usize,
    },
    /// Round order or identity differs from the broadcast candidate order.
    #[error("round {index} measured candidate {found:#x}, expected {expected:#x}")]
    RoundCandidate {
        /// Round index.
        index: usize,
        /// Expected candidate.
        expected: u64,
        /// Measured candidate.
        found: u64,
    },
    /// Candidate transient storage crosses the problem budget.
    #[error("candidate requires {required} workspace bytes, budget is {budget}")]
    WorkspaceExceeded {
        /// Required transient bytes.
        required: usize,
        /// Hard budget.
        budget: usize,
    },
    /// Required determinism filtered a candidate.
    #[error("candidate {hash:#x} does not satisfy required determinism")]
    NondeterministicCandidate {
        /// Rejected candidate.
        hash: u64,
    },
    /// Chunk is larger than the whole message.
    #[error("candidate chunk {chunk} exceeds message size {message}")]
    ChunkExceedsMessage {
        /// Chunk bytes.
        chunk: usize,
        /// Message bytes.
        message: usize,
    },
    /// Round lacks one report per rank.
    #[error("candidate round requires {expected} rank reports, found {found}")]
    RankReportCount {
        /// Required reports.
        expected: usize,
        /// Supplied reports.
        found: usize,
    },
    /// Rank is outside the exact group.
    #[error("rank {rank} is outside group size {ranks}")]
    RankOutOfRange {
        /// Rejected rank.
        rank: usize,
        /// Group size.
        ranks: usize,
    },
    /// Rank reported twice.
    #[error("rank {rank} appears more than once")]
    DuplicateRank {
        /// Repeated rank.
        rank: usize,
    },
    /// A rank was absent after report validation.
    #[error("rank {rank} did not submit a measurement report")]
    MissingRank {
        /// Missing rank.
        rank: usize,
    },
    /// Rank measured a different problem.
    #[error("rank {rank} measured problem {found:#x}, expected {expected:#x}")]
    ProblemHash {
        /// Disagreeing rank.
        rank: usize,
        /// Expected identity.
        expected: u64,
        /// Submitted identity.
        found: u64,
    },
    /// Rank measured a different candidate.
    #[error("rank {rank} measured candidate {found:#x}, expected {expected:#x}")]
    CandidateHash {
        /// Disagreeing rank.
        rank: usize,
        /// Expected identity.
        expected: u64,
        /// Submitted identity.
        found: u64,
    },
    /// Rank launch, synchronization, validation, or timeout failed.
    #[error("rank {rank} failed candidate measurement")]
    RankMeasurementFailed {
        /// Failing rank.
        rank: usize,
    },
    /// Dedicated measurement payload changed.
    #[error("rank {rank} measurement mutated buffer digest {before:#x} to {after:#x}")]
    MeasurementMutatedBuffer {
        /// Failing rank.
        rank: usize,
        /// Initial digest.
        before: u64,
        /// Final digest.
        after: u64,
    },
    /// Rank supplied the wrong number of synchronized samples.
    #[error("rank {rank} supplied {found} samples, expected {expected}")]
    SampleCount {
        /// Failing rank.
        rank: usize,
        /// Required samples.
        expected: usize,
        /// Supplied samples.
        found: usize,
    },
    /// Commit does not contain exactly one vote per rank.
    #[error("commit requires {expected} votes, found {found}")]
    CommitVoteCount {
        /// Required votes.
        expected: usize,
        /// Supplied votes.
        found: usize,
    },
    /// Rank voted against committing.
    #[error("rank {rank} rejected the tuning commit")]
    CommitRejected {
        /// Rejecting rank.
        rank: usize,
    },
    /// Rank voted for a different problem, candidate, or score.
    #[error("rank {rank} voted for a different tuning result")]
    CommitMismatch {
        /// Disagreeing rank.
        rank: usize,
    },
}
