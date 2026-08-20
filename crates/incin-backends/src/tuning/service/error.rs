use super::context::TuningScope;
use crate::tuning::cache::CacheError;

/// General tuning-service failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TuningServiceError {
    /// A zero duration cannot bound a lease.
    #[error("tuning time budget must be nonzero")]
    ZeroTimeBudget,
    /// Runtime distributed scope omitted its topology.
    #[error("runtime tuning scope {scope:?} requires a topology")]
    TopologyRequired {
        /// Runtime scope.
        scope: TuningScope,
    },
    /// Runtime kernel scope supplied an irrelevant topology.
    #[error("runtime tuning scope {scope:?} must not carry a topology")]
    UnexpectedTopology {
        /// Runtime scope.
        scope: TuningScope,
    },
    /// Runtime policy carried a zero warmup budget.
    #[error("coordinated warmup budget must be nonzero")]
    ZeroWarmupBudget,
    /// Candidate encoding is not bounded canonical text.
    #[error("tuning candidate encoding must be nonempty bounded canonical text")]
    InvalidCandidateEncoding,
    /// No candidates were supplied.
    #[error("tuning candidate set must not be empty")]
    EmptyCandidates,
    /// Two candidates claim the same stable hash.
    #[error("duplicate tuning candidate hash {hash:#x}")]
    DuplicateCandidate {
        /// Repeated hash.
        hash: u64,
    },
    /// Policy filters removed every candidate.
    #[error("no candidate satisfies determinism and memory policy")]
    NoLegalCandidates,
    /// The declared fallback was filtered out or absent.
    #[error("fallback candidate {hash:#x} is not legal")]
    IllegalFallback {
        /// Candidate hash.
        hash: u64,
    },
    /// The declared heuristic winner was filtered out or absent.
    #[error("heuristic candidate {hash:#x} is not legal")]
    IllegalHeuristic {
        /// Candidate hash.
        hash: u64,
    },
    /// The key does not belong to the supplied context.
    #[error("tuning cache key does not match context {field}")]
    KeyContextMismatch {
        /// Mismatching key field.
        field: &'static str,
    },
    /// Another caller held the lease longer than this context permits.
    #[error("timed out waiting for tuning lease")]
    WaitTimeout,
    /// Measurement finished after its hard deadline.
    #[error("tuning permit epoch {epoch} expired")]
    PermitExpired {
        /// Lease epoch.
        epoch: u64,
    },
    /// The lease was cancelled or superseded.
    #[error("tuning permit epoch {epoch} is no longer active")]
    PermitNotActive {
        /// Lease epoch.
        epoch: u64,
    },
    /// Commit named a candidate outside the filtered set.
    #[error("candidate {hash:#x} was not in the permit's legal set")]
    CandidateNotLegal {
        /// Candidate hash.
        hash: u64,
    },
    /// A local commit was attempted for a multi-participant scope.
    #[error("local commit cannot satisfy {participants} tuning participants")]
    LocalCommitForDistributed {
        /// Expected participants.
        participants: usize,
    },
    /// Coordinated commit supplied the wrong number of votes.
    #[error("coordinated commit expected {expected} votes, found {found}")]
    VoteCount {
        /// Expected vote count.
        expected: usize,
        /// Supplied vote count.
        found: usize,
    },
    /// A vote named a rank outside the participant set.
    #[error("coordinated tuning vote rank {rank} is outside {participants} participants")]
    VoteRank {
        /// Invalid rank.
        rank: usize,
        /// Participant count.
        participants: usize,
    },
    /// A participant voted more than once.
    #[error("duplicate coordinated tuning vote from rank {rank}")]
    DuplicateVote {
        /// Repeated rank.
        rank: usize,
    },
    /// A vote was negative or named another epoch/candidate.
    #[error("coordinated tuning vote from rank {rank} does not accept this result")]
    VoteMismatch {
        /// Rejecting or mismatching rank.
        rank: usize,
    },
    /// A measured result had no samples.
    #[error("measured tuning result requires a nonzero sample count")]
    ZeroSamples,
    /// Persistent cache failure.
    #[error(transparent)]
    Cache(#[from] CacheError),
}
