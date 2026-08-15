//! Coordinated two-rank collective-tuning contracts.
//!
//! This module does not claim to control NCCL algorithms yet. It owns the
//! transport-neutral coordination rules that must be true before a measured
//! winner may enter a cache: every candidate is legal for the same problem,
//! every rank measures that candidate, scoring uses the median of per-sample
//! maximum-rank duration, measurement buffers are unchanged, and every rank
//! votes for the exact same result.

use alloc::{string::String, vec::Vec};
use core::marker::PhantomData;

use incin_core::dist::mesh::TopologyFingerprint;
use incin_core::dist::placement::PartialReduction;
use incin_core::dist::{
    CollectiveDType, CollectiveError, CollectiveKind, CollectiveReductionDType, GroupId,
    ShardDivisible,
};
use incin_core::exec::{Determinism, ReduceOp};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::{ConstDType, DTypeId};
use incin_core::typenum::{B1, IsLessOrEqual, NonZero, PowerOfTwo, U2, U32, U4294967295, Unsigned};

/// Collective algorithm family offered to a transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CollectiveAlgorithm {
    /// Bandwidth-oriented ring.
    Ring,
    /// Latency-oriented tree.
    Tree,
}

/// Collective wire protocol family offered to a transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CollectiveProtocol {
    /// General protocol.
    Simple,
    /// Low-latency protocol.
    LowLatency,
    /// Wider low-latency protocol.
    LowLatency128,
}

/// Type-level collective algorithm for static candidate construction.
pub trait StaticCollectiveAlgorithm: 'static {
    /// Runtime projection recorded in the candidate.
    const ALGORITHM: CollectiveAlgorithm;
}

/// Static ring marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ring;

impl StaticCollectiveAlgorithm for Ring {
    const ALGORITHM: CollectiveAlgorithm = CollectiveAlgorithm::Ring;
}

/// Static tree marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tree;

impl StaticCollectiveAlgorithm for Tree {
    const ALGORITHM: CollectiveAlgorithm = CollectiveAlgorithm::Tree;
}

/// Type-level collective protocol for static candidate construction.
pub trait StaticCollectiveProtocol: 'static {
    /// Runtime projection recorded in the candidate.
    const PROTOCOL: CollectiveProtocol;
}

/// Static general-protocol marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Simple;

impl StaticCollectiveProtocol for Simple {
    const PROTOCOL: CollectiveProtocol = CollectiveProtocol::Simple;
}

/// Static low-latency protocol marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LowLatency;

impl StaticCollectiveProtocol for LowLatency {
    const PROTOCOL: CollectiveProtocol = CollectiveProtocol::LowLatency;
}

/// Static 128-bit low-latency protocol marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LowLatency128;

impl StaticCollectiveProtocol for LowLatency128 {
    const PROTOCOL: CollectiveProtocol = CollectiveProtocol::LowLatency128;
}

/// Compile-time collective operation selected for one tuning problem.
pub trait StaticCollectiveTuning<K: CollectiveDType, Elements: Unsigned>: 'static {
    /// Runtime collective descriptor.
    const KIND: CollectiveKind;
}

/// Static all-gather marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneAllGather;

impl<K, Elements> StaticCollectiveTuning<K, Elements> for TuneAllGather
where
    K: CollectiveDType,
    Elements: Unsigned,
{
    const KIND: CollectiveKind = CollectiveKind::AllGather;
}

/// Static all-to-all marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneAllToAll;

impl<K, Elements> StaticCollectiveTuning<K, Elements> for TuneAllToAll
where
    K: CollectiveDType,
    Elements: Unsigned + ShardDivisible<U2>,
{
    const KIND: CollectiveKind = CollectiveKind::AllToAll;
}

/// Static all-reduce marker parameterized by a reduction proof marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneAllReduce<R>(PhantomData<R>);

impl<K, Elements, R> StaticCollectiveTuning<K, Elements> for TuneAllReduce<R>
where
    K: CollectiveReductionDType<R>,
    Elements: Unsigned,
    R: PartialReduction,
{
    const KIND: CollectiveKind = CollectiveKind::AllReduce(R::OP);
}

/// Static reduce-scatter marker parameterized by a reduction proof marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneReduceScatter<R>(PhantomData<R>);

impl<K, Elements, R> StaticCollectiveTuning<K, Elements> for TuneReduceScatter<R>
where
    K: CollectiveReductionDType<R>,
    Elements: Unsigned + ShardDivisible<U2>,
    R: PartialReduction,
{
    const KIND: CollectiveKind = CollectiveKind::ReduceScatter(R::OP);
}

/// Static point-to-point marker from rank zero to rank one.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneSendZeroToOne;

impl<K, Elements> StaticCollectiveTuning<K, Elements> for TuneSendZeroToOne
where
    K: CollectiveDType,
    Elements: Unsigned,
{
    const KIND: CollectiveKind = CollectiveKind::SendRecv {
        source: 0,
        destination: 1,
    };
}

/// Static point-to-point marker from rank one to rank zero.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneSendOneToZero;

impl<K, Elements> StaticCollectiveTuning<K, Elements> for TuneSendOneToZero
where
    K: CollectiveDType,
    Elements: Unsigned,
{
    const KIND: CollectiveKind = CollectiveKind::SendRecv {
        source: 1,
        destination: 0,
    };
}

/// Stable cache identity for a collective tuning problem.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CollectiveTuningKey {
    kind: CollectiveKind,
    dtype: DTypeId,
    message_size_bucket: u8,
    group: GroupId,
    topology: u64,
    transport: String,
    transport_version: (u32, u32, u32),
    determinism: Determinism,
    hash: u64,
}

impl CollectiveTuningKey {
    /// Collective semantics.
    #[must_use]
    pub const fn kind(&self) -> CollectiveKind {
        self.kind
    }

    /// Static projection or runtime dtype.
    #[must_use]
    pub const fn dtype(&self) -> DTypeId {
        self.dtype
    }

    /// Ceiling-log2 byte bucket.
    #[must_use]
    pub const fn message_size_bucket(&self) -> u8 {
        self.message_size_bucket
    }

    /// Exact ordered group.
    #[must_use]
    pub const fn group(&self) -> GroupId {
        self.group
    }

    /// Stable topology digest including rank-to-device mapping.
    #[must_use]
    pub const fn topology(&self) -> u64 {
        self.topology
    }

    /// Communication library.
    #[must_use]
    pub fn transport(&self) -> &str {
        &self.transport
    }

    /// Communication library version.
    #[must_use]
    pub const fn transport_version(&self) -> (u32, u32, u32) {
        self.transport_version
    }

    /// Determinism filter.
    #[must_use]
    pub const fn determinism(&self) -> Determinism {
        self.determinism
    }

    /// Stable cross-rank identity.
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.hash
    }
}

/// Fully validated two-rank collective tuning problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveTuningProblem {
    key: CollectiveTuningKey,
    elements: usize,
    message_bytes: usize,
    workspace_budget_bytes: usize,
}

impl CollectiveTuningProblem {
    /// Build a problem whose operation, dtype, and element count are static.
    pub fn new_static<K, Elements, Operation>(
        group: GroupId,
        topology: &TopologyFingerprint,
        determinism: Determinism,
        workspace_budget_bytes: usize,
    ) -> Result<Self, CollectiveTuningError>
    where
        K: ConstDType + CollectiveDType,
        Elements: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Operation: StaticCollectiveTuning<K, Elements>,
    {
        Self::new_dyn(
            Operation::KIND,
            K::DESCRIPTOR.builtin_id().unwrap_or(DTypeId::F32),
            Elements::USIZE,
            group,
            topology,
            determinism,
            workspace_budget_bytes,
        )
    }

    /// Build the matching runtime-selected (`Dyn`) problem.
    #[allow(clippy::too_many_arguments)]
    pub fn new_dyn(
        kind: CollectiveKind,
        dtype: DTypeId,
        elements: usize,
        group: GroupId,
        topology: &TopologyFingerprint,
        determinism: Determinism,
        workspace_budget_bytes: usize,
    ) -> Result<Self, CollectiveTuningError> {
        if group.ranks() != 2 {
            return Err(CollectiveTuningError::GroupSize {
                expected: 2,
                found: group.ranks(),
            });
        }
        if topology.devices().len() != group.ranks() {
            return Err(CollectiveTuningError::TopologyWorld {
                expected: group.ranks(),
                found: topology.devices().len(),
            });
        }
        if elements == 0 {
            return Err(CollectiveTuningError::ZeroElements);
        }
        if elements > u32::MAX as usize {
            return Err(CollectiveTuningError::ElementLimit {
                maximum: u32::MAX as usize,
                found: elements,
            });
        }
        validate_kind(kind, dtype, elements, group.ranks())?;
        let message_bytes = dtype
            .size_bytes(elements, OperationKind::Storage)
            .map_err(|_| CollectiveTuningError::MessageSize)?;
        let transport = topology.transport();
        let message_size_bucket = log2_bucket(message_bytes);
        let transport_version = transport.version();
        let mut key = CollectiveTuningKey {
            kind,
            dtype,
            message_size_bucket,
            group,
            topology: topology.digest(),
            transport: transport.library().into(),
            transport_version,
            determinism,
            hash: 0,
        };
        key.hash = key_hash(&key);
        Ok(Self {
            key,
            elements,
            message_bytes,
            workspace_budget_bytes,
        })
    }

    /// Stable problem/cache identity.
    #[must_use]
    pub const fn key(&self) -> &CollectiveTuningKey {
        &self.key
    }

    /// Exact logical element count used for legality checks.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Exact checked message byte count.
    #[must_use]
    pub const fn message_bytes(&self) -> usize {
        self.message_bytes
    }

    /// Maximum transient bytes allowed for a candidate.
    #[must_use]
    pub const fn workspace_budget_bytes(&self) -> usize {
        self.workspace_budget_bytes
    }
}

/// One transport candidate whose representation is stable across ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CollectiveTuningCandidate {
    algorithm: CollectiveAlgorithm,
    protocol: CollectiveProtocol,
    channels: u8,
    chunk_bytes: usize,
    stream_priority: i8,
    overlap_chunks: u8,
    workspace_bytes: usize,
    deterministic: bool,
    hash: u64,
}

impl CollectiveTuningCandidate {
    /// Construct a candidate with statically selected algorithm, protocol,
    /// channel count, and power-of-two chunk size.
    pub fn new_static<Algorithm, Protocol, Channels, ChunkBytes>(
        stream_priority: i8,
        overlap_chunks: u8,
        workspace_bytes: usize,
        deterministic: bool,
    ) -> Result<Self, CollectiveTuningError>
    where
        Algorithm: StaticCollectiveAlgorithm,
        Protocol: StaticCollectiveProtocol,
        Channels: Unsigned + NonZero + IsLessOrEqual<U32, Output = B1>,
        ChunkBytes: Unsigned + NonZero + PowerOfTwo,
    {
        Self::new_dyn(
            Algorithm::ALGORITHM,
            Protocol::PROTOCOL,
            Channels::USIZE,
            ChunkBytes::USIZE,
            stream_priority,
            overlap_chunks,
            workspace_bytes,
            deterministic,
        )
    }

    /// Construct the matching runtime-selected candidate.
    #[allow(clippy::too_many_arguments)]
    pub fn new_dyn(
        algorithm: CollectiveAlgorithm,
        protocol: CollectiveProtocol,
        channels: usize,
        chunk_bytes: usize,
        stream_priority: i8,
        overlap_chunks: u8,
        workspace_bytes: usize,
        deterministic: bool,
    ) -> Result<Self, CollectiveTuningError> {
        if !(1..=32).contains(&channels) {
            return Err(CollectiveTuningError::ChannelCount {
                minimum: 1,
                maximum: 32,
                found: channels,
            });
        }
        if chunk_bytes == 0 || !chunk_bytes.is_power_of_two() {
            return Err(CollectiveTuningError::ChunkBytes { found: chunk_bytes });
        }
        if !(-10..=10).contains(&stream_priority) {
            return Err(CollectiveTuningError::StreamPriority {
                minimum: -10,
                maximum: 10,
                found: stream_priority,
            });
        }
        let channels = channels as u8;
        let mut candidate = Self {
            algorithm,
            protocol,
            channels,
            chunk_bytes,
            stream_priority,
            overlap_chunks,
            workspace_bytes,
            deterministic,
            hash: 0,
        };
        candidate.hash = candidate_hash(candidate);
        Ok(candidate)
    }

    /// Algorithm family.
    #[must_use]
    pub const fn algorithm(self) -> CollectiveAlgorithm {
        self.algorithm
    }

    /// Protocol family.
    #[must_use]
    pub const fn protocol(self) -> CollectiveProtocol {
        self.protocol
    }

    /// Communication channels.
    #[must_use]
    pub const fn channels(self) -> u8 {
        self.channels
    }

    /// Chunk bytes.
    #[must_use]
    pub const fn chunk_bytes(self) -> usize {
        self.chunk_bytes
    }

    /// Requested stream priority.
    #[must_use]
    pub const fn stream_priority(self) -> i8 {
        self.stream_priority
    }

    /// Chunks allowed in the overlap window.
    #[must_use]
    pub const fn overlap_chunks(self) -> u8 {
        self.overlap_chunks
    }

    /// Candidate transient workspace.
    #[must_use]
    pub const fn workspace_bytes(self) -> usize {
        self.workspace_bytes
    }

    /// Whether this candidate may satisfy required determinism.
    #[must_use]
    pub const fn deterministic(self) -> bool {
        self.deterministic
    }

    /// Stable cross-rank identity.
    #[must_use]
    pub const fn hash(self) -> u64 {
        self.hash
    }

    #[cfg(feature = "autotune")]
    /// Converts this transport descriptor into the policy-neutral candidate
    /// metadata consumed by the general tuning service.
    pub fn service_candidate(
        self,
    ) -> Result<crate::tuning::service::TuningCandidate, crate::tuning::service::TuningServiceError>
    {
        crate::tuning::service::TuningCandidate::new(
            self.hash,
            &format!(
                "algorithm={:?},protocol={:?},channels={},chunk_bytes={},stream_priority={},overlap_chunks={},workspace_bytes={},deterministic={}",
                self.algorithm,
                self.protocol,
                self.channels,
                self.chunk_bytes,
                self.stream_priority,
                self.overlap_chunks,
                self.workspace_bytes,
                self.deterministic
            ),
            self.deterministic,
            self.workspace_bytes,
        )
    }
}

/// Hard bound on tuning search and measurement work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectiveTuningBudget {
    max_candidates: usize,
    samples_per_candidate: usize,
}

impl CollectiveTuningBudget {
    /// Build a statically nonzero bounded budget.
    #[must_use]
    pub fn new_static<Candidates, Samples>() -> Self
    where
        Candidates: Unsigned + NonZero + IsLessOrEqual<U32, Output = B1>,
        Samples: Unsigned + NonZero + IsLessOrEqual<U32, Output = B1>,
    {
        Self {
            max_candidates: Candidates::USIZE,
            samples_per_candidate: Samples::USIZE,
        }
    }

    /// Build the runtime-selected budget.
    pub const fn new_dyn(
        max_candidates: usize,
        samples_per_candidate: usize,
    ) -> Result<Self, CollectiveTuningError> {
        if max_candidates == 0 {
            return Err(CollectiveTuningError::ZeroCandidateBudget);
        }
        if samples_per_candidate == 0 {
            return Err(CollectiveTuningError::ZeroSampleBudget);
        }
        if max_candidates > 32 || samples_per_candidate > 32 {
            return Err(CollectiveTuningError::BudgetLimit {
                maximum: 32,
                candidates: max_candidates,
                samples: samples_per_candidate,
            });
        }
        Ok(Self {
            max_candidates,
            samples_per_candidate,
        })
    }

    /// Maximum candidates measured.
    #[must_use]
    pub const fn max_candidates(self) -> usize {
        self.max_candidates
    }

    /// Required synchronized samples from each rank and candidate.
    #[must_use]
    pub const fn samples_per_candidate(self) -> usize {
        self.samples_per_candidate
    }
}

/// One rank's synchronized samples for one candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankSampleReport {
    rank: usize,
    problem_hash: u64,
    candidate_hash: u64,
    samples_ns: Vec<u64>,
    success: bool,
    buffer_digest_before: u64,
    buffer_digest_after: u64,
}

impl RankSampleReport {
    /// Record one rank's complete measurement outcome.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        rank: usize,
        problem_hash: u64,
        candidate_hash: u64,
        samples_ns: Vec<u64>,
        success: bool,
        buffer_digest_before: u64,
        buffer_digest_after: u64,
    ) -> Self {
        Self {
            rank,
            problem_hash,
            candidate_hash,
            samples_ns,
            success,
            buffer_digest_before,
            buffer_digest_after,
        }
    }

    /// Reporting rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Synchronized duration samples.
    #[must_use]
    pub fn samples_ns(&self) -> &[u64] {
        &self.samples_ns
    }
}

/// All-rank reports for one candidate experiment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateRound {
    candidate: CollectiveTuningCandidate,
    reports: Vec<RankSampleReport>,
}

impl CandidateRound {
    /// Pair a candidate with one report per rank.
    #[must_use]
    pub fn new(candidate: CollectiveTuningCandidate, reports: Vec<RankSampleReport>) -> Self {
        Self { candidate, reports }
    }

    /// Candidate measured by the round.
    #[must_use]
    pub const fn candidate(&self) -> CollectiveTuningCandidate {
        self.candidate
    }

    /// Rank reports.
    #[must_use]
    pub fn reports(&self) -> &[RankSampleReport] {
        &self.reports
    }
}

/// Selected measurement awaiting unanimous rank commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvisionalCollectiveTuning {
    candidate: CollectiveTuningCandidate,
    problem_hash: u64,
    candidate_hash: u64,
    median_max_rank_ns: u64,
    rank_median_ns: [u64; 2],
    sample_count: usize,
}

impl ProvisionalCollectiveTuning {
    /// Selected candidate.
    #[must_use]
    pub const fn candidate(self) -> CollectiveTuningCandidate {
        self.candidate
    }

    /// Problem identity all ranks must vote on.
    #[must_use]
    pub const fn problem_hash(self) -> u64 {
        self.problem_hash
    }

    /// Candidate identity all ranks must vote on.
    #[must_use]
    pub const fn candidate_hash(self) -> u64 {
        self.candidate_hash
    }

    /// Median of per-sample maximum-rank durations.
    #[must_use]
    pub const fn median_max_rank_ns(self) -> u64 {
        self.median_max_rank_ns
    }

    /// Each rank's own median, retained for imbalance diagnostics.
    #[must_use]
    pub const fn rank_median_ns(self) -> [u64; 2] {
        self.rank_median_ns
    }

    /// Samples contributed by each rank.
    #[must_use]
    pub const fn sample_count(self) -> usize {
        self.sample_count
    }
}

/// One rank's final commit vote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitVote {
    rank: usize,
    problem_hash: u64,
    candidate_hash: u64,
    median_max_rank_ns: u64,
    accepted: bool,
}

impl CommitVote {
    /// Build a vote for an exact provisional result.
    #[must_use]
    pub const fn for_result(
        rank: usize,
        result: ProvisionalCollectiveTuning,
        accepted: bool,
    ) -> Self {
        Self {
            rank,
            problem_hash: result.problem_hash,
            candidate_hash: result.candidate_hash,
            median_max_rank_ns: result.median_max_rank_ns,
            accepted,
        }
    }

    /// Build a raw vote received from another process.
    #[must_use]
    pub const fn from_wire(
        rank: usize,
        problem_hash: u64,
        candidate_hash: u64,
        median_max_rank_ns: u64,
        accepted: bool,
    ) -> Self {
        Self {
            rank,
            problem_hash,
            candidate_hash,
            median_max_rank_ns,
            accepted,
        }
    }
}

/// Unanimously committed result; cache insertion should require this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommittedCollectiveTuning(ProvisionalCollectiveTuning);

impl CommittedCollectiveTuning {
    /// The measured result every rank accepted.
    #[must_use]
    pub const fn result(self) -> ProvisionalCollectiveTuning {
        self.0
    }
}

/// Select the legal candidate with the smallest maximum-rank objective.
pub fn select_collective_candidate(
    problem: &CollectiveTuningProblem,
    candidates: &[CollectiveTuningCandidate],
    rounds: &[CandidateRound],
    budget: CollectiveTuningBudget,
) -> Result<ProvisionalCollectiveTuning, CollectiveTuningError> {
    if candidates.is_empty() {
        return Err(CollectiveTuningError::EmptyCandidates);
    }
    if candidates.len() > budget.max_candidates {
        return Err(CollectiveTuningError::CandidateBudgetExceeded {
            maximum: budget.max_candidates,
            found: candidates.len(),
        });
    }
    if rounds.len() != candidates.len() {
        return Err(CollectiveTuningError::RoundCount {
            expected: candidates.len(),
            found: rounds.len(),
        });
    }

    let mut seen = Vec::with_capacity(candidates.len());
    let mut scored = Vec::with_capacity(candidates.len());
    for (index, (&candidate, round)) in candidates.iter().zip(rounds).enumerate() {
        validate_candidate(problem, candidate)?;
        if seen.contains(&candidate.hash) {
            return Err(CollectiveTuningError::DuplicateCandidate {
                hash: candidate.hash,
            });
        }
        seen.push(candidate.hash);
        if round.candidate != candidate {
            return Err(CollectiveTuningError::RoundCandidate {
                index,
                expected: candidate.hash,
                found: round.candidate.hash,
            });
        }
        let score = score_round(problem, round, budget.samples_per_candidate)?;
        scored.push((score.median_max_rank_ns, candidate.hash, score));
    }
    scored.sort_unstable_by_key(|&(score, hash, _)| (score, hash));
    Ok(scored[0].2)
}

/// Require one agreeing positive vote from each rank.
pub fn commit_collective_tuning(
    result: ProvisionalCollectiveTuning,
    votes: &[CommitVote],
) -> Result<CommittedCollectiveTuning, CollectiveTuningError> {
    if votes.len() != 2 {
        return Err(CollectiveTuningError::CommitVoteCount {
            expected: 2,
            found: votes.len(),
        });
    }
    let mut seen = [false; 2];
    for vote in votes {
        if vote.rank >= 2 {
            return Err(CollectiveTuningError::RankOutOfRange {
                rank: vote.rank,
                ranks: 2,
            });
        }
        if seen[vote.rank] {
            return Err(CollectiveTuningError::DuplicateRank { rank: vote.rank });
        }
        seen[vote.rank] = true;
        if !vote.accepted {
            return Err(CollectiveTuningError::CommitRejected { rank: vote.rank });
        }
        if vote.problem_hash != result.problem_hash
            || vote.candidate_hash != result.candidate_hash
            || vote.median_max_rank_ns != result.median_max_rank_ns
        {
            return Err(CollectiveTuningError::CommitMismatch { rank: vote.rank });
        }
    }
    Ok(CommittedCollectiveTuning(result))
}

fn score_round(
    problem: &CollectiveTuningProblem,
    round: &CandidateRound,
    samples: usize,
) -> Result<ProvisionalCollectiveTuning, CollectiveTuningError> {
    if round.reports.len() != 2 {
        return Err(CollectiveTuningError::RankReportCount {
            expected: 2,
            found: round.reports.len(),
        });
    }
    let mut by_rank: [Option<&RankSampleReport>; 2] = [None, None];
    for report in &round.reports {
        if report.rank >= 2 {
            return Err(CollectiveTuningError::RankOutOfRange {
                rank: report.rank,
                ranks: 2,
            });
        }
        if by_rank[report.rank].replace(report).is_some() {
            return Err(CollectiveTuningError::DuplicateRank { rank: report.rank });
        }
        if report.problem_hash != problem.key.hash {
            return Err(CollectiveTuningError::ProblemHash {
                rank: report.rank,
                expected: problem.key.hash,
                found: report.problem_hash,
            });
        }
        if report.candidate_hash != round.candidate.hash {
            return Err(CollectiveTuningError::CandidateHash {
                rank: report.rank,
                expected: round.candidate.hash,
                found: report.candidate_hash,
            });
        }
        if !report.success {
            return Err(CollectiveTuningError::RankMeasurementFailed { rank: report.rank });
        }
        if report.buffer_digest_before != report.buffer_digest_after {
            return Err(CollectiveTuningError::MeasurementMutatedBuffer {
                rank: report.rank,
                before: report.buffer_digest_before,
                after: report.buffer_digest_after,
            });
        }
        if report.samples_ns.len() != samples {
            return Err(CollectiveTuningError::SampleCount {
                rank: report.rank,
                expected: samples,
                found: report.samples_ns.len(),
            });
        }
    }
    let rank_zero = by_rank[0].ok_or(CollectiveTuningError::MissingRank { rank: 0 })?;
    let rank_one = by_rank[1].ok_or(CollectiveTuningError::MissingRank { rank: 1 })?;
    let mut maxima = Vec::with_capacity(samples);
    for index in 0..samples {
        maxima.push(core::cmp::max(
            rank_zero.samples_ns[index],
            rank_one.samples_ns[index],
        ));
    }
    let median_max_rank_ns = median(&mut maxima);
    let mut zero_samples = rank_zero.samples_ns.clone();
    let mut one_samples = rank_one.samples_ns.clone();
    Ok(ProvisionalCollectiveTuning {
        candidate: round.candidate,
        problem_hash: problem.key.hash,
        candidate_hash: round.candidate.hash,
        median_max_rank_ns,
        rank_median_ns: [median(&mut zero_samples), median(&mut one_samples)],
        sample_count: samples,
    })
}

fn validate_candidate(
    problem: &CollectiveTuningProblem,
    candidate: CollectiveTuningCandidate,
) -> Result<(), CollectiveTuningError> {
    if candidate.workspace_bytes > problem.workspace_budget_bytes {
        return Err(CollectiveTuningError::WorkspaceExceeded {
            required: candidate.workspace_bytes,
            budget: problem.workspace_budget_bytes,
        });
    }
    if problem.key.determinism.is_required() && !candidate.deterministic {
        return Err(CollectiveTuningError::NondeterministicCandidate {
            hash: candidate.hash,
        });
    }
    if candidate.chunk_bytes > problem.message_bytes {
        return Err(CollectiveTuningError::ChunkExceedsMessage {
            chunk: candidate.chunk_bytes,
            message: problem.message_bytes,
        });
    }
    Ok(())
}

fn validate_kind(
    kind: CollectiveKind,
    dtype: DTypeId,
    elements: usize,
    ranks: usize,
) -> Result<(), CollectiveTuningError> {
    match kind {
        CollectiveKind::AllReduce(op) | CollectiveKind::ReduceScatter(op) => {
            incin_core::dist::validate_collective_reduction(dtype, op)?;
            if matches!(kind, CollectiveKind::ReduceScatter(_)) && !elements.is_multiple_of(ranks) {
                return Err(CollectiveTuningError::NonDivisible { elements, ranks });
            }
        }
        CollectiveKind::AllGather => {
            incin_core::dist::validate_collective_dtype(dtype)?;
        }
        CollectiveKind::AllToAll => {
            incin_core::dist::validate_collective_dtype(dtype)?;
            if !elements.is_multiple_of(ranks) {
                return Err(CollectiveTuningError::NonDivisible { elements, ranks });
            }
        }
        CollectiveKind::SendRecv {
            source,
            destination,
        } => {
            incin_core::dist::validate_collective_dtype(dtype)?;
            if source >= ranks {
                return Err(CollectiveTuningError::PeerOutOfRange {
                    endpoint: "source",
                    rank: source,
                    ranks,
                });
            }
            if destination >= ranks {
                return Err(CollectiveTuningError::PeerOutOfRange {
                    endpoint: "destination",
                    rank: destination,
                    ranks,
                });
            }
            if source == destination {
                return Err(CollectiveTuningError::SamePeer { rank: source });
            }
        }
        _ => return Err(CollectiveTuningError::UnsupportedCollectiveKind),
    }
    Ok(())
}

fn median(samples: &mut [u64]) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn log2_bucket(size: usize) -> u8 {
    if size <= 1 {
        0
    } else {
        (usize::BITS - (size - 1).leading_zeros()).min(u8::MAX.into()) as u8
    }
}

fn key_hash(key: &CollectiveTuningKey) -> u64 {
    let mut digest = StableDigest::new()
        .bytes(b"incin.collective.tuning.key.v1")
        .collective(key.kind)
        .bytes(key.dtype.name().as_bytes())
        .number(u64::from(key.message_size_bucket))
        .number(key.group.token())
        .number(key.group.ranks() as u64)
        .number(key.topology)
        .bytes(key.transport.as_bytes())
        .number(u64::from(key.transport_version.0))
        .number(u64::from(key.transport_version.1))
        .number(u64::from(key.transport_version.2))
        .bytes(key.determinism.as_str().as_bytes());
    digest = digest.number(1);
    digest.finish()
}

fn candidate_hash(candidate: CollectiveTuningCandidate) -> u64 {
    StableDigest::new()
        .bytes(b"incin.collective.tuning.candidate.v1")
        .number(match candidate.algorithm {
            CollectiveAlgorithm::Ring => 1,
            CollectiveAlgorithm::Tree => 2,
        })
        .number(match candidate.protocol {
            CollectiveProtocol::Simple => 1,
            CollectiveProtocol::LowLatency => 2,
            CollectiveProtocol::LowLatency128 => 3,
        })
        .number(u64::from(candidate.channels))
        .number(candidate.chunk_bytes as u64)
        .number(candidate.stream_priority as i64 as u64)
        .number(u64::from(candidate.overlap_chunks))
        .number(candidate.workspace_bytes as u64)
        .number(u64::from(candidate.deterministic))
        .finish()
}

#[derive(Debug, Clone, Copy)]
struct StableDigest(u64);

impl StableDigest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        self = self.number(bytes.len() as u64);
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    fn number(mut self, value: u64) -> Self {
        for byte in value.to_le_bytes() {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    fn collective(self, kind: CollectiveKind) -> Self {
        match kind {
            CollectiveKind::AllReduce(op) => self.bytes(b"all-reduce").reduce(op),
            CollectiveKind::AllGather => self.bytes(b"all-gather"),
            CollectiveKind::ReduceScatter(op) => self.bytes(b"reduce-scatter").reduce(op),
            CollectiveKind::AllToAll => self.bytes(b"all-to-all"),
            CollectiveKind::SendRecv {
                source,
                destination,
            } => self
                .bytes(b"send-recv")
                .number(source as u64)
                .number(destination as u64),
            _ => self.bytes(b"unknown"),
        }
    }

    fn reduce(self, op: ReduceOp) -> Self {
        self.bytes(match op {
            ReduceOp::Sum => b"sum",
            ReduceOp::Mean => b"mean",
            ReduceOp::Max => b"max",
            ReduceOp::Min => b"min",
            ReduceOp::Prod => b"prod",
        })
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

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
