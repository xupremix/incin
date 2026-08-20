use super::*;

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

pub(super) fn validate_kind(
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

pub(super) fn log2_bucket(size: usize) -> u8 {
    if size <= 1 {
        0
    } else {
        (usize::BITS - (size - 1).leading_zeros()).min(u8::MAX.into()) as u8
    }
}

pub(super) fn key_hash(key: &CollectiveTuningKey) -> u64 {
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

pub(super) fn candidate_hash(candidate: CollectiveTuningCandidate) -> u64 {
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
