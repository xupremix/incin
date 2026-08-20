use super::candidate::TuningCandidate;
use super::engine::TuningPermit;
use crate::tuning::cache::CacheRecord;
use core::fmt;

/// How a selection was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SelectionSource {
    /// Declared safe fallback under `Disabled`.
    DisabledFallback,
    /// Caller-provided analytical heuristic.
    Heuristic,
    /// A verified result committed earlier in this process or cache.
    WarmupCache,
    /// A verified profile-database result.
    Profile,
    /// A newly committed measurement.
    Measurement,
}

/// Candidate selected without further measurement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningSelection {
    pub(super) candidate: TuningCandidate,
    pub(super) source: SelectionSource,
    pub(super) median_ns: Option<u64>,
    pub(super) sample_count: u32,
}

impl TuningSelection {
    /// Selected candidate.
    #[must_use]
    pub const fn candidate(&self) -> &TuningCandidate {
        &self.candidate
    }

    /// Selection provenance.
    #[must_use]
    pub const fn source(&self) -> SelectionSource {
        self.source
    }

    /// Measured median when available.
    pub const fn median_ns(&self) -> Option<u64> {
        self.median_ns
    }

    /// Synchronized samples behind the result.
    #[must_use]
    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }
}

/// Result of consulting the service.
pub enum ServiceDecision {
    /// Use a selected candidate now.
    Selected(TuningSelection),
    /// The caller owns the only active measurement lease for this key.
    Measure(TuningPermit),
}

impl fmt::Debug for ServiceDecision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Selected(selection) => {
                formatter.debug_tuple("Selected").field(selection).finish()
            }
            Self::Measure(permit) => formatter.debug_tuple("Measure").field(permit).finish(),
        }
    }
}

/// One participant's vote on a coordinated measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatedVote {
    pub(super) rank: usize,
    pub(super) epoch: u64,
    pub(super) candidate_hash: u64,
    pub(super) accepted: bool,
}

impl CoordinatedVote {
    /// Constructs a vote received from a participant.
    #[must_use]
    pub const fn new(rank: usize, epoch: u64, candidate_hash: u64, accepted: bool) -> Self {
        Self {
            rank,
            epoch,
            candidate_hash,
            accepted,
        }
    }
}

pub(super) fn selection_from_record(
    record: &CacheRecord,
    legal: &[TuningCandidate],
    source: SelectionSource,
) -> Option<TuningSelection> {
    let candidate = legal
        .iter()
        .find(|candidate| candidate.encoding == record.winner())?
        .clone();
    Some(TuningSelection {
        candidate,
        source,
        median_ns: record.median_ns(),
        sample_count: record.sample_count(),
    })
}
