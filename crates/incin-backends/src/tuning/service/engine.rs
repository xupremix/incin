use super::candidate::{TuningCandidate, legal_candidates_digest};
use super::context::{
    AutotunePolicy, CoordinatedWarmupTuning, DisabledTuning, HeuristicTuning, ProfileGuidedTuning,
    StaticAutotunePolicy, TuningContext,
};
use super::decision::{
    CoordinatedVote, SelectionSource, ServiceDecision, TuningSelection, selection_from_record,
};
use super::error::TuningServiceError;
use crate::tuning::cache::{CacheKey, CacheRecord, MeasurementMethod, PersistentTuningCache};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::{fmt, marker::PhantomData, time::Duration};
use incin_core::prelude::Dyn;
use std::path::Path;
use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

#[derive(Debug, Clone)]
struct ActiveLease {
    epoch: u64,
    deadline: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ResultKey {
    key: CacheKey<Dyn>,
    legal_digest: u64,
}

#[derive(Default)]
struct ServiceState {
    leases: BTreeMap<CacheKey<Dyn>, ActiveLease>,
    results: BTreeMap<ResultKey, CacheRecord>,
}

struct ServiceInner {
    policy: AutotunePolicy,
    state: Mutex<ServiceState>,
    ready: Condvar,
    cache: Mutex<Option<PersistentTuningCache>>,
    next_epoch: AtomicU64,
}

/// Policy-aware tuning service.
///
/// `P` is one of the static policy markers or `Dyn` for runtime policy
/// selection.
pub struct TuningService<P = Dyn> {
    inner: Arc<ServiceInner>,
    marker: PhantomData<fn() -> P>,
}

impl TuningService<DisabledTuning> {
    /// Constructs a statically disabled service.
    #[must_use]
    pub fn disabled() -> Self {
        Self::from_policy(AutotunePolicy::Disabled, None)
    }
}

impl TuningService<HeuristicTuning> {
    /// Constructs a statically heuristic service.
    #[must_use]
    pub fn heuristic() -> Self {
        Self::from_policy(AutotunePolicy::Heuristic, None)
    }
}

impl TuningService<CoordinatedWarmupTuning> {
    /// Constructs a statically coordinated service.
    pub fn coordinated_warmup(budget: Duration) -> core::result::Result<Self, TuningServiceError> {
        if budget.is_zero() {
            return Err(TuningServiceError::ZeroWarmupBudget);
        }
        Ok(Self::from_policy(
            AutotunePolicy::CoordinatedWarmup { budget },
            None,
        ))
    }
}

impl TuningService<ProfileGuidedTuning> {
    /// Opens a statically profile-guided service.
    pub fn profile_guided(
        database: impl AsRef<Path>,
        limits: crate::tuning::cache::CacheLimits,
    ) -> core::result::Result<Self, TuningServiceError> {
        let database = database.as_ref().to_path_buf();
        let cache = PersistentTuningCache::open(&database, limits)?;
        Ok(Self::from_policy(
            AutotunePolicy::ProfileGuided { database },
            Some(cache),
        ))
    }
}

impl TuningService<Dyn> {
    /// Constructs the runtime-selected policy, validating its budget and
    /// opening a profile database when required.
    pub fn new_dyn(
        policy: AutotunePolicy,
        profile_limits: crate::tuning::cache::CacheLimits,
    ) -> core::result::Result<Self, TuningServiceError> {
        let cache = match &policy {
            AutotunePolicy::CoordinatedWarmup { budget } if budget.is_zero() => {
                return Err(TuningServiceError::ZeroWarmupBudget);
            }
            AutotunePolicy::ProfileGuided { database } => {
                Some(PersistentTuningCache::open(database, profile_limits)?)
            }
            _ => None,
        };
        Ok(Self::from_policy(policy, cache))
    }
}

impl<P> TuningService<P> {
    fn from_policy(policy: AutotunePolicy, cache: Option<PersistentTuningCache>) -> Self {
        Self {
            inner: Arc::new(ServiceInner {
                policy,
                state: Mutex::new(ServiceState::default()),
                ready: Condvar::new(),
                cache: Mutex::new(cache),
                next_epoch: AtomicU64::new(1),
            }),
            marker: PhantomData,
        }
    }

    /// Adds a writable persistent cache to a coordinated service.
    #[must_use]
    pub fn with_cache(self, cache: PersistentTuningCache) -> Self {
        *self
            .inner
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(cache);
        self
    }

    /// Runtime policy projection.
    #[must_use]
    pub fn policy(&self) -> &AutotunePolicy {
        &self.inner.policy
    }

    /// Selects, waits for, or leases one tuning problem.
    #[allow(clippy::too_many_arguments)]
    pub fn decide<D, S>(
        &self,
        context: &TuningContext<D, S>,
        key: CacheKey<Dyn>,
        candidates: &[TuningCandidate],
        fallback_hash: u64,
        heuristic_hash: u64,
    ) -> core::result::Result<ServiceDecision, TuningServiceError> {
        validate_key(context, &key)?;
        let legal = legal_candidates(context, candidates)?;
        let fallback = legal
            .iter()
            .find(|candidate| candidate.hash == fallback_hash)
            .cloned()
            .ok_or(TuningServiceError::IllegalFallback {
                hash: fallback_hash,
            })?;
        let heuristic = legal
            .iter()
            .find(|candidate| candidate.hash == heuristic_hash)
            .cloned()
            .ok_or(TuningServiceError::IllegalHeuristic {
                hash: heuristic_hash,
            })?;
        let legal_digest = legal_candidates_digest(&legal);

        match &self.inner.policy {
            AutotunePolicy::Disabled => {
                return Ok(ServiceDecision::Selected(TuningSelection {
                    candidate: fallback,
                    source: SelectionSource::DisabledFallback,
                    median_ns: None,
                    sample_count: 0,
                }));
            }
            AutotunePolicy::Heuristic => {
                return Ok(ServiceDecision::Selected(TuningSelection {
                    candidate: heuristic,
                    source: SelectionSource::Heuristic,
                    median_ns: None,
                    sample_count: 0,
                }));
            }
            AutotunePolicy::ProfileGuided { .. } => {
                if let Some(selection) =
                    self.cached_selection(&key, legal_digest, &legal, SelectionSource::Profile)
                {
                    return Ok(ServiceDecision::Selected(selection));
                }
                return Ok(ServiceDecision::Selected(TuningSelection {
                    candidate: heuristic,
                    source: SelectionSource::Heuristic,
                    median_ns: None,
                    sample_count: 0,
                }));
            }
            AutotunePolicy::CoordinatedWarmup { .. } => {}
        }

        if let Some(selection) =
            self.cached_selection(&key, legal_digest, &legal, SelectionSource::WarmupCache)
        {
            return Ok(ServiceDecision::Selected(selection));
        }

        let policy_budget = match self.inner.policy {
            AutotunePolicy::CoordinatedWarmup { budget } => budget,
            _ => unreachable!("non-coordinated policies returned above"),
        };
        let lease_budget = core::cmp::min(policy_budget, context.time_budget);
        let wait_deadline = Instant::now()
            .checked_add(context.time_budget)
            .ok_or(TuningServiceError::ZeroTimeBudget)?;
        let result_key = ResultKey {
            key: key.clone(),
            legal_digest,
        };
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        loop {
            if let Some(record) = state.results.get(&result_key)
                && let Some(selection) =
                    selection_from_record(record, &legal, SelectionSource::WarmupCache)
            {
                return Ok(ServiceDecision::Selected(selection));
            }
            let now = Instant::now();
            if let Some(active) = state.leases.get(&key).cloned() {
                if now >= active.deadline {
                    state.leases.remove(&key);
                    self.inner.ready.notify_all();
                    continue;
                }
                if now >= wait_deadline {
                    return Err(TuningServiceError::WaitTimeout);
                }
                let wake_at = core::cmp::min(active.deadline, wait_deadline);
                let duration = wake_at.saturating_duration_since(now);
                let (next, timeout) = self
                    .inner
                    .ready
                    .wait_timeout(state, duration)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                state = next;
                if timeout.timed_out() && Instant::now() >= wait_deadline {
                    return Err(TuningServiceError::WaitTimeout);
                }
                continue;
            }

            let epoch = self.inner.next_epoch.fetch_add(1, Ordering::Relaxed);
            let deadline = Instant::now()
                .checked_add(lease_budget)
                .ok_or(TuningServiceError::ZeroTimeBudget)?;
            state
                .leases
                .insert(key.clone(), ActiveLease { epoch, deadline });
            drop(state);
            let participants = match context.topology.as_ref() {
                Some(topology) => (0..topology.world()).collect(),
                None => vec![0],
            };
            let candidates = legal
                .into_iter()
                .map(|candidate| (candidate.hash, candidate))
                .collect();
            return Ok(ServiceDecision::Measure(TuningPermit {
                inner: Arc::clone(&self.inner),
                key,
                legal_digest,
                candidates,
                participants,
                epoch,
                deadline,
                completed: false,
            }));
        }
    }

    fn cached_selection(
        &self,
        key: &CacheKey<Dyn>,
        legal_digest: u64,
        legal: &[TuningCandidate],
        source: SelectionSource,
    ) -> Option<TuningSelection> {
        let result_key = ResultKey {
            key: key.clone(),
            legal_digest,
        };
        if let Some(record) = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .results
            .get(&result_key)
            .cloned()
            && let Some(selection) = selection_from_record(&record, legal, source)
        {
            return Some(selection);
        }
        let cache = self
            .inner
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let record = cache.as_ref()?.lookup(key, legal_digest)?;
        selection_from_record(record, legal, source)
    }
}

impl<P: StaticAutotunePolicy> TuningService<P> {
    /// Erases the static policy marker.
    #[must_use]
    pub fn erase(self) -> TuningService<Dyn> {
        TuningService {
            inner: self.inner,
            marker: PhantomData,
        }
    }
}

fn validate_key<D, S>(
    context: &TuningContext<D, S>,
    key: &CacheKey<Dyn>,
) -> core::result::Result<(), TuningServiceError> {
    if key.namespace() != context.scope.namespace() {
        return Err(TuningServiceError::KeyContextMismatch { field: "scope" });
    }
    if key.backend() != context.environment.device().backend() {
        return Err(TuningServiceError::KeyContextMismatch { field: "backend" });
    }
    if key.environment_digest() != context.environment.digest() {
        return Err(TuningServiceError::KeyContextMismatch {
            field: "environment",
        });
    }
    Ok(())
}

fn legal_candidates<D, S>(
    context: &TuningContext<D, S>,
    candidates: &[TuningCandidate],
) -> core::result::Result<Vec<TuningCandidate>, TuningServiceError> {
    if candidates.is_empty() {
        return Err(TuningServiceError::EmptyCandidates);
    }
    let mut seen = BTreeSet::new();
    for candidate in candidates {
        if !seen.insert(candidate.hash) {
            return Err(TuningServiceError::DuplicateCandidate {
                hash: candidate.hash,
            });
        }
    }
    let mut legal: Vec<_> = candidates
        .iter()
        .filter(|candidate| {
            (!context.determinism.is_required() || candidate.deterministic)
                && candidate.workspace_bytes <= context.memory_budget
        })
        .cloned()
        .collect();
    legal.sort_by(|left, right| (left.hash, &left.encoding).cmp(&(right.hash, &right.encoding)));
    if legal.is_empty() {
        return Err(TuningServiceError::NoLegalCandidates);
    }
    Ok(legal)
}

/// Exclusive bounded right to measure one tuning key.
pub struct TuningPermit {
    inner: Arc<ServiceInner>,
    key: CacheKey<Dyn>,
    legal_digest: u64,
    candidates: BTreeMap<u64, TuningCandidate>,
    participants: Vec<usize>,
    epoch: u64,
    deadline: Instant,
    completed: bool,
}

impl fmt::Debug for TuningPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TuningPermit")
            .field("key", &self.key)
            .field("legal_digest", &self.legal_digest)
            .field("participants", &self.participants)
            .field("epoch", &self.epoch)
            .field("deadline", &self.deadline)
            .finish_non_exhaustive()
    }
}

impl TuningPermit {
    /// Monotonic lease epoch.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Exact participant ranks which must commit.
    #[must_use]
    pub fn participants(&self) -> &[usize] {
        &self.participants
    }

    /// Stable digest of the filtered legal set.
    #[must_use]
    pub const fn legal_candidates_digest(&self) -> u64 {
        self.legal_digest
    }

    /// Commits a one-participant measurement.
    pub fn commit_local(
        self,
        candidate_hash: u64,
        median_ns: u64,
        sample_count: u32,
    ) -> core::result::Result<TuningSelection, TuningServiceError> {
        if self.participants.len() != 1 {
            return Err(TuningServiceError::LocalCommitForDistributed {
                participants: self.participants.len(),
            });
        }
        self.finish(candidate_hash, median_ns, sample_count)
    }

    /// Commits a result accepted by every participant in this permit's exact
    /// epoch.
    pub fn commit_coordinated(
        self,
        candidate_hash: u64,
        median_ns: u64,
        sample_count: u32,
        votes: &[CoordinatedVote],
    ) -> core::result::Result<TuningSelection, TuningServiceError> {
        if votes.len() != self.participants.len() {
            return Err(TuningServiceError::VoteCount {
                expected: self.participants.len(),
                found: votes.len(),
            });
        }
        let mut seen = BTreeSet::new();
        for vote in votes {
            if !self.participants.contains(&vote.rank) {
                return Err(TuningServiceError::VoteRank {
                    rank: vote.rank,
                    participants: self.participants.len(),
                });
            }
            if !seen.insert(vote.rank) {
                return Err(TuningServiceError::DuplicateVote { rank: vote.rank });
            }
            if !vote.accepted || vote.epoch != self.epoch || vote.candidate_hash != candidate_hash {
                return Err(TuningServiceError::VoteMismatch { rank: vote.rank });
            }
        }
        self.finish(candidate_hash, median_ns, sample_count)
    }

    #[cfg(feature = "distributed")]
    /// Commits the already-unanimous result minted by the two-rank collective
    /// coordinator.
    pub fn commit_collective(
        self,
        committed: crate::dist::tuning::CommittedCollectiveTuning,
    ) -> core::result::Result<TuningSelection, TuningServiceError> {
        let result = committed.result();
        if self.participants != [0, 1] {
            return Err(TuningServiceError::VoteCount {
                expected: self.participants.len(),
                found: 2,
            });
        }
        self.finish(
            result.candidate_hash(),
            result.median_max_rank_ns(),
            u32::try_from(result.sample_count()).map_err(|_| TuningServiceError::ZeroSamples)?,
        )
    }

    /// Explicitly releases a lease without committing.
    pub fn cancel(mut self) {
        self.release();
    }

    fn finish(
        mut self,
        candidate_hash: u64,
        median_ns: u64,
        sample_count: u32,
    ) -> core::result::Result<TuningSelection, TuningServiceError> {
        if sample_count == 0 {
            return Err(TuningServiceError::ZeroSamples);
        }
        if Instant::now() > self.deadline {
            return Err(TuningServiceError::PermitExpired { epoch: self.epoch });
        }
        let candidate = self.candidates.get(&candidate_hash).cloned().ok_or(
            TuningServiceError::CandidateNotLegal {
                hash: candidate_hash,
            },
        )?;
        {
            let state = self
                .inner
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !matches!(
                state.leases.get(&self.key),
                Some(active) if active.epoch == self.epoch
            ) {
                return Err(TuningServiceError::PermitNotActive { epoch: self.epoch });
            }
        }
        let record = CacheRecord::new(
            self.key.clone(),
            MeasurementMethod::CoordinatedWarmup,
            sample_count,
            Some(median_ns),
            self.legal_digest,
            &candidate.encoding,
        )?;
        if let Some(cache) = self
            .inner
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_mut()
        {
            cache.commit(record.clone())?;
        }
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !matches!(
            state.leases.get(&self.key),
            Some(active) if active.epoch == self.epoch
        ) {
            return Err(TuningServiceError::PermitNotActive { epoch: self.epoch });
        }
        state.leases.remove(&self.key);
        state.results.insert(
            ResultKey {
                key: self.key.clone(),
                legal_digest: self.legal_digest,
            },
            record,
        );
        drop(state);
        self.completed = true;
        self.inner.ready.notify_all();
        Ok(TuningSelection {
            candidate,
            source: SelectionSource::Measurement,
            median_ns: Some(median_ns),
            sample_count,
        })
    }

    fn release(&mut self) {
        if self.completed {
            return;
        }
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(
            state.leases.get(&self.key),
            Some(active) if active.epoch == self.epoch
        ) {
            state.leases.remove(&self.key);
        }
        drop(state);
        self.completed = true;
        self.inner.ready.notify_all();
    }
}

impl Drop for TuningPermit {
    fn drop(&mut self) {
        self.release();
    }
}
