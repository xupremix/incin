//! Policy-aware tuning orchestration.
//!
//! The service never benchmarks by itself. It filters a caller-supplied legal
//! candidate set, chooses deterministic fallback/heuristic results, issues a
//! bounded single-flight permit for coordinated measurement, and commits only
//! a complete local or all-participant result. Profile imports and warmup
//! results pass through the same persistent-cache legality checks.

use super::{
    cache::{
        CacheError, CacheKey, CacheLimits, CacheRecord, MeasurementMethod, PersistentTuningCache,
    },
    identity::{StaticBackend, TuningEnvironmentFingerprint, TuningTopologyFingerprint},
};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::{fmt, marker::PhantomData, time::Duration};
use incin_core::{exec::Determinism, prelude::Dyn};
use std::{
    path::{Path, PathBuf},
    sync::{
        Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

const MAX_CANDIDATE_ENCODING_BYTES: usize = 4096;

/// The operation layer whose choices are being tuned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum TuningScope {
    /// A local device kernel or library call.
    Kernel,
    /// A collective run by every rank in a communicator.
    Collective,
    /// A complete execution plan.
    ExecutionPlan,
}

impl TuningScope {
    /// Stable cache namespace.
    #[must_use]
    pub const fn namespace(self) -> &'static str {
        match self {
            Self::Kernel => "kernel",
            Self::Collective => "collective",
            Self::ExecutionPlan => "plan",
        }
    }

    const fn requires_topology(self) -> bool {
        matches!(self, Self::Collective | Self::ExecutionPlan)
    }
}

mod sealed {
    pub trait Scope {}
    pub trait Policy {}
}

/// A compile-time tuning-scope marker.
pub trait StaticTuningScope: sealed::Scope + 'static {
    /// Runtime projection.
    const SCOPE: TuningScope;
}

/// Static local-kernel scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelTuning;
impl sealed::Scope for KernelTuning {}
impl StaticTuningScope for KernelTuning {
    const SCOPE: TuningScope = TuningScope::Kernel;
}

/// Static collective scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectiveTuning;
impl sealed::Scope for CollectiveTuning {}
impl StaticTuningScope for CollectiveTuning {
    const SCOPE: TuningScope = TuningScope::Collective;
}

/// Static execution-plan scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPlanTuning;
impl sealed::Scope for ExecutionPlanTuning {}
impl StaticTuningScope for ExecutionPlanTuning {
    const SCOPE: TuningScope = TuningScope::ExecutionPlan;
}

/// User-visible tuning policy.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AutotunePolicy {
    /// Use the declared deterministic safe fallback.
    Disabled,
    /// Use the caller's analytical heuristic without measurement.
    Heuristic,
    /// Permit synchronized measurement within a hard time budget.
    CoordinatedWarmup {
        /// Maximum lease duration.
        budget: Duration,
    },
    /// Consume an offline or deployment profile without measuring.
    ProfileGuided {
        /// Persistent profile database.
        database: PathBuf,
    },
}

/// A compile-time tuning-policy marker.
pub trait StaticAutotunePolicy: sealed::Policy + 'static {}

/// Static disabled-policy marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisabledTuning;
impl sealed::Policy for DisabledTuning {}
impl StaticAutotunePolicy for DisabledTuning {}

/// Static heuristic-policy marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeuristicTuning;
impl sealed::Policy for HeuristicTuning {}
impl StaticAutotunePolicy for HeuristicTuning {}

/// Static coordinated-warmup marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinatedWarmupTuning;
impl sealed::Policy for CoordinatedWarmupTuning {}
impl StaticAutotunePolicy for CoordinatedWarmupTuning {}

/// Static profile-guided marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileGuidedTuning;
impl sealed::Policy for ProfileGuidedTuning {}
impl StaticAutotunePolicy for ProfileGuidedTuning {}

/// Static tuning context carrying backend and scope in its type.
///
/// `TuningContext<Dyn, Dyn>` is the matching runtime-selected form.
pub struct TuningContext<D = Dyn, S = Dyn> {
    scope: TuningScope,
    environment: TuningEnvironmentFingerprint<D>,
    topology: Option<TuningTopologyFingerprint<Dyn>>,
    determinism: Determinism,
    memory_budget: usize,
    time_budget: Duration,
    scope_marker: PhantomData<fn() -> S>,
}

impl<D: StaticBackend> TuningContext<D, KernelTuning> {
    /// Constructs a statically kernel-scoped context. A topology cannot be
    /// supplied to this spelling.
    pub fn kernel(
        environment: TuningEnvironmentFingerprint<D>,
        determinism: Determinism,
        memory_budget: usize,
        time_budget: Duration,
    ) -> core::result::Result<Self, TuningServiceError> {
        validate_time_budget(time_budget)?;
        Ok(Self {
            scope: TuningScope::Kernel,
            environment,
            topology: None,
            determinism,
            memory_budget,
            time_budget,
            scope_marker: PhantomData,
        })
    }
}

impl<D: StaticBackend> TuningContext<D, CollectiveTuning> {
    /// Constructs a statically collective-scoped context, requiring a checked
    /// topology.
    pub fn collective<W>(
        environment: TuningEnvironmentFingerprint<D>,
        topology: TuningTopologyFingerprint<W>,
        determinism: Determinism,
        memory_budget: usize,
        time_budget: Duration,
    ) -> core::result::Result<Self, TuningServiceError> {
        validate_time_budget(time_budget)?;
        Ok(Self {
            scope: TuningScope::Collective,
            environment,
            topology: Some(topology.erase()),
            determinism,
            memory_budget,
            time_budget,
            scope_marker: PhantomData,
        })
    }
}

impl<D: StaticBackend> TuningContext<D, ExecutionPlanTuning> {
    /// Constructs a statically execution-plan-scoped context, requiring a
    /// checked topology.
    pub fn execution_plan<W>(
        environment: TuningEnvironmentFingerprint<D>,
        topology: TuningTopologyFingerprint<W>,
        determinism: Determinism,
        memory_budget: usize,
        time_budget: Duration,
    ) -> core::result::Result<Self, TuningServiceError> {
        validate_time_budget(time_budget)?;
        Ok(Self {
            scope: TuningScope::ExecutionPlan,
            environment,
            topology: Some(topology.erase()),
            determinism,
            memory_budget,
            time_budget,
            scope_marker: PhantomData,
        })
    }
}

impl TuningContext<Dyn, Dyn> {
    /// Constructs the runtime-selected form and checks the topology
    /// requirement corresponding to `scope`.
    pub fn new_dyn(
        scope: TuningScope,
        environment: TuningEnvironmentFingerprint<Dyn>,
        topology: Option<TuningTopologyFingerprint<Dyn>>,
        determinism: Determinism,
        memory_budget: usize,
        time_budget: Duration,
    ) -> core::result::Result<Self, TuningServiceError> {
        validate_time_budget(time_budget)?;
        if scope.requires_topology() && topology.is_none() {
            return Err(TuningServiceError::TopologyRequired { scope });
        }
        if !scope.requires_topology() && topology.is_some() {
            return Err(TuningServiceError::UnexpectedTopology { scope });
        }
        Ok(Self {
            scope,
            environment,
            topology,
            determinism,
            memory_budget,
            time_budget,
            scope_marker: PhantomData,
        })
    }
}

impl<D, S> TuningContext<D, S> {
    /// Runtime scope projection.
    #[must_use]
    pub const fn scope(&self) -> TuningScope {
        self.scope
    }

    /// Device/compiler environment.
    #[must_use]
    pub const fn environment(&self) -> &TuningEnvironmentFingerprint<D> {
        &self.environment
    }

    /// Exact topology for distributed scopes.
    #[must_use]
    pub const fn topology(&self) -> Option<&TuningTopologyFingerprint<Dyn>> {
        self.topology.as_ref()
    }

    /// Determinism requirement used to filter candidates.
    #[must_use]
    pub const fn determinism(&self) -> Determinism {
        self.determinism
    }

    /// Hard transient-memory budget.
    #[must_use]
    pub const fn memory_budget(&self) -> usize {
        self.memory_budget
    }

    /// Hard operation-local wait/measurement budget.
    #[must_use]
    pub const fn time_budget(&self) -> Duration {
        self.time_budget
    }
}

impl<D: StaticBackend, S: StaticTuningScope> TuningContext<D, S> {
    /// Erases the static backend and scope markers after retaining their
    /// checked runtime projections.
    #[must_use]
    pub fn erase(self) -> TuningContext<Dyn, Dyn> {
        TuningContext {
            scope: self.scope,
            environment: self.environment.erase(),
            topology: self.topology,
            determinism: self.determinism,
            memory_budget: self.memory_budget,
            time_budget: self.time_budget,
            scope_marker: PhantomData,
        }
    }
}

impl<D, S> fmt::Debug for TuningContext<D, S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TuningContext")
            .field("scope", &self.scope)
            .field("environment", &self.environment)
            .field("topology", &self.topology)
            .field("determinism", &self.determinism)
            .field("memory_budget", &self.memory_budget)
            .field("time_budget", &self.time_budget)
            .finish()
    }
}

/// Candidate metadata the general service can enforce without understanding
/// the backend-specific payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningCandidate {
    hash: u64,
    encoding: String,
    deterministic: bool,
    workspace_bytes: usize,
}

impl TuningCandidate {
    /// Constructs a candidate with a stable backend-owned hash and encoding.
    pub fn new(
        hash: u64,
        encoding: &str,
        deterministic: bool,
        workspace_bytes: usize,
    ) -> core::result::Result<Self, TuningServiceError> {
        if encoding.is_empty()
            || encoding.len() > MAX_CANDIDATE_ENCODING_BYTES
            || encoding.trim() != encoding
            || encoding.chars().any(char::is_control)
        {
            return Err(TuningServiceError::InvalidCandidateEncoding);
        }
        Ok(Self {
            hash,
            encoding: encoding.to_string(),
            deterministic,
            workspace_bytes,
        })
    }

    /// Stable backend-owned candidate hash.
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Persistent encoding which the backend must parse and revalidate.
    #[must_use]
    pub fn encoding(&self) -> &str {
        &self.encoding
    }

    /// Whether this candidate satisfies required determinism.
    #[must_use]
    pub const fn deterministic(&self) -> bool {
        self.deterministic
    }

    /// Transient workspace needed while running the candidate.
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
}

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
    candidate: TuningCandidate,
    source: SelectionSource,
    median_ns: Option<u64>,
    sample_count: u32,
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
    #[must_use]
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
    rank: usize,
    epoch: u64,
    candidate_hash: u64,
    accepted: bool,
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

fn validate_time_budget(budget: Duration) -> core::result::Result<(), TuningServiceError> {
    if budget.is_zero() {
        Err(TuningServiceError::ZeroTimeBudget)
    } else {
        Ok(())
    }
}

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
        limits: CacheLimits,
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
        profile_limits: CacheLimits,
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

/// Stable digest of a canonical legal candidate set.
#[must_use]
pub fn legal_candidates_digest(candidates: &[TuningCandidate]) -> u64 {
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|left, right| (left.hash, &left.encoding).cmp(&(right.hash, &right.encoding)));
    let mut digest = Digest::new().field(b"incin.tuning.legal-candidates.v1");
    for candidate in ordered {
        digest = digest
            .number(candidate.hash)
            .text(&candidate.encoding)
            .number(u64::from(candidate.deterministic))
            .number(candidate.workspace_bytes as u64);
    }
    digest.finish()
}

fn selection_from_record(
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

#[derive(Clone, Copy)]
struct Digest(u64);

impl Digest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    fn field(self, bytes: &[u8]) -> Self {
        self.number(bytes.len() as u64).bytes(bytes)
    }

    fn text(self, value: &str) -> Self {
        self.field(value.as_bytes())
    }

    fn number(self, value: u64) -> Self {
        self.bytes(&value.to_le_bytes())
    }

    const fn finish(self) -> u64 {
        self.0
    }
}
