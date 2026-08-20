use super::error::TuningServiceError;
use crate::tuning::identity::{
    StaticBackend, TuningEnvironmentFingerprint, TuningTopologyFingerprint,
};
use core::{fmt, marker::PhantomData, time::Duration};
use incin_core::{exec::Determinism, prelude::Dyn};
use std::path::PathBuf;

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
    pub(super) scope: TuningScope,
    pub(super) environment: TuningEnvironmentFingerprint<D>,
    pub(super) topology: Option<TuningTopologyFingerprint<Dyn>>,
    pub(super) determinism: Determinism,
    pub(super) memory_budget: usize,
    pub(super) time_budget: Duration,
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

fn validate_time_budget(budget: Duration) -> core::result::Result<(), TuningServiceError> {
    if budget.is_zero() {
        Err(TuningServiceError::ZeroTimeBudget)
    } else {
        Ok(())
    }
}
