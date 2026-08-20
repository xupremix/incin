//! Policy-aware tuning orchestration.
//!
//! The service never benchmarks by itself. It filters a caller-supplied legal
//! candidate set, chooses deterministic fallback/heuristic results, issues a
//! bounded single-flight permit for coordinated measurement, and commits only
//! a complete local or all-participant result. Profile imports and warmup
//! results pass through the same persistent-cache legality checks.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `context` is the static/dynamic
//! scope and policy vocabulary plus `TuningContext` itself; `candidate` is
//! candidate description (`TuningCandidate`, its stable digest); `decision`
//! is the outcome vocabulary (`TuningSelection`, `ServiceDecision`,
//! `CoordinatedVote`); `error` is `TuningServiceError`, named by every other
//! file; `engine` is the stateful orchestration engine (`TuningService`,
//! its internal lease/result bookkeeping, and `TuningPermit`) kept as one
//! file rather than split further, since `TuningService::decide` and
//! `TuningPermit::finish` are two halves of one lease protocol sharing
//! several private types no other file needs to see.

mod candidate;
mod context;
mod decision;
mod engine;
mod error;

pub use candidate::{TuningCandidate, legal_candidates_digest};
pub use context::{
    AutotunePolicy, CollectiveTuning, CoordinatedWarmupTuning, DisabledTuning, ExecutionPlanTuning,
    HeuristicTuning, KernelTuning, ProfileGuidedTuning, StaticAutotunePolicy, StaticTuningScope,
    TuningContext, TuningScope,
};
pub use decision::{CoordinatedVote, SelectionSource, ServiceDecision, TuningSelection};
pub use engine::{TuningPermit, TuningService};
pub use error::TuningServiceError;
