//! Reserved types for unavailable preview plan tuning.
//!
//! The CPU compiled path has no measurement hook or tuning executor. It never
//! produces a tuning report and returns [`TuningUnavailable`] instead.

use crate::compiled::CompiledPlan;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningUnavailable;

impl core::fmt::Display for TuningUnavailable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("compiled tuning requires a real executor measurement hook")
    }
}

/// Reserved report shape for a future measured tuner; no report is produced today.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanTuningReport {
    pub baseline_latency_us: f64,
    pub tuned_latency_us: f64,
    pub speedup_ratio: f64,
    pub iterations_evaluated: usize,
    pub is_bounded: bool,
}

/// Placeholder interface for a future bounded, measured plan tuner.
#[derive(Debug, Clone)]
pub struct BoundedPlanTuner {
    pub max_iterations: usize,
    pub min_speedup_target: f64,
}

impl BoundedPlanTuner {
    pub fn new(max_iterations: usize, min_speedup_target: f64) -> Self {
        Self {
            max_iterations,
            min_speedup_target,
        }
    }

    /// Returns [`TuningUnavailable`]; no tuning or proxy measurement is performed.
    pub fn tune_plan(&self, _plan: &CompiledPlan) -> Result<PlanTuningReport, TuningUnavailable> {
        let _ = (self.max_iterations, self.min_speedup_target);
        Err(TuningUnavailable)
    }
}
