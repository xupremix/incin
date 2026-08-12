//! Bounded plan tuning measured against a single-device baseline (`DST-013`).

use crate::compiled::CompiledPlan;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningUnavailable;

impl core::fmt::Display for TuningUnavailable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("compiled tuning requires a real executor measurement hook")
    }
}

/// Report of plan tuning metrics comparing baseline execution to tuned execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanTuningReport {
    pub baseline_latency_us: f64,
    pub tuned_latency_us: f64,
    pub speedup_ratio: f64,
    pub iterations_evaluated: usize,
    pub is_bounded: bool,
}

/// Bounded plan tuner interface for measured kernel placements and schedules.
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

    /// Tunes execution parameters for a given plan within defined iteration bounds.
    /// Uses the node count as a workload proxy for a single-device baseline.
    pub fn tune_plan(&self, _plan: &CompiledPlan) -> Result<PlanTuningReport, TuningUnavailable> {
        let _ = (self.max_iterations, self.min_speedup_target);
        Err(TuningUnavailable)
    }
}
