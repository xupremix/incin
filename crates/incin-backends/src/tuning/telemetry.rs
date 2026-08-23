//! Tuning telemetry, provenance, and explain formatting.

use alloc::format;
use alloc::string::{String, ToString};

use incin_core::shapes::Dyn;

use crate::tuning::cache::CacheKey;
use crate::tuning::identity::TuningEnvironmentFingerprint;
use crate::tuning::service::{AutotunePolicy, SelectionSource, TuningScope, TuningSelection};

/// Structured environment and candidate provenance for a tuning decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningProvenance {
    /// Cache key this record explains.
    pub key: CacheKey<Dyn>,
    /// Tuning scope the key belongs to.
    pub scope: TuningScope,
    /// Environment fingerprint at measurement time.
    pub environment: TuningEnvironmentFingerprint<Dyn>,
    /// Digest of the legal candidate set.
    pub candidate_digest: u64,
    /// Hash identifying the winning candidate.
    pub winner_hash: u64,
    /// Encoding format of the stored winner.
    pub winner_encoding: String,
    /// How the winner was chosen.
    pub source: SelectionSource,
    /// Median measured latency in nanoseconds.
    pub median_ns: Option<u64>,
    /// Number of samples behind the median.
    pub sample_count: u32,
}

impl TuningProvenance {
    /// Constructs provenance from key, scope, environment, candidate digest, and selection.
    pub fn new(
        key: CacheKey<Dyn>,
        scope: TuningScope,
        environment: TuningEnvironmentFingerprint<Dyn>,
        candidate_digest: u64,
        selection: &TuningSelection,
    ) -> Self {
        Self {
            key,
            scope,
            environment,
            candidate_digest,
            winner_hash: selection.candidate().hash(),
            winner_encoding: selection.candidate().encoding().to_string(),
            source: selection.source(),
            median_ns: selection.median_ns(),
            sample_count: selection.sample_count(),
        }
    }
}

/// Diagnostic explain formatter for human and machine consumption.
#[derive(Debug, Clone)]
pub struct TuningExplain {
    /// Provenance of this selection.
    pub provenance: TuningProvenance,
    /// Policy that governed tuning.
    pub policy: AutotunePolicy,
    /// Total candidates considered.
    pub total_candidates: usize,
    /// Candidates legal under pruning.
    pub legal_candidates: usize,
}

impl TuningExplain {
    /// Constructs a new explain helper.
    pub fn new(
        provenance: TuningProvenance,
        policy: AutotunePolicy,
        total_candidates: usize,
        legal_candidates: usize,
    ) -> Self {
        Self {
            provenance,
            policy,
            total_candidates,
            legal_candidates,
        }
    }

    /// Formats a human-readable text explanation of the tuning decision.
    pub fn explain_text(&self) -> String {
        let p = &self.provenance;
        let mut out = String::new();
        out.push_str(&format!("Tuning Explain [{:?}]\n", p.scope));
        out.push_str(&format!("  Problem Key:       {}\n", p.key.problem()));
        out.push_str(&format!("  Selection Source:  {:?}\n", p.source));
        out.push_str(&format!(
            "  Winner Candidate:  {} (hash={:#x})\n",
            p.winner_encoding, p.winner_hash
        ));
        if let Some(median) = p.median_ns {
            out.push_str(&format!(
                "  Measured Median:   {:.3} ms ({} samples)\n",
                median as f64 / 1_000_000.0,
                p.sample_count
            ));
        } else {
            out.push_str("  Measured Median:   None (unmeasured/fallback)\n");
        }
        out.push_str(&format!(
            "  Candidate Set:     {} legal / {} total (digest={:#x})\n",
            self.legal_candidates, self.total_candidates, p.candidate_digest
        ));
        out.push_str(&format!(
            "  Device Target:     {} ({})\n",
            p.environment.device().persistent_id(),
            p.environment.device().architecture()
        ));
        out.push_str(&format!(
            "  Compiler Target:   {} ({})\n",
            p.environment.compiler().implementation(),
            p.environment.compiler().target()
        ));
        out
    }

    /// Formats a structured JSON report of the tuning decision.
    pub fn explain_json(&self) -> String {
        let p = &self.provenance;
        let median_str = match p.median_ns {
            Some(m) => format!("{}", m),
            None => "null".to_string(),
        };
        format!(
            r#"{{"scope":"{:?}","problem":"{}","source":"{:?}","winner_hash":"{:#x}","winner_encoding":"{}","median_ns":{},"sample_count":{},"legal_candidates":{},"total_candidates":{},"candidate_digest":"{:#x}","device_id":"{}","architecture":"{}"}}"#,
            p.scope,
            p.key.problem(),
            p.source,
            p.winner_hash,
            p.winner_encoding,
            median_str,
            p.sample_count,
            self.legal_candidates,
            self.total_candidates,
            p.candidate_digest,
            p.environment.device().persistent_id(),
            p.environment.device().architecture(),
        )
    }
}

/// Emit scalar telemetry for a tuning decision if `telemetry` feature is enabled.
pub fn emit_tuning_telemetry(step: usize, provenance: &TuningProvenance) {
    #[cfg(feature = "telemetry")]
    {
        if let Some(median) = provenance.median_ns {
            crate::telemetry::emit_scalar(step, "tuning/median_ns", median as f64);
        }
        crate::telemetry::emit_scalar(step, "tuning/sample_count", provenance.sample_count as f64);
    }
    let _ = step;
    let _ = provenance;
}
