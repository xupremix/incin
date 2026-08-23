//! Reproducibility manifest replay and incompatibility diffs (`UX-008`).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Reproducibility manifest capturing execution environmental parameters, static graph hashes, and seed configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReproducibilityManifest {
    /// Framework version that produced this plan.
    pub incin_version: String,
    /// Determinism seed recorded for replay.
    pub seed: u64,
    /// Precision policy as text.
    pub precision: String,
    /// Mesh topology description.
    pub mesh_spec: String,
    /// Tuning environment fingerprint as text.
    pub environment: String,
    /// Hash binding this manifest to one plan.
    pub plan_hash: String,
}

impl ReproducibilityManifest {
    /// Records every reproducibility field for one plan.
    pub fn new(
        seed: u64,
        precision: impl Into<String>,
        mesh_spec: impl Into<String>,
        environment: impl Into<String>,
        plan_hash: impl Into<String>,
    ) -> Self {
        Self {
            incin_version: env!("CARGO_PKG_VERSION").into(),
            seed,
            precision: precision.into(),
            mesh_spec: mesh_spec.into(),
            environment: environment.into(),
            plan_hash: plan_hash.into(),
        }
    }

    #[cfg(feature = "serde_json")]
    /// Serializes manifest to JSON format.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    #[cfg(feature = "serde_json")]
    /// Deserializes manifest from JSON format.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Compares `self` against `current` and returns a list of incompatibility diff strings.
    pub fn replay_diff(&self, current: &Self) -> Vec<String> {
        let mut diffs = Vec::new();

        if self.incin_version != current.incin_version {
            diffs.push(format!(
                "Incompatible Incin version: manifest={}, current={}",
                self.incin_version, current.incin_version
            ));
        }
        if self.seed != current.seed {
            diffs.push(format!(
                "Mismatched random seed: manifest={}, current={}",
                self.seed, current.seed
            ));
        }
        if self.precision != current.precision {
            diffs.push(format!(
                "Mismatched precision policy: manifest={}, current={}",
                self.precision, current.precision
            ));
        }
        if self.mesh_spec != current.mesh_spec {
            diffs.push(format!(
                "Mismatched mesh topology: manifest={}, current={}",
                self.mesh_spec, current.mesh_spec
            ));
        }
        if self.plan_hash != current.plan_hash {
            diffs.push(format!(
                "Mismatched plan hash: manifest={}, current={}",
                self.plan_hash, current.plan_hash
            ));
        }

        diffs
    }
}
