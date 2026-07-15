//! Wire event schema for `kindle-telemetry`.
//!
//! These types are the serde-derived, schema-versioned payloads that a
//! training process emits (via a future `Reporter` implementation, see
//! `reporter.rs`) and that a separate `kindle-viz` process will eventually
//! deserialize over an out-of-process transport. No transport exists yet in
//! this phase — only the type schema.

/// Current wire schema version. Bump this whenever an event struct's shape
/// changes in a way that is not backward-compatible for readers.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// A single named scalar metric sample (loss, learning rate, throughput,
/// etc.), salvaging the retired prototype's `StepData` field shape
/// (`step`/`loss`/`lr`/`throughput`) as a generic named-metric event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScalarEvent {
    pub schema_version: u32,
    pub step: usize,
    pub name: String,
    pub value: f64,
}

/// Per-parameter gradient L2-norm sample, salvaging `StepData.gradients:
/// BTreeMap<String, f32>`'s per-param-name-to-norm shape and the retired
/// `log_step_with_grads`'s L2-norm algorithm (`sum_sq: f64 =
/// vec.iter().map(|&x| x*x).sum(); magnitude = sum_sq.sqrt() as f32`) as the
/// reference computation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GradientNormEvent {
    pub schema_version: u32,
    pub step: usize,
    pub param_name: String,
    pub l2_norm: f32,
}

/// Per-parameter weight L2-norm sample, same shape as `GradientNormEvent`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WeightNormEvent {
    pub schema_version: u32,
    pub step: usize,
    pub param_name: String,
    pub l2_norm: f32,
}

/// Process resident-set-size sample.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryEvent {
    pub schema_version: u32,
    pub step: usize,
    pub rss_bytes: u64,
}

/// Epoch-level aggregate metrics, salvaging `EpochData`'s exact
/// `epoch`/`metrics: BTreeMap<String, f32>` shape.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EpochEvent {
    pub schema_version: u32,
    pub epoch: usize,
    pub metrics: alloc::collections::BTreeMap<String, f32>,
}

/// Static hyperparameter/config snapshot, emitted once per run. Values are
/// string-typed to accommodate heterogeneous hyperparameter types without a
/// tagged-union payload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HyperparamEvent {
    pub schema_version: u32,
    pub params: alloc::collections::BTreeMap<String, String>,
}

/// A snapshot of the traced computation graph, wrapping the now-serializable
/// `Graph` IR from `kindle_core::graph`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphSnapshotEvent {
    pub schema_version: u32,
    pub graph: kindle_core::prelude::Graph,
}

/// Forward-compatible envelope wrapping every wire event variant.
///
/// The internally-tagged `type` representation below is load-bearing: the
/// catch-all fallback variant only works on internally- or
/// adjacently-tagged enums, not the default externally-tagged
/// representation. A reader on an older schema encountering a future,
/// unrecognized `type` tag deserializes to `Event::Unknown` instead of
/// failing outright.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum Event {
    Scalar(ScalarEvent),
    GradientNorm(GradientNormEvent),
    WeightNorm(WeightNormEvent),
    Memory(MemoryEvent),
    Epoch(EpochEvent),
    Hyperparam(HyperparamEvent),
    GraphSnapshot(GraphSnapshotEvent),
    #[serde(other)]
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_event_schema_version_round_trips_through_json() {
        let event = ScalarEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step: 42,
            name: "loss".to_string(),
            value: 0.1234,
        };

        let json = serde_json::to_string(&event).expect("serialize ScalarEvent");
        let round_tripped: ScalarEvent =
            serde_json::from_str(&json).expect("deserialize ScalarEvent");

        assert_eq!(round_tripped.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(round_tripped.step, event.step);
        assert_eq!(round_tripped.name, event.name);
        assert_eq!(round_tripped.value, event.value);
    }

    #[test]
    fn unrecognized_event_type_deserializes_to_unknown() {
        let future_event_json = r#"{"type":"SomeFutureEventType","schema_version":99,"foo":"bar"}"#;

        let event: Event =
            serde_json::from_str(future_event_json).expect("deserialize unknown event variant");

        assert!(matches!(event, Event::Unknown));
    }
}
