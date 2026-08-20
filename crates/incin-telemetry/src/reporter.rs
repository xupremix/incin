//! The `Reporter` trait: the call surface a training loop reports telemetry
//! against. [`crate::emitter::Emitter`] is this crate's concrete,
//! non-blocking implementation (see its own docs); this trait exists so
//! `incin-viz`/tests/future implementations only need to depend on the
//! contract, not `Emitter` itself, deliberately avoiding the retired
//! prototype's `Watcher` anti-pattern (a lock-guarded state holder
//! performing synchronous I/O inline with training).

use crate::events::{
    CURRENT_SCHEMA_VERSION, EpochEvent, GradientNormEvent, GraphSnapshotEvent, HyperparamEvent,
    MemoryEvent, ScalarEvent, WeightNormEvent,
};
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};

/// Fire-and-forget telemetry sink. One method per wire event type in
/// `events`. Implementors decide how (and whether) events are buffered,
/// batched, or transported; this trait only fixes the call surface a
/// training loop reports against.
///
/// The `log_*` methods below are the required, schema-precise call surface;
/// the unprefixed convenience methods (`scalar`, `gradient_norm`, ...) are
/// default-provided wrappers that fill in `schema_version` and build the
/// event struct for you, so a training loop can write
/// `reporter.scalar("loss", step, value)` instead of hand-constructing a
/// `ScalarEvent` - this is `docs/growth/05-observability-and-scaffolding.md`
/// Task 05.1's ergonomic entry point.
pub trait Reporter {
    /// Emit a scalar metric event.
    fn log_scalar(&self, event: ScalarEvent);
    /// Emit a gradient norm event.
    fn log_gradient_norm(&self, event: GradientNormEvent);
    /// Emit a weight norm event.
    fn log_weight_norm(&self, event: WeightNormEvent);
    /// Emit a memory usage event.
    fn log_memory(&self, event: MemoryEvent);
    /// Emit an epoch metrics event.
    fn log_epoch(&self, event: EpochEvent);
    /// Emit a hyperparameter configuration event.
    fn log_hyperparam(&self, event: HyperparamEvent);
    /// Emit a graph snapshot event.
    fn log_graph_snapshot(&self, event: GraphSnapshotEvent);

    /// Reports a named scalar sample (e.g. `"loss"`, `"learning_rate"`) at
    /// `step`. Convenience wrapper over [`Reporter::log_scalar`].
    fn scalar(&self, name: &str, step: usize, value: f64) {
        self.log_scalar(ScalarEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            name: name.to_string(),
            value,
        });
    }

    /// Reports one parameter's gradient L2-norm at `step`. Convenience
    /// wrapper over [`Reporter::log_gradient_norm`].
    fn gradient_norm(&self, step: usize, param_name: &str, l2_norm: f32) {
        self.log_gradient_norm(GradientNormEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            param_name: param_name.to_string(),
            l2_norm,
        });
    }

    /// Reports one parameter's weight L2-norm at `step`. Convenience
    /// wrapper over [`Reporter::log_weight_norm`].
    fn weight_norm(&self, step: usize, param_name: &str, l2_norm: f32) {
        self.log_weight_norm(WeightNormEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            param_name: param_name.to_string(),
            l2_norm,
        });
    }

    /// Reports resident-set-size memory usage at `step`. Convenience
    /// wrapper over [`Reporter::log_memory`].
    fn memory(&self, step: usize, rss_bytes: u64) {
        self.log_memory(MemoryEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            rss_bytes,
        });
    }

    /// Reports epoch-level aggregate metrics. Convenience wrapper over
    /// [`Reporter::log_epoch`].
    fn epoch(&self, epoch: usize, metrics: BTreeMap<String, f32>) {
        self.log_epoch(EpochEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            epoch,
            metrics,
        });
    }

    /// Reports a static hyperparameter/config snapshot (see
    /// [`HyperparamEvent`]'s doc comment on secrets before logging
    /// arbitrary strings here). Convenience wrapper over
    /// [`Reporter::log_hyperparam`].
    fn hyperparam(&self, params: BTreeMap<String, String>) {
        self.log_hyperparam(HyperparamEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            params,
        });
    }

    /// Reports a traced computation graph snapshot. Convenience wrapper
    /// over [`Reporter::log_graph_snapshot`].
    fn graph_snapshot(&self, graph: incin_core::graph::Graph) {
        self.log_graph_snapshot(GraphSnapshotEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            graph,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Spies on every `log_*` call by recording the event it was given,
    /// tagged with which method delivered it -- lets these tests assert
    /// that the ergonomic wrappers build the exact event a caller would
    /// have hand-constructed, without needing a real transport/emitter.
    #[derive(Default)]
    struct SpyReporter {
        scalar: Mutex<Option<ScalarEvent>>,
        gradient_norm: Mutex<Option<GradientNormEvent>>,
        weight_norm: Mutex<Option<WeightNormEvent>>,
        memory: Mutex<Option<MemoryEvent>>,
        epoch: Mutex<Option<EpochEvent>>,
        hyperparam: Mutex<Option<HyperparamEvent>>,
    }

    impl Reporter for SpyReporter {
        fn log_scalar(&self, event: ScalarEvent) {
            *self.scalar.lock().unwrap() = Some(event);
        }
        fn log_gradient_norm(&self, event: GradientNormEvent) {
            *self.gradient_norm.lock().unwrap() = Some(event);
        }
        fn log_weight_norm(&self, event: WeightNormEvent) {
            *self.weight_norm.lock().unwrap() = Some(event);
        }
        fn log_memory(&self, event: MemoryEvent) {
            *self.memory.lock().unwrap() = Some(event);
        }
        fn log_epoch(&self, event: EpochEvent) {
            *self.epoch.lock().unwrap() = Some(event);
        }
        fn log_hyperparam(&self, event: HyperparamEvent) {
            *self.hyperparam.lock().unwrap() = Some(event);
        }
        fn log_graph_snapshot(&self, _event: GraphSnapshotEvent) {}
    }

    #[test]
    fn scalar_wrapper_builds_the_expected_event() {
        let rep = SpyReporter::default();
        rep.scalar("loss", 7, 0.25);
        let event = rep
            .scalar
            .lock()
            .unwrap()
            .clone()
            .expect("log_scalar called");
        assert_eq!(event.schema_version, CURRENT_SCHEMA_VERSION);
        assert_eq!(event.step, 7);
        assert_eq!(event.name, "loss");
        assert_eq!(event.value, 0.25);
    }

    #[test]
    fn gradient_norm_wrapper_builds_the_expected_event() {
        let rep = SpyReporter::default();
        rep.gradient_norm(3, "fc1.weight", 1.5);
        let event = rep
            .gradient_norm
            .lock()
            .unwrap()
            .clone()
            .expect("log_gradient_norm called");
        assert_eq!(event.step, 3);
        assert_eq!(event.param_name, "fc1.weight");
        assert_eq!(event.l2_norm, 1.5);
    }

    #[test]
    fn weight_norm_wrapper_builds_the_expected_event() {
        let rep = SpyReporter::default();
        rep.weight_norm(3, "fc1.weight", 2.5);
        let event = rep
            .weight_norm
            .lock()
            .unwrap()
            .clone()
            .expect("log_weight_norm called");
        assert_eq!(event.step, 3);
        assert_eq!(event.param_name, "fc1.weight");
        assert_eq!(event.l2_norm, 2.5);
    }

    #[test]
    fn memory_wrapper_builds_the_expected_event() {
        let rep = SpyReporter::default();
        rep.memory(1, 2048);
        let event = rep
            .memory
            .lock()
            .unwrap()
            .clone()
            .expect("log_memory called");
        assert_eq!(event.step, 1);
        assert_eq!(event.rss_bytes, 2048);
    }

    #[test]
    fn epoch_wrapper_builds_the_expected_event() {
        let rep = SpyReporter::default();
        let mut metrics = BTreeMap::new();
        metrics.insert("accuracy".to_string(), 0.9);
        rep.epoch(2, metrics.clone());
        let event = rep.epoch.lock().unwrap().clone().expect("log_epoch called");
        assert_eq!(event.epoch, 2);
        assert_eq!(event.metrics, metrics);
    }

    #[test]
    fn hyperparam_wrapper_builds_the_expected_event() {
        let rep = SpyReporter::default();
        let mut params = BTreeMap::new();
        params.insert("lr".to_string(), "0.001".to_string());
        rep.hyperparam(params.clone());
        let event = rep
            .hyperparam
            .lock()
            .unwrap()
            .clone()
            .expect("log_hyperparam called");
        assert_eq!(event.params, params);
    }
}
