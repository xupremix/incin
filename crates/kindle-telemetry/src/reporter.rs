//! The `Reporter` trait: a pure, signature-only contract for emitting
//! telemetry events. No concrete implementation lives in this crate — a
//! non-blocking, bounded-channel emitter is Phase 7 scope. This trait exists
//! only to define what "reporting an event" means, deliberately avoiding the
//! retired prototype's `Watcher` anti-pattern (a lock-guarded state holder
//! performing synchronous I/O inline with training).

use crate::events::{
    EpochEvent, GradientNormEvent, GraphSnapshotEvent, HyperparamEvent, MemoryEvent, ScalarEvent,
    WeightNormEvent,
};

/// Fire-and-forget telemetry sink. One method per wire event type in
/// `events`. Implementors decide how (and whether) events are buffered,
/// batched, or transported; this trait only fixes the call surface a
/// training loop reports against.
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
}
