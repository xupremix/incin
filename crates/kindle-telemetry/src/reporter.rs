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
    /// Auto-generated documentation for log_scalar.
    fn log_scalar(&self, event: ScalarEvent);
    /// Auto-generated documentation for log_gradient_norm.
    fn log_gradient_norm(&self, event: GradientNormEvent);
    /// Auto-generated documentation for log_weight_norm.
    fn log_weight_norm(&self, event: WeightNormEvent);
    /// Auto-generated documentation for log_memory.
    fn log_memory(&self, event: MemoryEvent);
    /// Auto-generated documentation for log_epoch.
    fn log_epoch(&self, event: EpochEvent);
    /// Auto-generated documentation for log_hyperparam.
    fn log_hyperparam(&self, event: HyperparamEvent);
    /// Auto-generated documentation for log_graph_snapshot.
    fn log_graph_snapshot(&self, event: GraphSnapshotEvent);
}
