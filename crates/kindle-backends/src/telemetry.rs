//! Optional telemetry integration for `kindle-backends`.
//!
//! When the `telemetry` feature is enabled, a process-global `Emitter` can be
//! installed via [`set_emitter`].  The autograd tape hooks in
//! `cpu::tape` and `wgpu::tape` call [`emit_scalar`] and [`emit_graph_snapshot`]
//! at key points without taking any locks on the hot path when no emitter is
//! installed.
//!
//! Design:
//! * The emitter is stored in a `static` `OnceLock<Emitter>` so it can be set
//!   once at program startup and read cheaply (pointer deref) thereafter.
//! * `emit_*` functions are no-ops when the `telemetry` feature is disabled,
//!   allowing the rest of the codebase to call them unconditionally inside
//!   `#[cfg(feature = "telemetry")]` guards.

#[cfg(feature = "telemetry")]
pub use kindle_telemetry::prelude::{CURRENT_SCHEMA_VERSION, Emitter, MemoryEvent, ScalarEvent};
#[cfg(feature = "telemetry")]
use kindle_telemetry::reporter::Reporter;

#[cfg(feature = "telemetry")]
use std::sync::OnceLock;

#[cfg(feature = "telemetry")]
static GLOBAL_EMITTER: OnceLock<Emitter> = OnceLock::new();

/// Install the process-global `Emitter`.  Should be called once at program
/// startup, before any training loop begins.  Subsequent calls are silently
/// ignored (OnceLock semantics).
#[cfg(feature = "telemetry")]
pub fn set_emitter(emitter: Emitter) {
    let _ = GLOBAL_EMITTER.set(emitter);
}

/// Emit a scalar telemetry event if an emitter has been installed.
/// No-op when the `telemetry` feature is disabled.
#[cfg(feature = "telemetry")]
pub fn emit_scalar(step: usize, name: &str, value: f64) {
    if let Some(emitter) = GLOBAL_EMITTER.get() {
        emitter.log_scalar(ScalarEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            name: name.to_string(),
            value,
        });
    }
}

/// Emit a memory event (RSS bytes) if an emitter has been installed.
#[cfg(feature = "telemetry")]
pub fn emit_memory(step: usize, rss_bytes: u64) {
    if let Some(emitter) = GLOBAL_EMITTER.get() {
        emitter.log_memory(MemoryEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            rss_bytes,
        });
    }
}

/// Emit a computation graph snapshot if an emitter has been installed.
#[cfg(feature = "telemetry")]
pub fn emit_graph_snapshot(graph: kindle_core::prelude::Graph) {
    use kindle_telemetry::prelude::*;
    if let Some(emitter) = GLOBAL_EMITTER.get() {
        emitter.log_graph_snapshot(GraphSnapshotEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            graph,
        });
    }
}
