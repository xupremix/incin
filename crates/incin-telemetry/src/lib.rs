//! `incin-telemetry`: the UI-dependency-free wire protocol for training
//! observability. Defines schema-versioned event types (`events`) and the
//! `Reporter` trait contract (`reporter`) that a training-process emitter
//! implements against. `emitter` provides the non-blocking, dual-channel
//! `Reporter` implementation; `transport` provides the I/O sinks it drains
//! into.
#[macro_use]
extern crate alloc;

/// Asynchronous event emitter and transport fan-out.
pub mod emitter;
/// Error types and result alias for telemetry operations.
pub mod err;
/// Schema-versioned telemetry events (scalars, norms, memory, graph).
pub mod events;
/// Trait contract for telemetry reporters.
pub mod reporter;
/// Utilities for creating structured run directories.
pub mod run_dir;
/// File and IPC transport sinks for telemetry events.
pub mod transport;

/// Convenient re-exports for telemetry usage.
pub mod prelude {
    pub use crate::emitter::Emitter;
    pub use crate::events::{
        CURRENT_SCHEMA_VERSION, EpochEvent, Event, GradientNormEvent, GraphSnapshotEvent,
        HyperparamEvent, MemoryEvent, ScalarEvent, WeightNormEvent,
    };
    pub use crate::reporter::Reporter;
}
