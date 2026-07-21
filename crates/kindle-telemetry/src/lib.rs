//! `kindle-telemetry`: the UI-dependency-free wire protocol for training
//! observability. Defines schema-versioned event types (`events`) and the
//! `Reporter` trait contract (`reporter`) that a training-process emitter
//! implements against. `emitter` provides the non-blocking, dual-channel
//! `Reporter` implementation; `transport` provides the I/O sinks it drains
//! into.
#[macro_use]
extern crate alloc;

/// The `emitter` module.
pub mod emitter;
/// Auto-generated documentation for err.
pub mod err;
/// Auto-generated documentation for events.
pub mod events;
/// Auto-generated documentation for reporter.
pub mod reporter;
/// Auto-generated documentation for run_dir.
pub mod run_dir;
/// Auto-generated documentation for transport.
pub mod transport;

/// Auto-generated documentation for prelude.
pub mod prelude {
    pub use crate::emitter::Emitter;
    pub use crate::events::*;
    pub use crate::reporter::Reporter;
}
