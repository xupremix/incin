//! `kindle-telemetry`: the UI-dependency-free wire protocol for training
//! observability. Defines schema-versioned event types (`events`) and the
//! `Reporter` trait contract (`reporter`) that a training-process emitter
//! implements against. `emitter` provides the non-blocking, dual-channel
//! `Reporter` implementation; `transport` provides the I/O sinks it drains
//! into.
#[macro_use]
extern crate alloc;

pub mod emitter;
pub mod err;
pub mod events;
pub mod reporter;
pub mod run_dir;
pub mod transport;

pub mod prelude {
    pub use crate::emitter::Emitter;
    pub use crate::events::*;
    pub use crate::reporter::Reporter;
}
