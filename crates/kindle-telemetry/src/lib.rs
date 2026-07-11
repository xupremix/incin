//! `kindle-telemetry`: the UI-dependency-free wire protocol for training
//! observability. Defines schema-versioned event types (`events`) and the
//! `Reporter` trait contract (`reporter`) that a training-process emitter
//! implements against. No concrete transport or emitter lives in this crate.

pub mod events;

pub mod prelude {
    pub use crate::events::*;
}
