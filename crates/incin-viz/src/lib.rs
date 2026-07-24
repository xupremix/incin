//! `incin-viz` — out-of-process terminal UI for observing live Incin
//! training runs. Tails an out-of-process telemetry transport (see
//! `incin-telemetry`) and renders it through a plugin-extensible panel
//! system (see `incin-viz-plugin-api`).
#[macro_use]
extern crate alloc;

/// App.
pub mod app;
/// Dispatch.
pub mod dispatch;
/// Err.
pub mod err;
/// Panels.
pub mod panels;
/// Transport reader.
pub mod transport_reader;
