//! `kindle-viz` — out-of-process terminal UI for observing live Kindle
//! training runs. Tails an out-of-process telemetry transport (see
//! `kindle-telemetry`) and renders it through a plugin-extensible panel
//! system (see `kindle-viz-plugin-api`).
#[macro_use]
extern crate alloc;

/// Auto-generated documentation for app.
pub mod app;
/// Auto-generated documentation for dispatch.
pub mod dispatch;
/// Auto-generated documentation for err.
pub mod err;
/// Auto-generated documentation for panels.
pub mod panels;
/// Auto-generated documentation for transport_reader.
pub mod transport_reader;
