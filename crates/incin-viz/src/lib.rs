//! `incin-viz` — out-of-process terminal UI for observing live Incin
//! training runs. Tails an out-of-process telemetry transport (see
//! `incin-telemetry`) and renders it through a plugin-extensible panel
//! system (see `incin-viz-plugin-api`).
#[macro_use]
extern crate alloc;

/// Application state and event-loop integration for the visualizer.
pub mod app;
/// Dispatches telemetry data and user actions to visualizer panels.
pub mod dispatch;
/// Errors reported while loading or rendering visualizer data.
pub mod err;
/// Built-in panels for graph, loss, norm, scalar, and system views.
pub mod panels;
/// Reads schema-versioned telemetry events from a run transport.
pub mod transport_reader;
