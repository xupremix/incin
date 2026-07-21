//! `kindle-viz` — out-of-process terminal UI for observing live Kindle
//! training runs. Tails an out-of-process telemetry transport (see
//! `kindle-telemetry`) and renders it through a plugin-extensible panel
//! system (see `kindle-viz-plugin-api`).
#[macro_use]
extern crate alloc;

/// Core abstraction for `app` within the Kindle framework.
pub mod app;
/// Core abstraction for `dispatch` within the Kindle framework.
pub mod dispatch;
/// Core abstraction for `err` within the Kindle framework.
pub mod err;
/// Core abstraction for `panels` within the Kindle framework.
pub mod panels;
/// Core abstraction for `transport_reader` within the Kindle framework.
pub mod transport_reader;
