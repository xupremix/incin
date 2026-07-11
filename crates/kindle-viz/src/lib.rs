//! `kindle-viz` — out-of-process terminal UI for observing live Kindle
//! training runs. Tails an out-of-process telemetry transport (see
//! `kindle-telemetry`) and renders it through a plugin-extensible panel
//! system (see `kindle-viz-plugin-api`).

pub mod dispatch;
pub mod err;
pub mod transport_reader;
