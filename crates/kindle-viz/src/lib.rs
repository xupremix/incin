//! `kindle-viz` — out-of-process terminal UI for observing live Kindle
//! training runs. Tails an out-of-process telemetry transport (see
//! `kindle-telemetry`) and renders it through a plugin-extensible panel
//! system (see `kindle-viz-plugin-api`).
#[macro_use]
extern crate alloc;

pub mod app;
pub mod dispatch;
pub mod err;
pub mod panels;
pub mod transport_reader;
