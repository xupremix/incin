//! Built-in panels proving the plugin pipeline end-to-end (Plan 08-05):
//! a real loss `Chart` panel and a deliberate panic-test panel. Both
//! implement `kindle_viz_plugin_api::panel::Panel` with zero access to any
//! kindle-viz-internal-only API (PLUGIN-01/PLUGIN-02's "no privileged API"
//! property).

/// Graph.
pub mod graph;
/// Loss.
pub mod loss;
/// Norms.
pub mod norms;
/// Panic test.
pub mod panic_test;
/// Scalar.
pub mod scalar;
/// System.
pub mod system;
