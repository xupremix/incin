//! Built-in panels proving the plugin pipeline end-to-end (Plan 08-05):
//! a real loss `Chart` panel and a deliberate panic-test panel. Both
//! implement `kindle_viz_plugin_api::panel::Panel` with zero access to any
//! kindle-viz-internal-only API (PLUGIN-01/PLUGIN-02's "no privileged API"
//! property).

/// Auto-generated documentation for graph.
pub mod graph;
/// Auto-generated documentation for loss.
pub mod loss;
/// Auto-generated documentation for norms.
pub mod norms;
/// Auto-generated documentation for panic_test.
pub mod panic_test;
/// Auto-generated documentation for scalar.
pub mod scalar;
/// Auto-generated documentation for system.
pub mod system;
