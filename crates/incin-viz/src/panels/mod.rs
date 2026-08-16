//! Built-in panels proving the plugin pipeline end-to-end (Plan 08-05):
//! a real loss `Chart` panel and a deliberate panic-test panel. Both
//! implement `incin_viz_plugin_api::panel::Panel` with zero access to any
//! incin-viz-internal-only API (PLUGIN-01/PLUGIN-02's "no privileged API"
//! property).

/// Graph snapshot and structure panel.
pub mod graph;
/// Scalar loss history panel.
pub mod loss;
/// Gradient and weight norm panel.
pub mod norms;
/// Diagnostic panel for displaying panic events.
pub mod panic_test;
/// Generic scalar-series panel.
pub mod scalar;
/// Runtime and process resource panel.
pub mod system;
