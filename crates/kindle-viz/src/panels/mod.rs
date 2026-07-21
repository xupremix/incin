//! Built-in panels proving the plugin pipeline end-to-end (Plan 08-05):
//! a real loss `Chart` panel and a deliberate panic-test panel. Both
//! implement `kindle_viz_plugin_api::panel::Panel` with zero access to any
//! kindle-viz-internal-only API (PLUGIN-01/PLUGIN-02's "no privileged API"
//! property).

/// Core abstraction for `graph` within the Kindle framework.
pub mod graph;
/// Core abstraction for `loss` within the Kindle framework.
pub mod loss;
/// Core abstraction for `norms` within the Kindle framework.
pub mod norms;
/// Core abstraction for `panic_test` within the Kindle framework.
pub mod panic_test;
/// Core abstraction for `scalar` within the Kindle framework.
pub mod scalar;
/// Core abstraction for `system` within the Kindle framework.
pub mod system;
