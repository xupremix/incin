//! Built-in panels proving the plugin pipeline end-to-end (Plan 08-05):
//! a real loss `Chart` panel and a deliberate panic-test panel. Both
//! implement `kindle_viz_plugin_api::panel::Panel` with zero access to any
//! kindle-viz-internal-only API (PLUGIN-01/PLUGIN-02's "no privileged API"
//! property).

pub mod loss;
pub mod panic_test;
pub mod scalar;
pub mod norms;
pub mod system;
pub mod graph;
