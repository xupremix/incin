//! `kindle-viz-plugin-api`: The stable, independently-compilable trait
//! surface for building custom visualizations and UI plugins for `kindle-viz`.
//!
//! This crate provides the foundational types and traits (`Panel`, `Plugin`, `KeymapProvider`,
//! and `RenderCtx`) needed to extend the Kindle Visualizer dashboard. By separating
//! the plugin API from the main `kindle-viz` crate, developers can build custom telemetry
//! visualizers without pulling in heavy TUI or application runtime dependencies.
//!
//! # Architecture
//!
//! The plugin architecture revolves around the [`Panel`] trait. A `Panel` is a stateful
//! struct that:
//! 1. Receives incoming `kindle-telemetry` [`Event`]s via `update()`.
//! 2. Renders itself to a bounded screen region using a [`RenderCtx`] via `render()`.
//! 3. Responds to keyboard and mouse inputs via `handle_event()`.
//!
//! The [`RenderCtx`] provides a safe, abstracted interface to the underlying rendering
//! engine (`ratatui`), preventing plugins from accidentally overwriting other panels
//! or crashing the host application by drawing outside their assigned boundaries.
//!
//! # Getting Started
//!
//! To create a custom panel, implement the [`Panel`] trait:
//!
//! ```rust,no_run
//! use kindle_viz_plugin_api::prelude::*;
//! use kindle_telemetry::events::{Event, ScalarEvent};
//!
//! pub struct MyCustomPanel {
//!     current_value: f64,
//! }
//!
//! impl Panel for MyCustomPanel {
//!     fn id(&self) -> &'static str { "my_custom_panel" }
//!     fn title(&self) -> &str { "Custom Metric" }
//!     
//!     fn update(&mut self, event: &Event) {
//!         if let Event::Scalar(ScalarEvent { value, .. }) = event {
//!             self.current_value = *value;
//!         }
//!     }
//!     
//!     fn render(&mut self, ctx: &mut RenderCtx<'_, '_>) {
//!         // Use ctx.buf() and ctx.area() to draw custom UI using ratatui
//!     }
//!     
//!     fn handle_event(&mut self, _event: &PanelEvent) -> bool { false }
//!     fn reset(&mut self) { self.current_value = 0.0; }
//! }
//! ```
//!
//! Once implemented, you can register it with `kindle-viz`'s `App` at startup.
extern crate alloc;
/// Auto-generated documentation for err.
pub mod err;
/// Auto-generated documentation for event.
pub mod event;
/// Auto-generated documentation for keymap.
pub mod keymap;
/// Auto-generated documentation for panel.
pub mod panel;
/// Auto-generated documentation for plugin.
pub mod plugin;
/// Auto-generated documentation for render_ctx.
pub mod render_ctx;

/// Auto-generated documentation for prelude.
pub mod prelude {
    pub use crate::err::{Error, Result};
    pub use crate::event::*;
    pub use crate::keymap::{Action, KeymapProvider};
    pub use crate::panel::Panel;
    pub use crate::plugin::Plugin;
    pub use crate::render_ctx::{HitId, RenderCtx};
}
