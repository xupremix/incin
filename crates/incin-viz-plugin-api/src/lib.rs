//! `incin-viz-plugin-api`: The stable, independently-compilable trait
//! surface for building custom visualizations and UI plugins for `incin-viz`.
//!
//! This crate provides the foundational types and traits (`Panel`, `Plugin`, `KeymapProvider`,
//! and `RenderCtx`) needed to extend the Incin Visualizer dashboard. By separating
//! the plugin API from the main `incin-viz` crate, developers can build custom telemetry
//! visualizers without pulling in heavy TUI or application runtime dependencies.
//!
//! # Architecture
//!
//! The plugin architecture revolves around the [`panel::Panel`] trait. A `Panel` is a stateful
//! struct that:
//! 1. Receives incoming `incin-telemetry` `Event`s via `update()`.
//! 2. Renders itself to a bounded screen region using a [`render_ctx::RenderCtx`] via `render()`.
//! 3. Responds to keyboard and mouse inputs via `handle_event()`.
//!
//! The [`render_ctx::RenderCtx`] provides a safe, abstracted interface to the underlying rendering
//! engine (`ratatui`), preventing plugins from accidentally overwriting other panels
//! or crashing the host application by drawing outside their assigned boundaries.
//!
//! # Getting Started
//!
//! To create a custom panel, implement the [`panel::Panel`] trait:
//!
//! ```rust,no_run
//! use incin_viz_plugin_api::prelude::*;
//! use incin_telemetry::events::{Event, ScalarEvent};
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
//! Once implemented, you can register it with `incin-viz`'s `App` at startup.
extern crate alloc;
/// Error types and result alias for plugin operations.
pub mod err;
/// Input events (keyboard, mouse, focus) for interactive panels.
pub mod event;
/// Keymap actions and resolution.
pub mod keymap;
/// State and lifecycle interface for visualizer panels.
pub mod panel;
/// Plugin bundling interface.
pub mod plugin;
/// Rendering context wrapper for ratatui buffer access.
pub mod render_ctx;

/// Convenient re-exports for building visualizer plugins.
pub mod prelude {
    pub use crate::err::{Error, Result};
    pub use crate::event::*;
    pub use crate::keymap::{Action, KeymapProvider};
    pub use crate::panel::Panel;
    pub use crate::plugin::Plugin;
    pub use crate::render_ctx::{HitId, RenderCtx};
}
