//! Contains the `Panel` trait which defines the core interface for UI plugins.
//!
//! A [`crate::panel::Panel`] acts as a single, self-contained view within the `incin-viz` dashboard.
//! It is responsible for tracking its own state, responding to incoming telemetry events,
//! handling user input (like mouse clicks or key presses when focused), and rendering
//! its contents into a designated region on the screen.
//!
//! Because `incin-viz` may spawn background tasks, panels must be `Send` and are
//! updated using mutable references (`&mut self`), clearly defining ownership and
//! allowing them to hold persistent state across frames without global mutability or locks.

use crate::event::PanelEvent;
use crate::render_ctx::RenderCtx;
use incin_telemetry::events::Event;

/// A single pane's behavior: receives telemetry updates, receives input
/// events, and renders itself into a `RenderCtx`-bounded region every
/// render tick. `Send` is required because panels may be constructed and
/// held across `incin-viz`'s async runtime's task boundaries.
pub trait Panel: Send {
    /// Stable identifier for focus-cycling/logging. Not user-visible.
    fn id(&self) -> &'static str;

    /// Human-readable title rendered in the panel's `Block::title()`.
    fn title(&self) -> &str;

    /// Called once per new telemetry event this panel subscribes to.
    fn update(&mut self, event: &Event);

    /// Renders this panel's current state into the given context.
    fn render(&mut self, ctx: &mut RenderCtx<'_, '_>);

    /// Handles an input event routed to this panel; returns whether it was consumed.
    fn handle_event(&mut self, event: &PanelEvent) -> bool;

    /// Returns hover context text for a registered hit region.
    fn hover_text(&self, _id: crate::render_ctx::HitId) -> Option<String> {
        None
    }

    /// Re-initializes this panel's internal state in place, for the
    /// 'r'-key retry UX. Required -- no default no-op body.
    fn reset(&mut self);
}
