//! Source: signatures derived from UI-SPEC.md's render-context primitives
//! + ARCHITECTURE.md's Panel responsibility list (render/handle_event/
//! title/id) + Pitfall 6's explicit call-out that "does a plugin own
//! persistent state across frames" must be answered by the trait shape
//! (answered here: yes, via `&mut self` on every method -- a Panel is a
//! stateful, owned object, not a stateless render function).

use crate::event::PanelEvent;
use crate::render_ctx::RenderCtx;
use kindle_telemetry::events::Event;

/// A single pane's behavior: receives telemetry updates, receives input
/// events, and renders itself into a `RenderCtx`-bounded region every
/// render tick. `Send` is required because panels may be constructed and
/// held across `kindle-viz`'s async runtime's task boundaries.
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

    /// Re-initializes this panel's internal state in place, for the
    /// 'r'-key retry UX. Required -- no default no-op body.
    fn reset(&mut self);
}
