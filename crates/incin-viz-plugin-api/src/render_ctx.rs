//! Source: UI-SPEC.md's locked render-context boundary (thin wrapper over
//! `ratatui::Frame`/`Rect`, re-exporting `ratatui-widgets` types directly).

use ratatui_core::layout::Rect;
use ratatui_core::terminal::Frame;

/// Opaque identifier for a hit-testable region a panel registers during
/// render (Phase 10 mouse support consumes this; signature only this
/// phase per UI-SPEC.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HitId(pub u32);

/// Thin wrapper over `ratatui::Frame`/`Rect` that every `Panel::render`
/// call receives. A panel calls `frame_mut().render_widget(widget, area)`
/// exactly as it would in raw `ratatui` code -- no bespoke draw-command
/// abstraction layer to learn.
pub struct RenderCtx<'a, 'b> {
    frame: &'a mut Frame<'b>,
    area: Rect,
    hit_regions: &'a mut Vec<(Rect, HitId)>,
    alert_msg: Option<String>,
}

impl<'a, 'b> RenderCtx<'a, 'b> {
    /// Constructs a new render context for the given frame/area. This is
    /// the crate's public construction path -- the host (incin-viz's
    /// dispatch.rs/app.rs) builds one per panel per render tick.
    pub fn new(
        frame: &'a mut Frame<'b>,
        area: Rect,
        hit_regions: &'a mut Vec<(Rect, HitId)>,
    ) -> Self {
        Self {
            frame,
            area,
            hit_regions,
            alert_msg: None,
        }
    }

    /// Returns a mutable reference to the ratatui frame for rendering widgets.
    pub fn frame_mut(&mut self) -> &mut Frame<'b> {
        self.frame
    }

    /// Returns the bounding layout area assigned to this panel.
    pub fn area(&self) -> Rect {
        self.area
    }

    /// Registers a hit-testable region within this panel's area. Additive
    /// -- does not change how panels call `frame_mut().render_widget(...)`.
    pub fn register_hit_region(&mut self, rect: Rect, id: HitId) {
        self.hit_regions.push((rect, id));
    }

    /// Set an alert message to be displayed for this panel.
    pub fn set_alert(&mut self, msg: String) {
        self.alert_msg = Some(msg);
    }

    /// Takes the alert message, consuming it.
    pub fn take_alert(&mut self) -> Option<String> {
        self.alert_msg.take()
    }
}
