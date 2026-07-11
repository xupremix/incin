//! Loss-curve panel (PANEL-01, D-01/D-02): a real ratatui `Chart` widget
//! plotting accumulated `ScalarEvent { name: "loss" }` samples.

use kindle_telemetry::events::Event;
use kindle_viz_plugin_api::event::PanelEvent;
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;

/// Tracks the training loss curve as `(step, loss)` points and renders it
/// as a braille-marker `Chart` (cyan accent line per UI-SPEC.md).
#[derive(Default)]
pub struct LossPanel {
    points: Vec<(f64, f64)>,
}

impl LossPanel {
    /// Creates an empty loss panel (infallible -- no pre-allocation).
    pub fn new() -> Self {
        Self::default()
    }
}

impl Panel for LossPanel {
    fn id(&self) -> &'static str {
        "loss"
    }

    fn title(&self) -> &str {
        "Loss"
    }

    fn update(&mut self, _event: &Event) {
        // RED stub -- implemented in the GREEN phase.
    }

    fn render(&mut self, _ctx: &mut RenderCtx<'_, '_>) {
        // RED stub -- implemented in the GREEN phase.
        let _ = &self.points;
    }

    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        false
    }

    fn reset(&mut self) {
        // RED stub -- implemented in the GREEN phase.
    }
}
