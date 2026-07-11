//! Deliberate panic-test panel (D-04): exists solely to make PLUGIN-03's
//! panic-isolation guarantee visibly provable -- pressing 'p' while this
//! panel is focused panics its `handle_event` on purpose.

use kindle_telemetry::events::Event;
use kindle_viz_plugin_api::event::PanelEvent;
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;

/// Stateless panel whose only behavior is a deliberate panic on 'p'.
pub struct PanicTestPanel;

impl Panel for PanicTestPanel {
    fn id(&self) -> &'static str {
        "panic-test"
    }

    fn title(&self) -> &str {
        "Panic Test"
    }

    fn update(&mut self, _event: &Event) {}

    fn render(&mut self, _ctx: &mut RenderCtx<'_, '_>) {
        // RED stub -- implemented in the GREEN phase.
    }

    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        // RED stub -- implemented in the GREEN phase.
        false
    }

    fn reset(&mut self) {
        // stateless -- nothing to reset
    }
}
