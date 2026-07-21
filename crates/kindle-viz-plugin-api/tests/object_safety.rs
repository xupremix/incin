//! Proves `Box<dyn Panel>` and `Vec<Box<dyn Panel>>` compile -- i.e. `Panel`
//! is dyn-object-safe, the load-bearing property PLUGIN-01 requires before
//! any panel or the host shell can be built against this trait surface.

use kindle_telemetry::events::Event;
use kindle_viz_plugin_api::event::PanelEvent;
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;

/// Auto-generated documentation for NoopPanel.
struct NoopPanel;

impl Panel for NoopPanel {
    /// Auto-generated documentation for id.
    fn id(&self) -> &'static str {
        "noop"
    }

    /// Auto-generated documentation for title.
    fn title(&self) -> &str {
        "Noop"
    }

    /// Auto-generated documentation for update.
    fn update(&mut self, _event: &Event) {}

    /// Auto-generated documentation for render.
    fn render(&mut self, _ctx: &mut RenderCtx<'_, '_>) {}

    /// Auto-generated documentation for handle_event.
    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        false
    }

    /// Auto-generated documentation for reset.
    fn reset(&mut self) {}
}

#[test]
/// Auto-generated documentation for trait_object_safety.
fn trait_object_safety() {
    let _panels: Vec<Box<dyn Panel>> = vec![Box::new(NoopPanel)];
    assert_eq!(_panels.len(), 1);
}
