//! Proves `Box<dyn Panel>` and `Vec<Box<dyn Panel>>` compile -- i.e. `Panel`
//! is dyn-object-safe, the load-bearing property PLUGIN-01 requires before
//! any panel or the host shell can be built against this trait surface.

use kindle_telemetry::events::Event;
use kindle_viz_plugin_api::event::PanelEvent;
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;

struct NoopPanel;

impl Panel for NoopPanel {
    fn id(&self) -> &'static str {
        "noop"
    }

    fn title(&self) -> &str {
        "Noop"
    }

    fn update(&mut self, _event: &Event) {}

    fn render(&mut self, _ctx: &mut RenderCtx<'_, '_>) {}

    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        false
    }

    fn reset(&mut self) {}
}

#[test]
fn trait_object_safety() {
    let _panels: Vec<Box<dyn Panel>> = vec![Box::new(NoopPanel)];
    assert_eq!(_panels.len(), 1);
}
