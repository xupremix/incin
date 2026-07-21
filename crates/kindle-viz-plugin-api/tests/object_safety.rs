//! Proves `Box<dyn Panel>` and `Vec<Box<dyn Panel>>` compile -- i.e. `Panel`
//! is dyn-object-safe, the load-bearing property PLUGIN-01 requires before
//! any panel or the host shell can be built against this trait surface.

use kindle_telemetry::events::Event;
use kindle_viz_plugin_api::event::PanelEvent;
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;

/// Core abstraction for `NoopPanel` within the Kindle framework.
struct NoopPanel;

impl Panel for NoopPanel {
    /// Core abstraction for `id` within the Kindle framework.
    fn id(&self) -> &'static str {
        "noop"
    }

    /// Core abstraction for `title` within the Kindle framework.
    fn title(&self) -> &str {
        "Noop"
    }

    /// Core abstraction for `update` within the Kindle framework.
    fn update(&mut self, _event: &Event) {}

    /// Core abstraction for `render` within the Kindle framework.
    fn render(&mut self, _ctx: &mut RenderCtx<'_, '_>) {}

    /// Core abstraction for `handle_event` within the Kindle framework.
    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        false
    }

    /// Core abstraction for `reset` within the Kindle framework.
    fn reset(&mut self) {}
}

#[test]
/// Core abstraction for `trait_object_safety` within the Kindle framework.
fn trait_object_safety() {
    let _panels: Vec<Box<dyn Panel>> = vec![Box::new(NoopPanel)];
    assert_eq!(_panels.len(), 1);
}
