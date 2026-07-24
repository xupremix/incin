//! Proves `Box<dyn Panel>` and `Vec<Box<dyn Panel>>` compile -- i.e. `Panel`
//! is dyn-object-safe, the load-bearing property PLUGIN-01 requires before
//! any panel or the host shell can be built against this trait surface.

use incin_telemetry::events::Event;
use incin_viz_plugin_api::event::PanelEvent;
use incin_viz_plugin_api::panel::Panel;
use incin_viz_plugin_api::render_ctx::RenderCtx;

/// Noop panel.
struct NoopPanel;

impl Panel for NoopPanel {
    /// Id.
    fn id(&self) -> &'static str {
        "noop"
    }

    /// Title.
    fn title(&self) -> &str {
        "Noop"
    }

    /// Update.
    fn update(&mut self, _event: &Event) {}

    /// Render.
    fn render(&mut self, _ctx: &mut RenderCtx<'_, '_>) {}

    /// Handle event.
    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        false
    }

    /// Reset.
    fn reset(&mut self) {}
}

#[test]
/// Trait object safety.
fn trait_object_safety() {
    let _panels: Vec<Box<dyn Panel>> = vec![Box::new(NoopPanel)];
    assert_eq!(_panels.len(), 1);
}
