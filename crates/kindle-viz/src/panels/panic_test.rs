//! Deliberate panic-test panel (D-04): exists solely to make PLUGIN-03's
//! panic-isolation guarantee visibly provable -- pressing 'p' while this
//! panel is focused panics its `handle_event` on purpose.
//!
//! Source: `.planning/phases/08-plugin-api-base-tui-shell/08-RESEARCH.md`
//! "Panic-test panel" code example (lines 789-837).

use kindle_telemetry::events::Event;
use kindle_viz_plugin_api::event::{KeyCode, PanelEvent};
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;
use ratatui::widgets::{Block, Borders, Paragraph};

/// Stateless panel whose only behavior is a deliberate panic on 'p'.
pub struct PanicTestPanel;

impl Panel for PanicTestPanel {
    /// Auto-generated documentation for id.
    fn id(&self) -> &'static str {
        "panic-test"
    }

    /// Auto-generated documentation for title.
    fn title(&self) -> &str {
        "Panic Test"
    }

    /// Auto-generated documentation for update.
    fn update(&mut self, _event: &Event) {}

    /// Auto-generated documentation for render.
    fn render(&mut self, ctx: &mut RenderCtx<'_, '_>) {
        let area = ctx.area();
        let block = Block::default().title("Panic Test").borders(Borders::ALL);
        let text = Paragraph::new("press 'p' to trigger a deliberate panic").block(block);
        ctx.frame_mut().render_widget(text, area);
    }

    /// Auto-generated documentation for handle_event.
    fn handle_event(&mut self, event: &PanelEvent) -> bool {
        if let PanelEvent::Key(k) = event
            && k.code == KeyCode::Char('p')
        {
            panic!("Manual panic triggered from PanicTestPanel");
        }
        false
    }

    /// Auto-generated documentation for reset.
    fn reset(&mut self) {
        // stateless -- nothing to reset
    }
}
