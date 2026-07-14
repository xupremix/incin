//! Loss-curve panel (PANEL-01, D-01/D-02): a real ratatui `Chart` widget
//! plotting accumulated `ScalarEvent { name: "loss" }` samples.
//!
//! Source: `.planning/phases/08-plugin-api-base-tui-shell/08-RESEARCH.md`
//! "Loss panel render" code example (lines 700-787), adapted to the
//! `ratatui` facade paths `kindle-viz` depends on.

use kindle_telemetry::events::Event;
use kindle_viz_plugin_api::event::PanelEvent;
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;
use ratatui::layout::Alignment;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::Span;
use ratatui::widgets::{Axis, Block, Borders, Chart, Dataset, Paragraph};

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

    fn update(&mut self, event: &Event) {
        if let Event::Scalar(s) = event
            && s.name == "loss"
        {
            self.points.push((s.step as f64, s.value));
        }
    }

    fn render(&mut self, ctx: &mut RenderCtx<'_, '_>) {
        let area = ctx.area();

        if let Some(last_val) = self.points.last().map(|p| p.1) {
            if !last_val.is_finite() {
                ctx.set_alert(format!("NaN/Inf detected"));
            }
        }

        let frame = ctx.frame_mut();

        if self.points.is_empty() {
            // Exact UI-SPEC.md copy -- distinguishes "no data yet" from a
            // broken chart.
            let placeholder = Paragraph::new("waiting for loss events…")
                .style(Style::default().add_modifier(Modifier::DIM))
                .alignment(Alignment::Center)
                .block(Block::default().title(self.title()).borders(Borders::ALL));
            frame.render_widget(placeholder, area);
            return;
        }

        let dataset = Dataset::default()
            .name("loss")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .data(&self.points);

        let x_bounds = [0.0, self.points.last().map(|p| p.0).unwrap_or(1.0)];
        let y_max = self.points.iter().map(|p| p.1).fold(f64::MIN, f64::max);
        // Floor at 0.01 to avoid a degenerate zero-height y axis.
        let y_bounds = [0.0, y_max.max(0.01)];

        let chart = Chart::new(vec![dataset])
            .x_axis(Axis::default().title(Span::raw("step")).bounds(x_bounds))
            .y_axis(Axis::default().title(Span::raw("loss")).bounds(y_bounds))
            .block(Block::default().title(self.title()).borders(Borders::ALL));

        frame.render_widget(chart, area);
    }

    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        false // loss panel has no panel-local key handling this phase
    }

    fn reset(&mut self) {
        self.points.clear();
    }
}
