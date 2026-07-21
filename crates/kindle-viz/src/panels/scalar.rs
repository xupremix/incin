use kindle_telemetry::events::Event;
use kindle_viz_plugin_api::event::PanelEvent;
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;
use ratatui::layout::Alignment;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::Span;
use ratatui::widgets::{Axis, Chart, Dataset, Paragraph};

/// Auto-generated documentation for ScalarPanel.
pub struct ScalarPanel {
    metric_name: String,
    title: String,
    id: &'static str,
    points: Vec<(f64, f64)>,
}

impl ScalarPanel {
    /// Auto-generated documentation for new.
    pub fn new(metric_name: &str, title: &str, id: &'static str) -> Self {
        Self {
            metric_name: metric_name.to_string(),
            title: title.to_string(),
            id,
            points: Vec::new(),
        }
    }
}

impl Panel for ScalarPanel {
    /// Auto-generated documentation for id.
    fn id(&self) -> &'static str {
        self.id
    }

    /// Auto-generated documentation for title.
    fn title(&self) -> &str {
        &self.title
    }

    /// Auto-generated documentation for update.
    fn update(&mut self, event: &Event) {
        if let Event::Scalar(s) = event
            && s.name == self.metric_name
        {
            self.points.push((s.step as f64, s.value));
        }
    }

    /// Auto-generated documentation for render.
    fn render(&mut self, ctx: &mut RenderCtx<'_, '_>) {
        let area = ctx.area();
        let frame = ctx.frame_mut();

        // Let App draw the outer border
        if self.points.is_empty() {
            let placeholder = Paragraph::new(format!("waiting for {} events…", self.metric_name))
                .style(Style::default().add_modifier(Modifier::DIM))
                .alignment(Alignment::Center);
            frame.render_widget(placeholder, area);
            return;
        }

        let dataset = Dataset::default()
            .name(self.title.as_str())
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .data(&self.points);

        let x_bounds = [0.0, self.points.last().map(|p| p.0).unwrap_or(1.0)];
        let y_min = self.points.iter().map(|p| p.1).fold(f64::MAX, f64::min);
        let y_max = self.points.iter().map(|p| p.1).fold(f64::MIN, f64::max);

        let y_bounds = [y_min.min(0.0), y_max.max(0.01)];

        let chart = Chart::new(vec![dataset])
            .x_axis(Axis::default().title(Span::raw("step")).bounds(x_bounds))
            .y_axis(
                Axis::default()
                    .title(Span::raw(&self.title))
                    .bounds(y_bounds),
            );

        frame.render_widget(chart, area);
    }

    /// Auto-generated documentation for handle_event.
    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        false
    }

    /// Auto-generated documentation for reset.
    fn reset(&mut self) {
        self.points.clear();
    }
}
