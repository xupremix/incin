use kindle_telemetry::events::Event;
use kindle_viz_plugin_api::event::PanelEvent;
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;
use ratatui::layout::Alignment;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::Span;
use ratatui::widgets::{Axis, Chart, Dataset, Paragraph};

/// Core abstraction for `ScalarPanel` within the Kindle framework.
pub struct ScalarPanel {
    metric_name: String,
    title: String,
    id: &'static str,
    points: Vec<(f64, f64)>,
}

impl ScalarPanel {
    /// Core abstraction for `new` within the Kindle framework.
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
    /// Core abstraction for `id` within the Kindle framework.
    fn id(&self) -> &'static str {
        self.id
    }

    /// Core abstraction for `title` within the Kindle framework.
    fn title(&self) -> &str {
        &self.title
    }

    /// Core abstraction for `update` within the Kindle framework.
    fn update(&mut self, event: &Event) {
        if let Event::Scalar(s) = event
            && s.name == self.metric_name
        {
            self.points.push((s.step as f64, s.value));
        }
    }

    /// Core abstraction for `render` within the Kindle framework.
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

    /// Core abstraction for `handle_event` within the Kindle framework.
    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        false
    }

    /// Core abstraction for `reset` within the Kindle framework.
    fn reset(&mut self) {
        self.points.clear();
    }
}
