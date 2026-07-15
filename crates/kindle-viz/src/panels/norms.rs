use alloc::collections::BTreeMap;
use kindle_telemetry::events::{Event, GradientNormEvent, WeightNormEvent};
use kindle_viz_plugin_api::event::PanelEvent;
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;
use ratatui::layout::Alignment;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::Span;
use ratatui::widgets::{Axis, Chart, Dataset, Paragraph};

pub enum NormType {
    Gradient,
    Weight,
}

pub struct NormsPanel {
    norm_type: NormType,
    title: String,
    id: &'static str,
    // (step, sum_sq_of_norms)
    step_aggregates: BTreeMap<usize, f64>,
    points: Vec<(f64, f64)>, // (step, global_l2_norm)
    alert_threshold: Option<f64>,
}

impl NormsPanel {
    pub fn new(norm_type: NormType, title: &str, id: &'static str, alert_threshold: Option<f64>) -> Self {
        Self {
            norm_type,
            title: title.to_string(),
            id,
            step_aggregates: BTreeMap::new(),
            points: Vec::new(),
            alert_threshold,
        }
    }
}

impl Panel for NormsPanel {
    fn id(&self) -> &'static str {
        self.id
    }

    fn title(&self) -> &str {
        &self.title
    }

    fn update(&mut self, event: &Event) {
        let (step, l2_norm) = match (&self.norm_type, event) {
            (NormType::Gradient, Event::GradientNorm(GradientNormEvent { step, l2_norm, .. })) => {
                (*step, *l2_norm)
            }
            (NormType::Weight, Event::WeightNorm(WeightNormEvent { step, l2_norm, .. })) => {
                (*step, *l2_norm)
            }
            _ => return,
        };

        let sum_sq = self.step_aggregates.entry(step).or_insert(0.0);
        *sum_sq += (l2_norm as f64).powi(2);

        // Update the points array with the latest global norm for this step.
        // We'll just rebuild the points vector or update the last point if it's the same step.
        let global_norm = sum_sq.sqrt();
        
        if let Some(last) = self.points.last_mut() {
            if last.0 as usize == step {
                last.1 = global_norm;
            } else {
                self.points.push((step as f64, global_norm));
            }
        } else {
            self.points.push((step as f64, global_norm));
        }
    }

    fn render(&mut self, ctx: &mut RenderCtx<'_, '_>) {
        let area = ctx.area();

        if let Some(threshold) = self.alert_threshold {
            if let Some(last_val) = self.points.last().map(|p| p.1) {
                if !last_val.is_finite() || last_val > threshold {
                    ctx.set_alert(format!("Spike detected ({:.2} > {})", last_val, threshold));
                }
            }
        }

        let frame = ctx.frame_mut();

        if self.points.is_empty() {
            let placeholder = Paragraph::new(format!("waiting for {} events…", self.title))
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
        let y_max = self.points.iter().map(|p| p.1).fold(f64::MIN, f64::max);
        
        let y_bounds = [0.0, y_max.max(0.01)];

        let chart = Chart::new(vec![dataset])
            .x_axis(Axis::default().title(Span::raw("step")).bounds(x_bounds))
            .y_axis(Axis::default().title(Span::raw(&self.title)).bounds(y_bounds));

        frame.render_widget(chart, area);
    }

    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        false
    }

    fn reset(&mut self) {
        self.points.clear();
        self.step_aggregates.clear();
    }
}
