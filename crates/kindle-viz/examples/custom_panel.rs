use kindle_viz_plugin_api::prelude::*;
use kindle_telemetry::events::{Event, ScalarEvent};
use kindle_viz::app::{App, DefaultKeymap};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::style::{Style, Color};
use ratatui::layout::Alignment;
use std::io::stdout;

struct CustomMetricPanel {
    current_value: f64,
    border_color: Color,
}

impl CustomMetricPanel {
    fn new() -> Self {
        Self {
            current_value: 0.0,
            border_color: Color::Blue,
        }
    }
}

impl Panel for CustomMetricPanel {
    fn id(&self) -> &'static str {
        "custom_metric"
    }

    fn title(&self) -> &str {
        "Custom Metric Tracker"
    }

    fn update(&mut self, event: &Event) {
        if let Event::Scalar(ScalarEvent { name, value, .. }) = event {
            if name == "custom_metric" {
                self.current_value = *value;
            }
        }
    }

    fn render(&mut self, ctx: &mut RenderCtx<'_, '_>) {
        // Register a hit region spanning the entire panel area
        ctx.register_hit_region(ctx.area(), HitId(1));

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.border_color));

        let text = format!("Current Value: {:.2}\n(Press 'c' to change border color)", self.current_value);
        let paragraph = Paragraph::new(text)
            .block(block)
            .alignment(Alignment::Center);

        let area = ctx.area();
        ctx.frame_mut().render_widget(paragraph, area);
    }

    fn handle_event(&mut self, event: &PanelEvent) -> bool {
        match event {
            PanelEvent::Key(k) => {
                if let KeyCode::Char('c') = k.code {
                    self.border_color = if self.border_color == Color::Blue {
                        Color::Green
                    } else {
                        Color::Blue
                    };
                    true
                } else {
                    false
                }
            }
            PanelEvent::Mouse(_) => {
                false
            }
        }
    }

    fn hover_text(&self, id: HitId) -> Option<String> {
        if id == HitId(1) {
            Some("Hovering over custom metric panel!".to_string())
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.current_value = 0.0;
        self.border_color = Color::Blue;
    }
}

// Dummy transport to simulate events
struct DummyTransport {
    last_val: f64,
}

impl kindle_viz::transport_reader::TransportReader for DummyTransport {
    fn poll_new_events(&mut self) -> kindle_viz::err::Result<Vec<Event>> {
        self.last_val += 1.0;
        if self.last_val > 100.0 {
            self.last_val = 0.0;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
        Ok(vec![Event::Scalar(ScalarEvent {
            schema_version: 1,
            step: self.last_val as usize,
            name: "custom_metric".to_string(),
            value: self.last_val,
        })])
    }
}

fn install_panic_hook() {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        previous_hook(panic_info);
        let _ = crossterm::execute!(stdout(), crossterm::event::DisableMouseCapture);
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(stdout(), crossterm::terminal::LeaveAlternateScreen);
    }));
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let transport = Box::new(DummyTransport { last_val: 0.0 });
    let mut app = App::new(transport, "Custom Panel Example".to_string());
    
    app.register_panel(Box::new(CustomMetricPanel::new()));

    install_panic_hook();
    let terminal = ratatui::init();
    let _ = crossterm::execute!(stdout(), crossterm::event::EnableMouseCapture);
    
    let keymap = Box::new(DefaultKeymap);
    let result = kindle_viz::app::run(app, terminal, keymap).await;
    
    let _ = crossterm::execute!(stdout(), crossterm::event::DisableMouseCapture);
    ratatui::restore();
    
    result
}
