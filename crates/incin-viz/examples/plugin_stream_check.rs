//! Drives the plugin hook contract against a real telemetry wire stream.
//!
//! Builds the `App` exactly as an embedding host would, registers a plugin
//! panel implementing [`Panel`], feeds it events read from a stream written
//! by the actual `Emitter` + `FileTransport` path, and renders through
//! ratatui's `TestBackend` so the buffer contents are assertable without a
//! terminal. This is the "at least one real plugin" half of the 0.1.0 viz
//! verification: the hooks exercised are `id`, `title`, `update`, and
//! `render`, plus the default keymap resolution path.
//!
//! Usage: `cargo run -p incin-viz --example plugin_stream_check -- <stream.jsonl>`

use incin_telemetry::events::Event;
use incin_viz::app::{App, DefaultKeymap};
use incin_viz::transport_reader::FileTransportReader;
use incin_viz_plugin_api::event::{KeyCode, KeyModifiers, PanelKeyEvent};
use incin_viz_plugin_api::keymap::KeymapProvider;
use incin_viz_plugin_api::panel::Panel;
use incin_viz_plugin_api::prelude::*;
use ratatui::{Terminal, backend::TestBackend};

/// A real plugin: counts `custom_metric` scalars and renders the tally.
struct CustomMetricPanel {
    seen: usize,
    last_value: f64,
}

impl CustomMetricPanel {
    fn new() -> Self {
        Self {
            seen: 0,
            last_value: f64::NAN,
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
        if let Event::Scalar(scalar) = event
            && scalar.name == "custom_metric"
        {
            self.seen += 1;
            self.last_value = scalar.value;
        }
    }

    fn handle_event(&mut self, _event: &PanelEvent) -> bool {
        // The plugin consumes no panel-local input.
        false
    }

    fn reset(&mut self) {
        self.seen = 0;
        self.last_value = f64::NAN;
    }

    fn render(&mut self, ctx: &mut RenderCtx<'_, '_>) {
        use ratatui::widgets::{Block, Borders, Paragraph, Widget};
        // The host draws the border chrome after this hook returns, so the
        // body belongs in the inner area.
        let inner = Block::default().borders(Borders::ALL).inner(ctx.area());
        let frame = ctx.frame_mut();
        Paragraph::new(format!(
            "samples: {} last: {:.3}",
            self.seen, self.last_value
        ))
        .render(inner, frame.buffer_mut());
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: plugin_stream_check <stream.jsonl>");

    let reader = FileTransportReader::open(path.as_ref())?;
    let mut app = App::new(Box::new(reader), path.clone());
    app.register_panel(Box::new(CustomMetricPanel::new()));

    // Pull the whole fixture through the transport into the panels. The
    // fixture writes 250 events; polling well past that drains the file.
    for _ in 0..1000 {
        app.poll_transport();
    }

    // Render through a test backend so the plugin's render hook output is
    // assertable without a terminal.
    let mut terminal = Terminal::new(TestBackend::new(240, 80))?;
    terminal.draw(|frame| app.render(frame))?;
    let buffer = terminal.backend().buffer();
    let content: String = buffer
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<Vec<_>>()
        .join("");
    assert!(
        content.contains("Custom Metric Tracker"),
        "plugin panel title missing from rendered buffer"
    );
    // Dump for diagnosis when the body assertion fails.
    eprintln!("--- rendered buffer ---\n{content}\n---");
    assert!(
        content.contains("samples:") && content.contains("last:"),
        "plugin panel body missing from rendered buffer"
    );

    // The default keymap must resolve quit through the plugin-api types.
    let quit = DefaultKeymap.resolve(PanelKeyEvent {
        code: KeyCode::Char('q'),
        modifiers: KeyModifiers {
            ctrl: false,
            shift: false,
            alt: false,
        },
    });
    assert_eq!(quit, Some(Action::Quit), "q must resolve to Quit");

    eprintln!("plugin_stream_check passed");
    Ok(())
}
