//! Behavior tests for the built-in panels (Plan 08-05 Task 1, TDD).
//!
//! Rendering assertions drive a `ratatui::backend::TestBackend` and inspect
//! the produced cell buffer as text, mirroring `dispatch.rs`'s in-crate
//! TestBackend test setup.

use kindle_telemetry::events::{CURRENT_SCHEMA_VERSION, Event, MemoryEvent, ScalarEvent};
use kindle_viz::panels::loss::LossPanel;
use kindle_viz::panels::panic_test::PanicTestPanel;
use kindle_viz_plugin_api::event::{KeyCode, KeyModifiers, PanelEvent, PanelKeyEvent};
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;

/// Loss event.
fn loss_event(step: usize, value: f64) -> Event {
    Event::Scalar(ScalarEvent {
        schema_version: CURRENT_SCHEMA_VERSION,
        step,
        name: String::from("loss"),
        value,
    })
}

/// Key.
fn key(c: char) -> PanelEvent {
    PanelEvent::Key(PanelKeyEvent {
        code: KeyCode::Char(c),
        modifiers: KeyModifiers {
            ctrl: false,
            shift: false,
            alt: false,
        },
    })
}

/// Renders `panel` into a 60x12 TestBackend and returns the buffer content
/// as one string (row-contiguous, so `contains` works for single-row text).
fn render_to_text(panel: &mut dyn Panel) -> String {
    let backend = ratatui::backend::TestBackend::new(60, 12);
    let mut terminal = ratatui::Terminal::new(backend).expect("terminal should construct");
    terminal
        .draw(|frame| {
            let area = frame.area();
            let mut hits = Vec::new();
            let mut ctx = RenderCtx::new(frame, area, &mut hits);
            panel.render(&mut ctx);
        })
        .expect("draw should succeed");
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect()
}

#[test]
/// Loss panel renders waiting placeholder when empty.
fn loss_panel_renders_waiting_placeholder_when_empty() {
    let mut panel = LossPanel::new();
    let text = render_to_text(&mut panel);
    assert!(
        text.contains("waiting for loss events…"),
        "empty LossPanel must render the exact UI-SPEC placeholder, got: {text:?}"
    );
    assert!(text.contains("Loss"), "block title must be present");
}

#[test]
/// Loss panel accumulates only loss scalars and renders chart.
fn loss_panel_accumulates_only_loss_scalars_and_renders_chart() {
    let mut panel = LossPanel::new();
    panel.update(&loss_event(0, 2.0));
    panel.update(&loss_event(1, 1.5));
    // Non-loss scalar and non-scalar events must be ignored.
    panel.update(&Event::Scalar(ScalarEvent {
        schema_version: CURRENT_SCHEMA_VERSION,
        step: 1,
        name: String::from("lr"),
        value: 0.01,
    }));
    panel.update(&Event::Memory(MemoryEvent {
        schema_version: CURRENT_SCHEMA_VERSION,
        step: 1,
        rss_bytes: 1024,
    }));
    panel.update(&loss_event(2, 1.1));

    let text = render_to_text(&mut panel);
    assert!(
        !text.contains("waiting for loss events…"),
        "LossPanel with data must render the chart, not the placeholder"
    );
    assert!(text.contains("Loss"), "block title must be present");
    assert!(text.contains("step"), "x-axis title must be present");
    assert!(text.contains("loss"), "y-axis title must be present");
}

#[test]
/// Loss panel reset clears accumulated points.
fn loss_panel_reset_clears_accumulated_points() {
    let mut panel = LossPanel::new();
    panel.update(&loss_event(0, 2.0));
    panel.reset();
    let text = render_to_text(&mut panel);
    assert!(
        text.contains("waiting for loss events…"),
        "reset LossPanel must render the placeholder again"
    );
}

#[test]
/// Loss panel handle event consumes nothing.
fn loss_panel_handle_event_consumes_nothing() {
    let mut panel = LossPanel::new();
    assert!(!panel.handle_event(&key('p')));
}

#[test]
/// Panic test panel panics on p in handle event.
fn panic_test_panel_panics_on_p_in_handle_event() {
    // Suppress the default panic hook's stderr print -- the panic is the
    // expected behavior under test, not a failure.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let mut panel = PanicTestPanel;
    let event = key('p');
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        panel.handle_event(&event)
    }));

    std::panic::set_hook(previous_hook);
    assert!(
        result.is_err(),
        "PanicTestPanel::handle_event must panic on 'p' (PLUGIN-03 proof)"
    );
}

#[test]
/// Panic test panel ignores other keys and renders hint.
fn panic_test_panel_ignores_other_keys_and_renders_hint() {
    let mut panel = PanicTestPanel;
    assert!(!panel.handle_event(&key('x')));
    let text = render_to_text(&mut panel);
    assert!(
        text.contains("press 'p' to trigger a deliberate panic"),
        "PanicTestPanel must render its hint text, got: {text:?}"
    );
    assert!(text.contains("Panic Test"), "block title must be present");
}
