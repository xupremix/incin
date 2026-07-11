//! Host-owned panic-containment layer: every `Panel` trait-method call from
//! `app.rs`'s event loop is routed through one of these wrapper functions,
//! never called on the panel directly. This is the only boundary between
//! untrusted-quality (not untrusted-intent) plugin code and host process
//! stability -- see PLUGIN-03 / T-08-01 in `08-04-PLAN.md`'s threat model.
//!
//! Source: `.planning/phases/08-plugin-api-base-tui-shell/08-RESEARCH.md`
//! "Panic Isolation: catch_unwind Mechanics" (lines 580-660) -- this module
//! is a direct copy of that fully-worked pattern, no in-repo analog exists
//! per `08-PATTERNS.md`.

use std::panic::{self, AssertUnwindSafe};

use kindle_telemetry::events::Event;
use kindle_viz_plugin_api::event::PanelEvent;
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;

/// Result of a dispatched panel call: either it ran normally, or it
/// panicked and the host must now treat this panel as crashed (D-04 /
/// UI-SPEC.md Panic Isolation UX).
pub enum DispatchOutcome<T> {
    Ok(T),
    Panicked,
}

/// Dispatches `Panel::render`, catching any panic raised inside it.
///
/// `AssertUnwindSafe` is sound here specifically because: (1) `panel` and
/// `ctx` are the *only* mutable state the closure touches, (2) if the
/// closure panics mid-render, this panel instance is immediately
/// marked-crashed by the caller (`app.rs`) and never rendered into again
/// until an explicit `r`-triggered `dispatch_reset` -- so a partially-
/// mutated `Panel` is never observed again in a way that matters. This is
/// the textbook justification `AssertUnwindSafe`'s own docs describe.
pub fn dispatch_render(panel: &mut dyn Panel, ctx: &mut RenderCtx<'_, '_>) -> DispatchOutcome<()> {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        panel.render(ctx);
    }));
    match result {
        Ok(()) => DispatchOutcome::Ok(()),
        Err(_payload) => DispatchOutcome::Panicked,
    }
}

/// Dispatches `Panel::update`, catching any panic raised inside it. Same
/// `AssertUnwindSafe` justification as [`dispatch_render`].
pub fn dispatch_update(panel: &mut dyn Panel, event: &Event) -> DispatchOutcome<()> {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        panel.update(event);
    }));
    match result {
        Ok(()) => DispatchOutcome::Ok(()),
        Err(_) => DispatchOutcome::Panicked,
    }
}

/// Dispatches `Panel::handle_event`, catching any panic raised inside it.
/// Same `AssertUnwindSafe` justification as [`dispatch_render`].
pub fn dispatch_handle_event(panel: &mut dyn Panel, event: &PanelEvent) -> DispatchOutcome<bool> {
    let result = panic::catch_unwind(AssertUnwindSafe(|| panel.handle_event(event)));
    match result {
        Ok(consumed) => DispatchOutcome::Ok(consumed),
        Err(_) => DispatchOutcome::Panicked,
    }
}

/// Dispatches `Panel::reset`, catching any panic raised inside it. Not part
/// of RESEARCH.md's original draft, but required since the 'r'-key retry
/// (UI-SPEC.md's Panic Isolation UX) calls `Panel::reset` and that call must
/// ALSO be panic-contained -- a panel whose `reset()` itself panics must not
/// re-crash the host either.
pub fn dispatch_reset(panel: &mut dyn Panel) -> DispatchOutcome<()> {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        panel.reset();
    }));
    match result {
        Ok(()) => DispatchOutcome::Ok(()),
        Err(_) => DispatchOutcome::Panicked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PanickingPanel;

    impl Panel for PanickingPanel {
        fn id(&self) -> &'static str {
            "panicking-test-panel"
        }

        fn title(&self) -> &str {
            "Panicking Test Panel"
        }

        fn update(&mut self, _event: &Event) {}

        fn render(&mut self, _ctx: &mut RenderCtx<'_, '_>) {
            panic!("deliberate test panic");
        }

        fn handle_event(&mut self, _event: &PanelEvent) -> bool {
            false
        }

        fn reset(&mut self) {}
    }

    #[test]
    fn panicking_panel_render_is_caught() {
        // Suppress the default panic hook's stderr print for the duration
        // of this test only -- the panic is expected/handled, not a genuine
        // test failure, and printing it would be misleading test output.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let mut panel = PanickingPanel;
        let backend = ratatui::backend::TestBackend::new(20, 10);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal should construct");

        terminal
            .draw(|frame| {
                let area = frame.area();
                let mut hits = Vec::new();
                let mut ctx = RenderCtx::new(frame, area, &mut hits);
                let outcome = dispatch_render(&mut panel, &mut ctx);
                assert!(
                    matches!(outcome, DispatchOutcome::Panicked),
                    "a panicking Panel::render must be caught and reported as Panicked"
                );
            })
            .expect("terminal draw should succeed even though the panel panicked");

        std::panic::set_hook(previous_hook);
    }
}
