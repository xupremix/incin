//! `kindle-viz`'s application state and `tokio::select!`-driven async event
//! loop, multiplexing three streams: terminal input
//! (`crossterm::event::EventStream`), transport-tail polling, and a fixed
//! render tick.
//!
//! Source: `.planning/phases/08-plugin-api-base-tui-shell/08-RESEARCH.md`
//! Pattern 3 (the full `tokio::select!` event loop) -- adapted to this
//! plan's empty-panel-registry skeleton. Plan 08-05 registers the real
//! loss/panic-test panels via [`App::register_panel`].

use std::time::Duration;

use crossterm::event::{Event as CrosstermEvent, EventStream, KeyCode as CtKeyCode, KeyEvent};
use futures_util::StreamExt;
use kindle_viz_plugin_api::event::{KeyCode, KeyModifiers, PanelEvent, PanelKeyEvent};
use kindle_viz_plugin_api::keymap::{Action, KeymapProvider};
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;
use ratatui::layout::{Alignment, Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders, Paragraph};
use tokio::time::interval;

use crate::dispatch::{self, DispatchOutcome};
use crate::transport_reader::TransportReader;

/// ~30fps render tick, well under human-perceptible flicker (research
/// Pitfall 8: decouple render rate from input/transport arrival rates).
const RENDER_TICK: Duration = Duration::from_millis(33);
/// Transport-tail poll cadence.
const TRANSPORT_POLL: Duration = Duration::from_millis(50);

/// Exact UI-SPEC.md footer copy.
const FOOTER_HINTS: &str = "q: quit  Tab: focus next  p: trigger panic (test panel)";

/// Exact UI-SPEC.md crashed-panel placeholder copy (Red/Bold, rendered
/// inside the crashed panel's still-intact border/title).
const CRASHED_PLACEHOLDER: &str = "⚠ panel crashed — press r to retry";

/// The host application: panel registry, per-panel crash state, focus, and
/// the transport being tailed.
pub struct App {
    /// Registered panels. Empty this plan -- Plan 08-05 populates it via
    /// [`App::register_panel`].
    panels: Vec<Box<dyn Panel>>,
    /// Per-panel crash state, indexed alongside `panels`. A crashed panel
    /// is never rendered into / updated again until an explicit reset
    /// (Plan 08-05 wires the 'r'-key retry and placeholder text).
    crashed: Vec<bool>,
    /// Index of the focused panel (meaningless while `panels` is empty).
    focused: usize,
    /// The transport being tailed for new telemetry events.
    transport: Box<dyn TransportReader>,
    /// Display string for the header (run id or resolved path).
    run_id_or_path: String,
    /// Set by `Action::Quit`; the event loop exits when true.
    should_quit: bool,
}

impl App {
    /// Creates an app with an empty panel registry tailing `transport`.
    pub fn new(transport: Box<dyn TransportReader>, run_id_or_path: String) -> Self {
        Self {
            panels: Vec::new(),
            crashed: Vec::new(),
            focused: 0,
            transport,
            run_id_or_path,
            should_quit: false,
        }
    }

    /// Registers a panel into the registry (Plan 08-05's stable extension
    /// point -- pushes into both `panels` and `crashed`).
    pub fn register_panel(&mut self, panel: Box<dyn Panel>) {
        self.panels.push(panel);
        self.crashed.push(false);
    }

    /// Applies a resolved keymap action to app state.
    pub fn handle_key(&mut self, action: Action) {
        match action {
            Action::Quit => self.should_quit = true,
            Action::FocusNext => {
                self.focused = (self.focused + 1) % self.panels.len().max(1);
            }
            Action::FocusPrev => {
                let len = self.panels.len().max(1);
                self.focused = (self.focused + len - 1) % len;
            }
            Action::RetryPanel => {
                // Scoped to the focused panel, only active post-crash
                // (UI-SPEC.md 'r' contract) -- no-op otherwise.
                if !self.panels.is_empty() && self.crashed[self.focused] {
                    match dispatch::dispatch_reset(self.panels[self.focused].as_mut()) {
                        DispatchOutcome::Ok(()) => self.crashed[self.focused] = false,
                        // A panel whose reset() itself panics stays visibly
                        // crashed -- no silent panic-retry-panic loop
                        // (T-08-07).
                        DispatchOutcome::Panicked => {}
                    }
                }
            }
            // PanelLocal is routed directly via `handle_panel_local_key`
            // from the event loop, never through the keymap resolver.
            Action::PanelLocal => {}
        }
    }

    /// Routes a key the default keymap does NOT map to a global action to
    /// the focused panel's `handle_event` (panic-contained). This is how
    /// 'p' reaches `PanicTestPanel` (D-04/PLUGIN-03).
    pub fn handle_panel_local_key(&mut self, key: PanelKeyEvent) {
        if self.panels.is_empty() || self.crashed[self.focused] {
            // A crashed panel is never re-invoked until an explicit retry.
            return;
        }
        if let DispatchOutcome::Panicked = dispatch::dispatch_handle_event(
            self.panels[self.focused].as_mut(),
            &PanelEvent::Key(key),
        ) {
            self.crashed[self.focused] = true;
        }
    }

    /// Polls the transport for newly-arrived events and routes each one to
    /// every non-crashed panel via the panic-contained dispatch layer.
    pub fn poll_transport(&mut self) {
        let Ok(events) = self.transport.poll_new_events() else {
            // Transport errors are non-fatal this plan (RESEARCH.md's
            // footer transport-error copy is Plan 08-05's polish).
            return;
        };
        for event in &events {
            for (i, panel) in self.panels.iter_mut().enumerate() {
                if self.crashed[i] {
                    continue;
                }
                if let DispatchOutcome::Panicked = dispatch::dispatch_update(panel.as_mut(), event)
                {
                    self.crashed[i] = true;
                }
            }
        }
    }

    /// Renders the root layout: header row, panel area, footer row
    /// (UI-SPEC.md Layout Regions). With an empty registry the panel area
    /// stays blank -- Plan 08-05 populates it.
    pub fn render(&mut self, frame: &mut ratatui::Frame) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(frame.area());

        let header = Paragraph::new(format!("kindle-viz — run: {}", self.run_id_or_path));
        frame.render_widget(header, rows[0]);

        let footer =
            Paragraph::new(FOOTER_HINTS).style(Style::default().add_modifier(Modifier::DIM));
        frame.render_widget(footer, rows[2]);

        if self.panels.is_empty() {
            return;
        }

        let share = (100 / self.panels.len().max(1)) as u16;
        let constraints: Vec<Constraint> = self
            .panels
            .iter()
            .map(|_| Constraint::Percentage(share))
            .collect();
        let panel_areas = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(constraints)
            .split(rows[1]);

        let mut hit_regions = Vec::new();
        for (i, panel) in self.panels.iter_mut().enumerate() {
            // Focus border/title color rule applies to every panel's block
            // regardless of crashed state (UI-SPEC.md: White focused /
            // DarkGray unfocused; Cyan is reserved for the loss chart line).
            let focus_color = if i == self.focused {
                Color::White
            } else {
                Color::DarkGray
            };
            let focus_block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(focus_color))
                .title(Span::styled(
                    panel.title().to_string(),
                    Style::default()
                        .fg(focus_color)
                        .add_modifier(Modifier::BOLD),
                ));

            if self.crashed[i] {
                // Placeholder without re-invoking the panicked panel until
                // the user explicitly retries ('r') -- border/title intact.
                let placeholder = Paragraph::new(CRASHED_PLACEHOLDER)
                    .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
                    .alignment(Alignment::Center)
                    .block(focus_block);
                frame.render_widget(placeholder, panel_areas[i]);
                continue;
            }
            let mut ctx = RenderCtx::new(frame, panel_areas[i], &mut hit_regions);
            if let DispatchOutcome::Panicked = dispatch::dispatch_render(panel.as_mut(), &mut ctx) {
                self.crashed[i] = true;
                continue;
            }
            // Re-draw only the border/title chrome over the panel's own
            // block so the focus color rule holds without the Panel trait
            // needing a focus parameter (Block leaves inner content intact).
            frame.render_widget(focus_block, panel_areas[i]);
        }
    }
}

/// The hardcoded default keymap for this phase (UI-SPEC.md Interaction &
/// Keybinding table). Phase 10 replaces this with a configurable/
/// vim-swappable `KeymapProvider` system using the same trait.
pub struct DefaultKeymap;

impl KeymapProvider for DefaultKeymap {
    fn resolve(&self, key: PanelKeyEvent) -> Option<Action> {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => Some(Action::Quit),
            // crossterm delivers Ctrl+C as a regular key event in raw mode.
            KeyCode::Char('c') if key.modifiers.ctrl => Some(Action::Quit),
            KeyCode::Tab => Some(Action::FocusNext),
            KeyCode::BackTab => Some(Action::FocusPrev),
            // Retry the focused panel post-crash (no-op when not crashed,
            // guarded in App::handle_key per UI-SPEC.md's 'r' scoping).
            KeyCode::Char('r') => Some(Action::RetryPanel),
            _ => None,
        }
    }
}

/// Converts a crossterm key event into the plugin-api's crossterm-free
/// `PanelKeyEvent` newtype. Unmapped key codes return `None` (ignored).
fn convert_key(key: KeyEvent) -> Option<PanelKeyEvent> {
    let code = match key.code {
        CtKeyCode::Char(c) => KeyCode::Char(c),
        CtKeyCode::Tab => KeyCode::Tab,
        CtKeyCode::BackTab => KeyCode::BackTab,
        CtKeyCode::Esc => KeyCode::Esc,
        CtKeyCode::Enter => KeyCode::Enter,
        _ => return None,
    };
    let modifiers = KeyModifiers {
        ctrl: key
            .modifiers
            .contains(crossterm::event::KeyModifiers::CONTROL),
        shift: key
            .modifiers
            .contains(crossterm::event::KeyModifiers::SHIFT),
        alt: key.modifiers.contains(crossterm::event::KeyModifiers::ALT),
    };
    Some(PanelKeyEvent { code, modifiers })
}

/// Runs the event loop until quit: multiplexes terminal input, transport
/// polling, and the fixed render tick -- no branch blocks the others.
pub async fn run(mut app: App, mut terminal: ratatui::DefaultTerminal) -> anyhow::Result<()> {
    let keymap = DefaultKeymap;
    let mut term_events = EventStream::new();
    let mut render_interval = interval(RENDER_TICK);
    let mut transport_interval = interval(TRANSPORT_POLL);

    loop {
        tokio::select! {
            maybe_event = term_events.next() => {
                match maybe_event {
                    Some(Ok(CrosstermEvent::Key(key))) => {
                        if let Some(panel_key) = convert_key(key) {
                            if let Some(action) = keymap.resolve(panel_key) {
                                app.handle_key(action);
                                if app.should_quit {
                                    break;
                                }
                            } else {
                                // Keys unmapped by the global keymap route
                                // to the focused panel (Action::PanelLocal
                                // path) -- how 'p' reaches PanicTestPanel.
                                app.handle_panel_local_key(panel_key);
                            }
                        }
                    }
                    // Resize is handled by the next render's layout
                    // recompute (immediate-mode, no cached layout).
                    Some(Ok(_)) => {}
                    Some(Err(_)) => {}
                    None => break, // stdin closed
                }
            }

            _ = transport_interval.tick() => {
                app.poll_transport();
            }

            _ = render_interval.tick() => {
                terminal.draw(|frame| app.render(frame))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_keymap_resolves_quit_and_focus_keys() {
        let keymap = DefaultKeymap;
        let no_mods = KeyModifiers {
            ctrl: false,
            shift: false,
            alt: false,
        };
        assert_eq!(
            keymap.resolve(PanelKeyEvent {
                code: KeyCode::Char('q'),
                modifiers: no_mods
            }),
            Some(Action::Quit)
        );
        assert_eq!(
            keymap.resolve(PanelKeyEvent {
                code: KeyCode::Esc,
                modifiers: no_mods
            }),
            Some(Action::Quit)
        );
        assert_eq!(
            keymap.resolve(PanelKeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers {
                    ctrl: true,
                    shift: false,
                    alt: false
                }
            }),
            Some(Action::Quit)
        );
        assert_eq!(
            keymap.resolve(PanelKeyEvent {
                code: KeyCode::Tab,
                modifiers: no_mods
            }),
            Some(Action::FocusNext)
        );
        assert_eq!(
            keymap.resolve(PanelKeyEvent {
                code: KeyCode::BackTab,
                modifiers: no_mods
            }),
            Some(Action::FocusPrev)
        );
        assert_eq!(
            keymap.resolve(PanelKeyEvent {
                code: KeyCode::Char('r'),
                modifiers: no_mods
            }),
            Some(Action::RetryPanel)
        );
        assert_eq!(
            keymap.resolve(PanelKeyEvent {
                code: KeyCode::Char('x'),
                modifiers: no_mods
            }),
            None
        );
    }

    /// Panel that deliberately panics on 'p' in handle_event, mirroring
    /// PanicTestPanel's shape for exercising App's crash/retry state.
    struct CrashOnP;

    impl Panel for CrashOnP {
        fn id(&self) -> &'static str {
            "crash-on-p"
        }
        fn title(&self) -> &str {
            "Crash On P"
        }
        fn update(&mut self, _event: &kindle_telemetry::events::Event) {}
        fn render(&mut self, _ctx: &mut RenderCtx<'_, '_>) {}
        fn handle_event(&mut self, event: &PanelEvent) -> bool {
            let PanelEvent::Key(k) = event;
            if k.code == KeyCode::Char('p') {
                panic!("deliberate test panic");
            }
            false
        }
        fn reset(&mut self) {}
    }

    struct NoopTransport;

    impl crate::transport_reader::TransportReader for NoopTransport {
        fn poll_new_events(&mut self) -> crate::err::Result<Vec<kindle_telemetry::events::Event>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn panel_local_panic_marks_crashed_and_retry_recovers() {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let mut app = App::new(Box::new(NoopTransport), "test".to_string());
        app.register_panel(Box::new(CrashOnP));
        let no_mods = KeyModifiers {
            ctrl: false,
            shift: false,
            alt: false,
        };

        // 'p' routed panel-locally panics the panel -> marked crashed.
        app.handle_panel_local_key(PanelKeyEvent {
            code: KeyCode::Char('p'),
            modifiers: no_mods,
        });
        assert!(app.crashed[0], "panicking handle_event must mark crashed");

        // While crashed, panel-local keys are dropped (never re-invoked).
        app.handle_panel_local_key(PanelKeyEvent {
            code: KeyCode::Char('p'),
            modifiers: no_mods,
        });
        assert!(app.crashed[0]);

        // RetryPanel resets the panel and clears the crashed flag.
        app.handle_key(Action::RetryPanel);
        assert!(!app.crashed[0], "retry must recover a resettable panel");

        // RetryPanel on a non-crashed panel is a no-op.
        app.handle_key(Action::RetryPanel);
        assert!(!app.crashed[0]);

        std::panic::set_hook(previous_hook);
    }
}
