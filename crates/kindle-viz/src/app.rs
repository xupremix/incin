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
use kindle_viz_plugin_api::event::{KeyCode, KeyModifiers, PanelKeyEvent};
use kindle_viz_plugin_api::keymap::{Action, KeymapProvider};
use kindle_viz_plugin_api::panel::Panel;
use kindle_viz_plugin_api::render_ctx::RenderCtx;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::widgets::Paragraph;
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
            // Plan 08-05 wires retry/panel-local dispatch once real panels
            // exist.
            Action::RetryPanel | Action::PanelLocal => {}
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
            if self.crashed[i] {
                // Render nothing further into a crashed panel's region this
                // plan -- Plan 08-05 adds the "⚠ panel crashed" placeholder.
                continue;
            }
            let mut ctx = RenderCtx::new(frame, panel_areas[i], &mut hit_regions);
            if let DispatchOutcome::Panicked = dispatch::dispatch_render(panel.as_mut(), &mut ctx) {
                self.crashed[i] = true;
            }
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
                        if let Some(panel_key) = convert_key(key)
                            && let Some(action) = keymap.resolve(panel_key)
                        {
                            app.handle_key(action);
                            if app.should_quit {
                                break;
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
                code: KeyCode::Char('x'),
                modifiers: no_mods
            }),
            None
        );
    }
}
