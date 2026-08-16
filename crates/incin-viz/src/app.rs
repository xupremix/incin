//! `incin-viz`'s application state and `tokio::select!`-driven async event
//! loop, multiplexing three streams: terminal input
//! (`crossterm::event::EventStream`), transport-tail polling, and a fixed
//! render tick.
//!
//! Source: `.planning/phases/08-plugin-api-base-tui-shell/08-RESEARCH.md`
//! Pattern 3 (the full `tokio::select!` event loop) -- adapted to this
//! plan's empty-panel-registry skeleton. Plan 08-05 registers the real
//! loss/panic-test panels via [`App::register_panel`](crate::app::App::register_panel).

use std::time::Duration;

use crossterm::event::{Event as CrosstermEvent, EventStream, KeyCode as CtKeyCode, KeyEvent};
use futures_util::StreamExt;
use incin_viz_plugin_api::event::{KeyCode, KeyModifiers, PanelEvent, PanelKeyEvent};
use incin_viz_plugin_api::keymap::{Action, KeymapProvider};
use incin_viz_plugin_api::panel::Panel;
use incin_viz_plugin_api::render_ctx::RenderCtx;
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
const FOOTER_HINTS: &str =
    "q: quit  Tab: focus next  l: toggle layout  f: fullscreen  p: trigger panic";

/// Exact UI-SPEC.md crashed-panel placeholder copy (Red/Bold, rendered
const CRASHED_PLACEHOLDER: &str = "⚠ panel crashed; press r to retry";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Layout mode.
pub enum LayoutMode {
    /// Grid.
    Grid,
    /// Maximized.
    Maximized,
}

/// The host application: panel registry, per-panel crash state, focus, and
/// the transport being tailed.
pub struct App {
    /// Registered panels. Empty this plan -- Plan 08-05 populates it via
    /// [`App::register_panel`](crate::app::App::register_panel).
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
    layout_mode: LayoutMode,
    /// Log of conflict warnings (panel-local vs active keymap).
    pub conflicts: Vec<String>,
    /// Global hover text dynamically populated by hit-testing.
    pub hover_text: Option<String>,
    /// Panel areas from the last render pass, mapping panel index to its rendered area.
    last_panel_areas: Vec<(usize, ratatui::layout::Rect)>,
    /// Hit regions from the last render pass, with the panel index attached.
    last_hit_regions: Vec<(
        ratatui::layout::Rect,
        incin_viz_plugin_api::render_ctx::HitId,
        usize,
    )>,
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
            layout_mode: LayoutMode::Grid,
            conflicts: Vec::new(),
            hover_text: None,
            last_panel_areas: Vec::new(),
            last_hit_regions: Vec::new(),
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
            Action::FocusNext | Action::FocusRight => {
                self.focused = (self.focused + 1) % self.panels.len().max(1);
            }
            Action::FocusPrev | Action::FocusLeft => {
                let len = self.panels.len().max(1);
                self.focused = (self.focused + len - 1) % len;
            }
            Action::FocusUp => {
                if self.layout_mode == LayoutMode::Grid {
                    let len = self.panels.len();
                    if len > 0 {
                        let col = self.focused % 3;
                        let row = self.focused / 3;
                        if row > 0 {
                            self.focused = (row - 1) * 3 + col;
                        }
                    }
                } else {
                    let len = self.panels.len().max(1);
                    self.focused = (self.focused + len - 1) % len;
                }
            }
            Action::FocusDown => {
                if self.layout_mode == LayoutMode::Grid {
                    let len = self.panels.len();
                    if len > 0 {
                        let col = self.focused % 3;
                        let row = self.focused / 3;
                        let next = (row + 1) * 3 + col;
                        if next < len {
                            self.focused = next;
                        } else {
                            // If straight down is out of bounds, go to the very last panel
                            self.focused = len - 1;
                        }
                    }
                } else {
                    self.focused = (self.focused + 1) % self.panels.len().max(1);
                }
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
            Action::ToggleLayout => {
                self.layout_mode = match self.layout_mode {
                    LayoutMode::Grid => LayoutMode::Maximized,
                    LayoutMode::Maximized => LayoutMode::Grid,
                };
            }
            // PanelLocal is routed directly via `handle_panel_local_key`
            // from the event loop, never through the keymap resolver.
            Action::PanelLocal => {}
        }
    }

    /// Routes an event to the focused panel (panic-contained). Returns whether the panel consumed it.
    pub fn handle_panel_local_event(&mut self, event: PanelEvent) -> bool {
        if self.panels.is_empty() || self.crashed[self.focused] {
            return false;
        }
        match dispatch::dispatch_handle_event(self.panels[self.focused].as_mut(), &event) {
            DispatchOutcome::Ok(consumed) => consumed,
            DispatchOutcome::Panicked => {
                self.crashed[self.focused] = true;
                false
            }
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

        let header = Paragraph::new(format!("incin-viz; run: {}", self.run_id_or_path));
        frame.render_widget(header, rows[0]);

        let footer =
            Paragraph::new(FOOTER_HINTS).style(Style::default().add_modifier(Modifier::DIM));
        frame.render_widget(footer, rows[2]);

        if self.panels.is_empty() {
            return;
        }

        let mut panel_areas = Vec::new();
        match self.layout_mode {
            LayoutMode::Grid => {
                let row_constraints = vec![Constraint::Ratio(1, 3); 3];
                let grid_rows = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints(row_constraints)
                    .split(rows[1]);

                let col_constraints = vec![Constraint::Ratio(1, 3); 3];
                for r in 0..3 {
                    let cols = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints(col_constraints.clone())
                        .split(grid_rows[r as usize]);
                    for c in 0..3 {
                        panel_areas.push(cols[c as usize]);
                    }
                }
            }
            LayoutMode::Maximized => {
                panel_areas.push(rows[1]);
            }
        }

        let mut panel_hit_regions = Vec::new();
        let mut alerts = Vec::new();

        self.last_hit_regions.clear();
        self.last_panel_areas.clear();

        for (i, panel) in self.panels.iter_mut().enumerate() {
            if self.layout_mode == LayoutMode::Maximized && i != self.focused {
                continue;
            }
            let area_idx = if self.layout_mode == LayoutMode::Maximized {
                0
            } else {
                i
            };

            if area_idx >= panel_areas.len() {
                break; // Beyond allocated grid or scroll area
            }

            // Re-draw only the border/title chrome over the panel's own
            // block so the focus color rule holds without the Panel trait
            // needing a focus parameter (Block leaves inner content intact).
            let focus_color = if i == self.focused {
                Color::White
            } else {
                Color::DarkGray
            };

            let panel_title = panel.title().to_string();

            if self.crashed[i] {
                // Placeholder without re-invoking the panicked panel until
                // the user explicitly retries ('r') -- border/title intact.
                let focus_block = Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(focus_color))
                    .title(Span::styled(
                        panel_title,
                        Style::default()
                            .fg(focus_color)
                            .add_modifier(Modifier::BOLD),
                    ));
                let placeholder = Paragraph::new(CRASHED_PLACEHOLDER)
                    .style(Style::default().fg(Color::Red).add_modifier(Modifier::BOLD))
                    .alignment(Alignment::Center)
                    .block(focus_block);
                frame.render_widget(placeholder, panel_areas[area_idx]);
                continue;
            }

            panel_hit_regions.clear();
            let mut ctx = RenderCtx::new(frame, panel_areas[area_idx], &mut panel_hit_regions);
            if let DispatchOutcome::Panicked = dispatch::dispatch_render(panel.as_mut(), &mut ctx) {
                self.crashed[i] = true;
                continue;
            }
            let alert = ctx.take_alert();
            drop(ctx);

            for (rect, id) in &panel_hit_regions {
                self.last_hit_regions.push((*rect, *id, i));
            }

            let mut title_style = Style::default()
                .fg(focus_color)
                .add_modifier(Modifier::BOLD);
            let mut border_style = Style::default().fg(focus_color);

            if let Some(msg) = alert {
                alerts.push(format!("{}: {}", panel_title, msg));
                border_style = border_style.fg(Color::Red).add_modifier(Modifier::BOLD);
                title_style = title_style.fg(Color::Red);
            }

            let focus_block = Block::default()
                .borders(Borders::ALL)
                .border_style(border_style)
                .title(Span::styled(panel_title, title_style));

            frame.render_widget(focus_block, panel_areas[area_idx]);
            self.last_panel_areas.push((i, panel_areas[area_idx]));
        }

        let footer_text = if let Some(hover) = self.hover_text.take() {
            hover
        } else if !alerts.is_empty() {
            format!("ALERTS: {}", alerts.join(" | "))
        } else if !self.conflicts.is_empty() {
            format!("CONFLICTS: {}", self.conflicts.join(" | "))
        } else {
            FOOTER_HINTS.to_string()
        };

        let footer_style = if !alerts.is_empty() || !self.conflicts.is_empty() {
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
        } else {
            Style::default().add_modifier(Modifier::DIM)
        };

        let footer = Paragraph::new(footer_text).style(footer_style);
        frame.render_widget(footer, rows[2]);
    }
}

/// The hardcoded default keymap for this phase (UI-SPEC.md Interaction &
/// Keybinding table). Phase 10 replaces this with a configurable/
/// vim-swappable `KeymapProvider` system using the same trait.
pub struct DefaultKeymap;

impl KeymapProvider for DefaultKeymap {
    /// Resolve.
    fn resolve(&self, key: PanelKeyEvent) -> Option<Action> {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => Some(Action::Quit),
            KeyCode::Char('c') if key.modifiers.ctrl => Some(Action::Quit),
            KeyCode::Tab => Some(Action::FocusNext),
            KeyCode::BackTab => Some(Action::FocusPrev),
            KeyCode::Up => Some(Action::FocusUp),
            KeyCode::Down => Some(Action::FocusDown),
            KeyCode::Left => Some(Action::FocusLeft),
            KeyCode::Right => Some(Action::FocusRight),
            KeyCode::Char('r') => Some(Action::RetryPanel),
            KeyCode::Char('l') | KeyCode::Char('f') | KeyCode::Enter => Some(Action::ToggleLayout),
            _ => None,
        }
    }
}

/// Vim-style keymap replacing arrows/tabs with hjkl equivalents.
pub struct VimKeymap;

impl KeymapProvider for VimKeymap {
    /// Resolve.
    fn resolve(&self, key: PanelKeyEvent) -> Option<Action> {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => Some(Action::Quit),
            KeyCode::Char('c') if key.modifiers.ctrl => Some(Action::Quit),
            KeyCode::Char('j') => Some(Action::FocusDown),
            KeyCode::Char('k') => Some(Action::FocusUp),
            KeyCode::Char('h') => Some(Action::FocusLeft),
            KeyCode::Char('l') => Some(Action::FocusRight),
            KeyCode::Char('r') => Some(Action::RetryPanel),
            KeyCode::Char('t') | KeyCode::Char('f') | KeyCode::Enter => Some(Action::ToggleLayout),
            _ => None,
        }
    }
}

/// Converts a crossterm key event into the plugin-api's crossterm-free
/// `PanelKeyEvent` newtype. Unmapped key codes return `None` (ignored).
fn convert_key(key: KeyEvent) -> Option<PanelKeyEvent> {
    if key.kind != crossterm::event::KeyEventKind::Press {
        return None;
    }
    let code = match key.code {
        CtKeyCode::Char(c) => KeyCode::Char(c),
        CtKeyCode::Tab => KeyCode::Tab,
        CtKeyCode::BackTab => KeyCode::BackTab,
        CtKeyCode::Esc => KeyCode::Esc,
        CtKeyCode::Enter => KeyCode::Enter,
        CtKeyCode::Up => KeyCode::Up,
        CtKeyCode::Down => KeyCode::Down,
        CtKeyCode::Left => KeyCode::Left,
        CtKeyCode::Right => KeyCode::Right,
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

/// Converts a crossterm mouse event into the plugin-api's `PanelMouseEvent`.
fn convert_mouse(
    mouse: crossterm::event::MouseEvent,
) -> Option<incin_viz_plugin_api::event::PanelMouseEvent> {
    use crossterm::event::MouseEventKind;
    use incin_viz_plugin_api::event::{KeyModifiers, PanelMouseEvent};
    let modifiers = KeyModifiers {
        ctrl: mouse
            .modifiers
            .contains(crossterm::event::KeyModifiers::CONTROL),
        shift: mouse
            .modifiers
            .contains(crossterm::event::KeyModifiers::SHIFT),
        alt: mouse
            .modifiers
            .contains(crossterm::event::KeyModifiers::ALT),
    };
    match mouse.kind {
        MouseEventKind::Down(_) => Some(PanelMouseEvent::Down {
            x: mouse.column,
            y: mouse.row,
            modifiers,
        }),
        MouseEventKind::Up(_) => Some(PanelMouseEvent::Up {
            x: mouse.column,
            y: mouse.row,
            modifiers,
        }),
        MouseEventKind::Drag(_) => Some(PanelMouseEvent::Drag {
            x: mouse.column,
            y: mouse.row,
            modifiers,
        }),
        MouseEventKind::ScrollDown => Some(PanelMouseEvent::ScrollDown {
            x: mouse.column,
            y: mouse.row,
            modifiers,
        }),
        MouseEventKind::ScrollUp => Some(PanelMouseEvent::ScrollUp {
            x: mouse.column,
            y: mouse.row,
            modifiers,
        }),
        MouseEventKind::Moved | MouseEventKind::ScrollLeft | MouseEventKind::ScrollRight => None,
    }
}

/// Runs the event loop until quit: multiplexes terminal input, transport
/// polling, and the fixed render tick -- no branch blocks the others.
pub async fn run(
    mut app: App,
    mut terminal: ratatui::DefaultTerminal,
    keymap: Box<dyn KeymapProvider>,
) -> anyhow::Result<()> {
    let mut term_events = EventStream::new();
    let mut render_interval = interval(RENDER_TICK);
    let mut transport_interval = interval(TRANSPORT_POLL);

    loop {
        tokio::select! {
            maybe_event = term_events.next() => {
                match maybe_event {
                    Some(Ok(CrosstermEvent::Key(key))) => {
                        if let Some(panel_key) = convert_key(key) {
                            let consumed = app.handle_panel_local_event(PanelEvent::Key(panel_key));
                            if let Some(action) = keymap.resolve(panel_key) {
                                if consumed {
                                    app.conflicts.push(format!("Panel {} intercepted {:?}", app.panels[app.focused].title(), panel_key.code));
                                } else {
                                    app.handle_key(action);
                                    if app.should_quit {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Some(Ok(CrosstermEvent::Mouse(mouse))) => {
                        use crossterm::event::MouseEventKind;
                        // Determine which panel was hovered/clicked
                        if let MouseEventKind::Moved = mouse.kind {
                            app.hover_text = None;
                            for (rect, id, panel_idx) in &app.last_hit_regions {
                                if mouse.column >= rect.x && mouse.column < rect.x + rect.width &&
                                   mouse.row >= rect.y && mouse.row < rect.y + rect.height {
                                   if let Some(text) = app.panels[*panel_idx].hover_text(*id) {
                                       app.hover_text = Some(text);
                                   }
                                   break;
                                }
                            }
                        } else if let MouseEventKind::Down(_) = mouse.kind {
                            for &(i, area) in &app.last_panel_areas {
                                if mouse.column >= area.x && mouse.column < area.x + area.width &&
                                   mouse.row >= area.y && mouse.row < area.y + area.height {
                                   app.focused = i;
                                   break;
                                }
                            }
                        }

                        if let Some(panel_mouse) = convert_mouse(mouse) {
                            let _consumed = app.handle_panel_local_event(PanelEvent::Mouse(panel_mouse));
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
/// Tests.
mod tests {
    use super::*;

    #[test]
    /// Default keymap resolves quit and focus keys.
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
        /// Id.
        fn id(&self) -> &'static str {
            "crash-on-p"
        }
        /// Title.
        fn title(&self) -> &str {
            "Crash On P"
        }
        /// Update.
        fn update(&mut self, _event: &incin_telemetry::events::Event) {}
        /// Render.
        fn render(&mut self, _ctx: &mut RenderCtx<'_, '_>) {}
        /// Handle event.
        fn handle_event(&mut self, event: &PanelEvent) -> bool {
            if let PanelEvent::Key(k) = event
                && k.code == KeyCode::Char('p')
            {
                panic!("deliberate test panic");
            }
            false
        }
        /// Reset.
        fn reset(&mut self) {}
    }

    /// Noop transport.
    struct NoopTransport;

    impl crate::transport_reader::TransportReader for NoopTransport {
        /// Poll new events.
        fn poll_new_events(&mut self) -> crate::err::Result<Vec<incin_telemetry::events::Event>> {
            Ok(Vec::new())
        }
    }

    #[test]
    /// Panel local panic marks crashed and retry recovers.
    fn panel_local_panic_marks_crashed_and_retry_recovers() {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let mut app = App::new(Box::new(NoopTransport), String::from("test"));
        app.register_panel(Box::new(CrashOnP));
        let no_mods = KeyModifiers {
            ctrl: false,
            shift: false,
            alt: false,
        };

        // 'p' routed panel-locally panics the panel -> marked crashed.
        app.handle_panel_local_event(PanelEvent::Key(PanelKeyEvent {
            code: KeyCode::Char('p'),
            modifiers: no_mods,
        }));
        assert!(app.crashed[0], "panicking handle_event must mark crashed");

        // While crashed, panel-local keys are dropped (never re-invoked).
        app.handle_panel_local_event(PanelEvent::Key(PanelKeyEvent {
            code: KeyCode::Char('p'),
            modifiers: no_mods,
        }));
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
