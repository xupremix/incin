//! Thin newtypes over the crossterm event shape panels need, per
//! UI-SPEC.md's "kindle-viz-plugin-api does not depend on crossterm"
//! constraint.

/// A subset of key-press variants panels need to handle. Kept minimal for
/// this phase's keybinding set; extend as needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCode {
    Char(char),
    Tab,
    BackTab,
    Esc,
    Enter,
    Up,
    Down,
    Left,
    Right,
}

/// Modifier keys held alongside a [`KeyCode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyModifiers {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

/// A single key-press event routed to a panel or the host keymap resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelKeyEvent {
    pub code: KeyCode,
    pub modifiers: KeyModifiers,
}

/// Subset of mouse events panels need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelMouseEvent {
    Down { x: u16, y: u16, modifiers: KeyModifiers },
    Up { x: u16, y: u16, modifiers: KeyModifiers },
    Drag { x: u16, y: u16, modifiers: KeyModifiers },
    ScrollDown { x: u16, y: u16, modifiers: KeyModifiers },
    ScrollUp { x: u16, y: u16, modifiers: KeyModifiers },
}

/// Events a panel may receive via `Panel::handle_event`. Kept distinct from
/// telemetry `Event`s (which arrive via `Panel::update`) per the "how does
/// a plugin request a re-render only when new data arrives" question
/// PITFALLS.md Pitfall 6 flags as needing resolution before the trait is
/// considered proven.
#[derive(Debug, Clone)]
pub enum PanelEvent {
    Key(PanelKeyEvent),
    Mouse(PanelMouseEvent),
}
