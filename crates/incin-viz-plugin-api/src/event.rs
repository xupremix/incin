//! Thin newtypes over the crossterm event shape panels need, per
//! UI-SPEC.md's "incin-viz-plugin-api does not depend on crossterm"
//! constraint.

/// A subset of key-press variants panels need to handle. Kept minimal for
/// this phase's keybinding set; extend as needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCode {
    /// Printable character key.
    Char(char),
    /// Tab key.
    Tab,
    /// Shift+Tab (BackTab) key.
    BackTab,
    /// Escape key.
    Esc,
    /// Enter/Return key.
    Enter,
    /// Up arrow key.
    Up,
    /// Down arrow key.
    Down,
    /// Left arrow key.
    Left,
    /// Right arrow key.
    Right,
}

/// Modifier keys held alongside a [`KeyCode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyModifiers {
    /// Control key indicator.
    pub ctrl: bool,
    /// Shift key indicator.
    pub shift: bool,
    /// Alt/Option key indicator.
    pub alt: bool,
}

/// A single key-press event routed to a panel or the host keymap resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelKeyEvent {
    /// Target key code.
    pub code: KeyCode,
    /// Active key modifiers.
    pub modifiers: KeyModifiers,
}

/// Subset of mouse events panels need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelMouseEvent {
    /// Mouse button press at (x, y).
    Down {
        x: u16,
        y: u16,
        modifiers: KeyModifiers,
    },
    /// Mouse button release at (x, y).
    Up {
        x: u16,
        y: u16,
        modifiers: KeyModifiers,
    },
    /// Mouse drag movement at (x, y).
    Drag {
        x: u16,
        y: u16,
        modifiers: KeyModifiers,
    },
    /// Mouse scroll down event.
    ScrollDown {
        x: u16,
        y: u16,
        modifiers: KeyModifiers,
    },
    /// Mouse scroll up event.
    ScrollUp {
        x: u16,
        y: u16,
        modifiers: KeyModifiers,
    },
}

/// Events a panel may receive via `Panel::handle_event`. Kept distinct from
/// telemetry `Event`s (which arrive via `Panel::update`) per the "how does
/// a plugin request a re-render only when new data arrives" question
/// PITFALLS.md Pitfall 6 flags as needing resolution before the trait is
/// considered proven.
#[derive(Debug, Clone)]
pub enum PanelEvent {
    /// Keyboard interaction event.
    Key(PanelKeyEvent),
    /// Mouse interaction event.
    Mouse(PanelMouseEvent),
}
