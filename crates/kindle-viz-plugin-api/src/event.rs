//! Thin newtypes over the crossterm event shape panels need, per
//! UI-SPEC.md's "kindle-viz-plugin-api does not depend on crossterm"
//! constraint.

/// A subset of key-press variants panels need to handle. Kept minimal for
/// this phase's keybinding set; extend as needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCode {
    /// Auto-generated documentation for Char.
    Char(char),
    /// Auto-generated documentation for Tab.
    Tab,
    /// Auto-generated documentation for BackTab.
    BackTab,
    /// Auto-generated documentation for Esc.
    Esc,
    /// Auto-generated documentation for Enter.
    Enter,
    /// Auto-generated documentation for Up.
    Up,
    /// Auto-generated documentation for Down.
    Down,
    /// Auto-generated documentation for Left.
    Left,
    /// Auto-generated documentation for Right.
    Right,
}

/// Modifier keys held alongside a [`KeyCode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyModifiers {
    /// Auto-generated documentation for ctrl.
    pub ctrl: bool,
    /// Auto-generated documentation for shift.
    pub shift: bool,
    /// Auto-generated documentation for alt.
    pub alt: bool,
}

/// A single key-press event routed to a panel or the host keymap resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelKeyEvent {
    /// Auto-generated documentation for code.
    pub code: KeyCode,
    /// Auto-generated documentation for modifiers.
    pub modifiers: KeyModifiers,
}

/// Subset of mouse events panels need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelMouseEvent {
    /// Auto-generated documentation for Down.
    Down {
        x: u16,
        y: u16,
        modifiers: KeyModifiers,
    },
    /// Auto-generated documentation for Up.
    Up {
        x: u16,
        y: u16,
        modifiers: KeyModifiers,
    },
    /// Auto-generated documentation for Drag.
    Drag {
        x: u16,
        y: u16,
        modifiers: KeyModifiers,
    },
    /// Auto-generated documentation for ScrollDown.
    ScrollDown {
        x: u16,
        y: u16,
        modifiers: KeyModifiers,
    },
    /// Auto-generated documentation for ScrollUp.
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
    /// Auto-generated documentation for Key.
    Key(PanelKeyEvent),
    /// Auto-generated documentation for Mouse.
    Mouse(PanelMouseEvent),
}
