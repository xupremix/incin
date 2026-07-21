//! Keybinding resolution contract.

use crate::event::PanelKeyEvent;

/// An action a resolved keybinding maps to. Minimal set for this phase's
/// hardcoded default keymap; Phase 10 extends this and wires
/// `KeymapProvider` to a configurable/vim-swappable system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Auto-generated documentation for Quit.
    Quit,
    /// Auto-generated documentation for FocusNext.
    FocusNext,
    /// Auto-generated documentation for FocusPrev.
    FocusPrev,
    /// Auto-generated documentation for FocusUp.
    FocusUp,
    /// Auto-generated documentation for FocusDown.
    FocusDown,
    /// Auto-generated documentation for FocusLeft.
    FocusLeft,
    /// Auto-generated documentation for FocusRight.
    FocusRight,
    /// Auto-generated documentation for RetryPanel.
    RetryPanel,
    /// Auto-generated documentation for ToggleLayout.
    ToggleLayout,
    /// Delegates to the focused panel's own `handle_event`.
    PanelLocal,
}

/// Resolves a raw key event to a semantic `Action`. This phase ships
/// exactly one hardcoded implementation wired directly into `kindle-viz`
/// (Plan 08-04); the trait exists now so Phase 10's vim-keymap plugin has
/// a stable contract.
pub trait KeymapProvider {
    /// Auto-generated documentation for resolve.
    fn resolve(&self, key: PanelKeyEvent) -> Option<Action>;
}
