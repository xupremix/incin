//! Keybinding resolution contract.

use crate::event::PanelKeyEvent;

/// An action a resolved keybinding maps to. Minimal set for this phase's
/// hardcoded default keymap; Phase 10 extends this and wires
/// `KeymapProvider` to a configurable/vim-swappable system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Quit,
    FocusNext,
    FocusPrev,
    RetryPanel,
    /// Delegates to the focused panel's own `handle_event`.
    PanelLocal,
}

/// Resolves a raw key event to a semantic `Action`. This phase ships
/// exactly one hardcoded implementation wired directly into `kindle-viz`
/// (Plan 08-04); the trait exists now so Phase 10's vim-keymap plugin has
/// a stable contract.
pub trait KeymapProvider {
    fn resolve(&self, key: PanelKeyEvent) -> Option<Action>;
}
