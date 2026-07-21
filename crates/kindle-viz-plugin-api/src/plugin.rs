//! A plugin contributes one or more panels to the host.

use crate::panel::Panel;

/// Constructs this plugin's panel(s). Called once at registry-build time,
/// not per-frame -- panels are long-lived, not reconstructed every render.
/// Both first-party built-in panels and third-party plugins implement this
/// identically: no privileged internal API.
pub trait Plugin {
    /// Auto-generated documentation for panels.
    fn panels(&self) -> Vec<Box<dyn Panel>>;
}
