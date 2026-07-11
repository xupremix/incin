//! `kindle-viz-plugin-api`: the stable, independently-compilable trait
//! surface (`Panel`, `Plugin`, `KeymapProvider`) plus a `RenderCtx`
//! rendering-context abstraction that wraps `ratatui::Frame`/`Rect`
//! directly. Depends only on `kindle-telemetry` (wire types) and
//! `ratatui-core`/`ratatui-widgets` -- never on `crossterm` or `kindle-viz`.

pub mod err;
pub mod event;
pub mod keymap;
pub mod panel;
pub mod plugin;
pub mod render_ctx;

pub mod prelude {
    pub use crate::err::{Error, Result};
    pub use crate::event::*;
    pub use crate::keymap::{Action, KeymapProvider};
    pub use crate::panel::Panel;
    pub use crate::plugin::Plugin;
    pub use crate::render_ctx::{HitId, RenderCtx};
}
