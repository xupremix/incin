//! `Transport`: a sink that durably records one `Event` at a time.
//! Implementors own their own I/O; `write_event` must never block the
//! caller indefinitely (the future `emitter.rs`'s writer thread is the
//! only caller — Plan 07-02).

/// Auto-generated documentation for file.
pub mod file;
/// The `socket` module.
pub mod socket;

use crate::events::Event;

/// A sink that durably records one [`Event`] at a time. `Send` is required
/// because `Emitter::new` moves `Vec<Box<dyn Transport>>` into the
/// exclusively-owning background writer thread (Plan 07-02).
pub trait Transport: Send {
    /// Auto-generated documentation for write_event.
    fn write_event(&mut self, event: &Event) -> crate::err::Result<()>;
}
