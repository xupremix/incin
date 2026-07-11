//! `Transport`: a sink that durably records one `Event` at a time.
//! Implementors own their own I/O; `write_event` must never block the
//! caller indefinitely (the future `emitter.rs`'s writer thread is the
//! only caller — Plan 07-02).

pub mod file;
pub mod socket;

use crate::events::Event;

/// A sink that durably records one [`Event`] at a time.
pub trait Transport {
    fn write_event(&mut self, event: &Event) -> crate::err::Result<()>;
}
