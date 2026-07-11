//! Read-side counterpart to `kindle-telemetry`'s `Transport` trait: a
//! poll-based tailer for the JSONL transport file a training process
//! writes to.

/// A source that yields newly-available events since the last poll.
/// Implementors own their own I/O and must never block indefinitely --
/// `poll_new_events` is called every event-loop tick from `app.rs`'s
/// `tokio::select!`, so a blocking implementation would stall the whole
/// render/input loop.
pub trait TransportReader: Send {
    /// Returns any newly-complete events discovered since the last call.
    /// A trailing incomplete line at EOF (the writer is mid-`write_all`)
    /// must NOT be returned yet -- it is buffered internally and completed
    /// on a future call once the rest of the line arrives.
    fn poll_new_events(&mut self) -> crate::err::Result<Vec<kindle_telemetry::events::Event>>;
}
