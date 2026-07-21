//! Centralized error type for `kindle-viz-plugin-api`, mirroring
//! `kindle-telemetry`'s `err.rs` convention. This crate has no JSON or
//! socket I/O of its own, so its variant set is smaller.

use core::fmt::Debug;

/// Auto-generated documentation for Result.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
/// Auto-generated documentation for Error.
pub enum Error {
    #[error("I/O error: {0}")]
    /// Auto-generated documentation for Io.
    Io(#[from] std::io::Error),

    #[error("Generic Message: {0}")]
    /// Auto-generated documentation for Msg.
    Msg(String),
}

impl Debug for Error {
    /// Auto-generated documentation for fmt.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
