//! Centralized error type for `incin-viz-plugin-api`, mirroring
//! `incin-telemetry`'s `err.rs` convention. This crate has no JSON or
//! socket I/O of its own, so its variant set is smaller.

use core::fmt::Debug;

/// Convenience type alias for `Result<T, Error>`.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
/// Plugin API error types.
pub enum Error {
    #[error("I/O error: {0}")]
    /// Underlying I/O error.
    Io(#[from] std::io::Error),

    #[error("Generic Message: {0}")]
    /// Generic message string error.
    Msg(String),
}

impl Debug for Error {
    /// Format error using Display representation.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
