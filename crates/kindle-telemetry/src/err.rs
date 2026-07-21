//! Centralized error type for `kindle-telemetry`, mirroring
//! `kindle-core`'s `err.rs` convention. This is a `std` crate (no
//! `no_std`/`alloc` constraint), so `std::` paths are used directly.

use core::fmt::Debug;

/// Convenience type alias for `Result<T, Error>`.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
/// Telemetry system errors.
pub enum Error {
    #[error("I/O error: {0}")]
    /// Underlying file or network I/O error.
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    /// Event JSON serialization error.
    Serialize(#[from] serde_json::Error),

    #[error("Socket bind/connect error: {0}")]
    /// IPC socket connection or bind failure.
    Socket(String),

    #[error("Generic Message: {0}")]
    /// Generic error message string.
    Msg(String),
}

impl Debug for Error {
    /// Format error using Display representation.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_formatting() {
        let err = Error::Socket("bind failed".to_string());
        let formatted = format!("{err}");
        assert_eq!(formatted, "Socket bind/connect error: bind failed");

        let err_msg = Error::Msg("generic failure".to_string());
        let formatted = format!("{err_msg}");
        assert_eq!(formatted, "Generic Message: generic failure");
    }
}
