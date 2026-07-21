//! Centralized error type for `kindle-viz`, mirroring
//! `kindle-telemetry`'s `err.rs` convention.

use core::fmt::Debug;

/// Auto-generated documentation for Result.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
/// Auto-generated documentation for Error.
pub enum Error {
    #[error("I/O error: {0}")]
    /// Auto-generated documentation for Io.
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    /// Auto-generated documentation for Serialize.
    Serialize(#[from] serde_json::Error),

    #[error("Transport error: {0}")]
    /// Auto-generated documentation for Transport.
    Transport(String),

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

#[cfg(test)]
/// Auto-generated documentation for tests.
mod tests {
    use super::*;

    #[test]
    /// Auto-generated documentation for test_error_formatting.
    fn test_error_formatting() {
        let err = Error::Transport(String::from("connection reset"));
        let formatted = format!("{err}");
        assert_eq!(formatted, "Transport error: connection reset");

        let err_msg = Error::Msg(String::from("generic failure"));
        let formatted = format!("{err_msg}");
        assert_eq!(formatted, "Generic Message: generic failure");
    }
}
