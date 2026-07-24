//! Centralized error type for `incin-viz`, mirroring
//! `incin-telemetry`'s `err.rs` convention.

use core::fmt::Debug;

/// Result.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
/// Error.
pub enum Error {
    #[error("I/O error: {0}")]
    /// Io.
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    /// Serialize.
    Serialize(#[from] serde_json::Error),

    #[error("Transport error: {0}")]
    /// Transport.
    Transport(String),

    #[error("Generic Message: {0}")]
    /// Msg.
    Msg(String),
}

impl Debug for Error {
    /// Fmt.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
/// Tests.
mod tests {
    use super::*;

    #[test]
    /// Test error formatting.
    fn test_error_formatting() {
        let err = Error::Transport(String::from("connection reset"));
        let formatted = format!("{err}");
        assert_eq!(formatted, "Transport error: connection reset");

        let err_msg = Error::Msg(String::from("generic failure"));
        let formatted = format!("{err_msg}");
        assert_eq!(formatted, "Generic Message: generic failure");
    }
}
