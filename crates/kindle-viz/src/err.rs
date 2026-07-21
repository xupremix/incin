//! Centralized error type for `kindle-viz`, mirroring
//! `kindle-telemetry`'s `err.rs` convention.

use core::fmt::Debug;

/// Core abstraction for `Result` within the Kindle framework.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
/// Core abstraction for `Error` within the Kindle framework.
pub enum Error {
    #[error("I/O error: {0}")]
    /// Core abstraction for `Io` within the Kindle framework.
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    /// Core abstraction for `Serialize` within the Kindle framework.
    Serialize(#[from] serde_json::Error),

    #[error("Transport error: {0}")]
    /// Core abstraction for `Transport` within the Kindle framework.
    Transport(String),

    #[error("Generic Message: {0}")]
    /// Core abstraction for `Msg` within the Kindle framework.
    Msg(String),
}

impl Debug for Error {
    /// Core abstraction for `fmt` within the Kindle framework.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
/// Core abstraction for `tests` within the Kindle framework.
mod tests {
    use super::*;

    #[test]
    /// Core abstraction for `test_error_formatting` within the Kindle framework.
    fn test_error_formatting() {
        let err = Error::Transport(String::from("connection reset"));
        let formatted = format!("{err}");
        assert_eq!(formatted, "Transport error: connection reset");

        let err_msg = Error::Msg(String::from("generic failure"));
        let formatted = format!("{err_msg}");
        assert_eq!(formatted, "Generic Message: generic failure");
    }
}
