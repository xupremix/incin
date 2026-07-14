//! Centralized error type for `kindle-viz`, mirroring
//! `kindle-telemetry`'s `err.rs` convention.

use core::fmt::Debug;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Generic Message: {0}")]
    Msg(String),
}

impl Debug for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_formatting() {
        let err = Error::Transport(String::from("connection reset"));
        let formatted = format!("{err}");
        assert_eq!(formatted, "Transport error: connection reset");

        let err_msg = Error::Msg(String::from("generic failure"));
        let formatted = format!("{err_msg}");
        assert_eq!(formatted, "Generic Message: generic failure");
    }
}
