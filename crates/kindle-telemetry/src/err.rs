//! Centralized error type for `kindle-telemetry`, mirroring
//! `kindle-core`'s `err.rs` convention. This is a `std` crate (no
//! `no_std`/`alloc` constraint), so `std::` paths are used directly.

use std::fmt::Debug;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("Socket bind/connect error: {0}")]
    Socket(String),

    #[error("Generic Message: {0}")]
    Msg(String),
}

impl Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
