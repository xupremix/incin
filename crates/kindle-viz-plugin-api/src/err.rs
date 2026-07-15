//! Centralized error type for `kindle-viz-plugin-api`, mirroring
//! `kindle-telemetry`'s `err.rs` convention. This crate has no JSON or
//! socket I/O of its own, so its variant set is smaller.

use core::fmt::Debug;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Generic Message: {0}")]
    Msg(String),
}

impl Debug for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
