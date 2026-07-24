use crate::candle;
use alloc::string::String;
use core::fmt::Debug;

/// Result.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
/// Error.
pub enum Error {
    #[error(transparent)]
    /// Candle.
    Candle(#[from] candle::Error),
    #[error("Invalid device provided: expected {expected}, got {got}")]
    /// Device initialization error.
    DeviceInitializationError { expected: String, got: String },
    #[error("Shape mismatch: expected {expected}, got {got}")]
    /// Shape mismatch.
    ShapeMismatch { expected: String, got: String },
    #[error("DType mismatch: expected {expected}, got {got}")]
    /// Dtype mismatch.
    DTypeMismatch { expected: String, got: String },
}

impl Debug for Error {
    /// Fmt.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
