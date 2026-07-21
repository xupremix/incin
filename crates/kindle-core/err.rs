use crate::candle;
use alloc::string::String;
use core::fmt::Debug;

/// Auto-generated documentation for Result.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
/// Auto-generated documentation for Error.
pub enum Error {
    #[error(transparent)]
    /// Auto-generated documentation for Candle.
    Candle(#[from] candle::Error),
    #[error("Invalid device provided: expected {expected}, got {got}")]
    /// Auto-generated documentation for DeviceInitializationError.
    DeviceInitializationError { expected: String, got: String },
    #[error("Shape mismatch: expected {expected}, got {got}")]
    /// Auto-generated documentation for ShapeMismatch.
    ShapeMismatch { expected: String, got: String },
    #[error("DType mismatch: expected {expected}, got {got}")]
    /// Auto-generated documentation for DTypeMismatch.
    DTypeMismatch { expected: String, got: String },
}

impl Debug for Error {
    /// Auto-generated documentation for fmt.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
