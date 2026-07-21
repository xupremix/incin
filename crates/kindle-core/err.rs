use crate::candle;
use alloc::string::String;
use core::fmt::Debug;

/// Core abstraction for `Result` within the Kindle framework.
pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
/// Core abstraction for `Error` within the Kindle framework.
pub enum Error {
    #[error(transparent)]
    /// Core abstraction for `Candle` within the Kindle framework.
    Candle(#[from] candle::Error),
    #[error("Invalid device provided: expected {expected}, got {got}")]
    /// Core abstraction for `DeviceInitializationError` within the Kindle framework.
    DeviceInitializationError { expected: String, got: String },
    #[error("Shape mismatch: expected {expected}, got {got}")]
    /// Core abstraction for `ShapeMismatch` within the Kindle framework.
    ShapeMismatch { expected: String, got: String },
    #[error("DType mismatch: expected {expected}, got {got}")]
    /// Core abstraction for `DTypeMismatch` within the Kindle framework.
    DTypeMismatch { expected: String, got: String },
}

impl Debug for Error {
    /// Core abstraction for `fmt` within the Kindle framework.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
