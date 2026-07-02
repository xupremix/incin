use crate::candle;
use alloc::string::String;
use core::fmt::Debug;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle::Error),
    #[error("Invalid device provided: expected {expected}, got {got}")]
    DeviceInitializationError { expected: String, got: String },
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },
    #[error("DType mismatch: expected {expected}, got {got}")]
    DTypeMismatch { expected: String, got: String },
}

impl Debug for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
