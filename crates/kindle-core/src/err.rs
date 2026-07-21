use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Debug;

/// Auto-generated documentation for Result.
pub type Result<T> = core::result::Result<T, Error>;

#[non_exhaustive]
#[derive(thiserror::Error)]
/// Auto-generated documentation for Error.
pub enum Error {
    #[error("Shape mismatch during '{op}': expected {expected:?}, got {got:?}. {msg}")]
    /// Auto-generated documentation for ShapeMismatch.
    ShapeMismatch {
        /// op
        op: &'static str,
        /// expected
        expected: Vec<usize>,
        /// got
        got: Vec<usize>,
        /// msg
        msg: String,
    },

    #[error("Out of Memory error on device: {device}")]
    /// Auto-generated documentation for OutOfMemory.
    OutOfMemory {
        /// device
        device: String,
    },

    #[error("Operation '{op}' is not supported by backend '{backend}'")]
    /// Auto-generated documentation for UnsupportedBackendOperation.
    UnsupportedBackendOperation {
        /// op
        op: &'static str,
        /// backend
        backend: &'static str,
    },

    #[error("Invalid device provided: expected {expected}, got {got}")]
    /// Auto-generated documentation for DeviceInitializationError.
    DeviceInitializationError {
        /// expected
        expected: String,
        /// got
        got: String,
    },

    #[error("Internal Backend Failure: {0}")]
    /// Auto-generated documentation for BackendFailure.
    BackendFailure(#[from] anyhow::Error),

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
        let err = Error::OutOfMemory {
            device: "CUDA:0".to_string(),
        };
        let formatted = alloc::format!("{}", err);
        assert_eq!(formatted, "Out of Memory error on device: CUDA:0");

        let err_unsupported = Error::UnsupportedBackendOperation {
            op: "matmul",
            backend: "Ndarray",
        };
        let formatted = alloc::format!("{}", err_unsupported);
        assert_eq!(
            formatted,
            "Operation 'matmul' is not supported by backend 'Ndarray'"
        );
    }
}
