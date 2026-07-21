use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Debug;

/// Convenience type alias for `Result<T, Error>`.
pub type Result<T> = core::result::Result<T, Error>;

#[non_exhaustive]
#[derive(thiserror::Error)]
/// Central error enum for the Kindle framework.
pub enum Error {
    #[error("Shape mismatch during '{op}': expected {expected:?}, got {got:?}. {msg}")]
    /// Incompatible shape during tensor operation execution.
    ShapeMismatch {
        /// Name of the operation that failed.
        op: &'static str,
        /// Expected dimension array.
        expected: Vec<usize>,
        /// Actual dimension array.
        got: Vec<usize>,
        /// Context message.
        msg: String,
    },

    #[error("Out of Memory error on device: {device}")]
    /// Device out-of-memory allocation failure.
    OutOfMemory {
        /// Target device string identifier.
        device: String,
    },

    #[error("Operation '{op}' is not supported by backend '{backend}'")]
    /// Operation unimplemented or unsupported by the target backend.
    UnsupportedBackendOperation {
        /// Name of the operation requested.
        op: &'static str,
        /// Name of the backend.
        backend: &'static str,
    },

    #[error("Invalid device provided: expected {expected}, got {got}")]
    /// Device initialization or mismatch error.
    DeviceInitializationError {
        /// Expected device string.
        expected: String,
        /// Actual device string.
        got: String,
    },

    #[error("Internal Backend Failure: {0}")]
    /// Internal backend execution failure.
    BackendFailure(#[from] anyhow::Error),

    #[error("Generic Message: {0}")]
    /// Generic error string.
    Msg(String),
}

impl Debug for Error {
    /// Format error using Display representation.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;

    #[test]
    /// `test_error_formatting`.
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
