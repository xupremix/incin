use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Debug;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
pub enum Error {
    #[error("Shape mismatch at runtime: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Out of Memory error on device: {device}")]
    OutOfMemory { device: String },

    #[error("Operation '{op}' is not supported by backend '{backend}'")]
    UnsupportedBackendOperation {
        op: &'static str,
        backend: &'static str,
    },

    #[error("Invalid device provided: expected {expected}, got {got}")]
    DeviceInitializationError { expected: String, got: String },

    #[error("Internal Backend Failure: {0}")]
    BackendFailure(#[from] anyhow::Error),

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
    use alloc::string::ToString;

    #[test]
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
