use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Debug;

#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
/// Structured failure returned by descriptor executors.
pub enum BackendError {
    #[error("{reason}")]
    /// A capability query rejected the request before launch.
    Unsupported {
        reason: crate::exec::capability::UnsupportedReason,
    },

    #[error("invalid input for {operation}: {reason}")]
    /// A checked handle could not be interpreted by the selected backend.
    InvalidInput {
        operation: crate::shapes::error::OperationKind,
        reason: &'static str,
    },

    #[error("backend execution failed for {operation}: {message}")]
    /// A device or native library failed after validation.
    Execution {
        operation: crate::shapes::error::OperationKind,
        message: String,
    },

    #[error("device {device:?} was lost during execution")]
    /// The fingerprinted device disappeared or was reset.
    DeviceLost {
        device: crate::tensor::device::DeviceId,
    },
}

impl From<crate::exec::capability::UnsupportedReason> for BackendError {
    fn from(reason: crate::exec::capability::UnsupportedReason) -> Self {
        Self::Unsupported { reason }
    }
}

/// Structured failure raised while walking the autograd tape backward
/// (`GRD-005`).
///
/// PROPOSALS.md sec. 3.9: "Backward closures must return structured errors.
/// NaN checking is an execution policy applied consistently across backends,
/// not a panic-only backend helper." Both halves of that produce one of these.
/// Before this type a recipe that could not produce a gradient had exactly one
/// way to say so, and 115 sites across three backends took it.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum BackwardError {
    #[error("gradient for tensor {tensor} is not finite (produced by {operation})")]
    /// A gradient held a `NaN` or an infinity, and the ambient
    /// [`NanPolicy`](crate::exec::NanPolicy) asked for that to be caught.
    ///
    /// The tensor is named because the value of checking at all is knowing
    /// *which* operation first produced it; a pass that reports only that some
    /// gradient went non-finite leaves the caller bisecting the graph.
    NonFinite {
        /// The tensor whose accumulated gradient failed the check.
        tensor: u64,
        /// Where in the pass it was found.
        operation: NonFiniteSite,
    },

    #[error("backward recipe for {operation} could not produce a gradient: {reason}")]
    /// A recipe reached a state it has no gradient for.
    ///
    /// Distinct from a kernel failure, which arrives as a
    /// [`BackendError`](crate::err::BackendError) through the same `?`: this
    /// is the recipe's own bookkeeping, and before `GRD-005` every one of them
    /// was an `unwrap` on an `Option` the author believed could not be `None`.
    Recipe {
        /// The forward operation whose gradient was being computed.
        operation: crate::shapes::error::OperationKind,
        /// What the recipe expected and did not find.
        reason: &'static str,
    },
}

/// Where a non-finite gradient was found.
///
/// A contribution and an accumulation fail for different reasons — one recipe
/// produced a bad value, or two individually finite contributions summed to an
/// infinity — and a report that cannot tell them apart sends the reader to the
/// wrong place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonFiniteSite {
    /// A single recipe's output, before it was summed with anything.
    Contribution,
    /// The running total after summing two contributions.
    Accumulation,
}

impl core::fmt::Display for NonFiniteSite {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Contribution => "a backward recipe",
            Self::Accumulation => "accumulating two contributions",
        })
    }
}

/// Convenience type alias for `Result<T, Error>`.
pub type Result<T> = core::result::Result<T, Error>;

#[non_exhaustive]
#[derive(thiserror::Error)]
/// Central error enum for the Incin framework.
pub enum Error {
    #[error(transparent)]
    /// A descriptor executor rejected or failed a validated request.
    Backend(#[from] BackendError),

    #[error(transparent)]
    /// A backward pass failed (`GRD-005`).
    Backward(#[from] BackwardError),

    #[error("Backend '{backend}' is unavailable in this build")]
    /// A runtime-selected backend was not enabled.
    BackendUnavailable { backend: &'static str },

    #[error("Dtype {dtype:?} is unsupported by backend '{backend}' for '{op}'")]
    /// The backend or operation cannot represent the requested dtype.
    UnsupportedDType {
        dtype: crate::prelude::DTypeId,
        backend: &'static str,
        op: &'static str,
    },

    #[error("Tensor dtype metadata {expected:?} does not match storage {got:?}")]
    /// Logical dtype differs from physical storage.
    DTypeStorageMismatch {
        expected: crate::prelude::DTypeId,
        got: crate::prelude::DTypeId,
    },

    #[error("Tensor device metadata {expected:?} does not match storage {got:?}")]
    /// Logical device differs from physical storage.
    DeviceStorageMismatch {
        expected: crate::prelude::DeviceId,
        got: crate::prelude::DeviceId,
    },

    #[error("Device mismatch: left {left:?}, right {right:?}")]
    /// Inputs reside on different devices.
    DeviceMismatch {
        left: crate::prelude::DeviceId,
        right: crate::prelude::DeviceId,
    },

    #[error("Invalid byte length: expected {expected}, got {got}")]
    /// Byte payload length does not match shape and dtype.
    InvalidByteLength { expected: usize, got: usize },

    #[error("Invalid {backend} device ordinal {ordinal}")]
    /// Device ordinal could not be selected.
    InvalidDeviceOrdinal {
        backend: &'static str,
        ordinal: usize,
    },

    #[error("{0}")]
    /// A shape rule could not be discharged.
    ///
    /// This is the structured successor to [`Error::ShapeMismatch`]: it names
    /// the operation, the axis, and the violated rule instead of a free-form
    /// message. New fallible shape code returns
    /// [`ShapeError`](crate::shapes::error::ShapeError) directly; `SHP-004`
    /// and `SHP-005` migrate the existing call sites onto it.
    Shape(#[from] crate::shapes::error::ShapeError),

    #[error("Shape mismatch during '{op}': expected {expected:?}, got {got:?}. {msg}")]
    /// Incompatible shape during tensor operation execution.
    ///
    /// Superseded by [`Error::Shape`]; retained until the last unmigrated call
    /// site moves over.
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

#[cfg(feature = "std")]
impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Msg(err.to_string())
    }
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
