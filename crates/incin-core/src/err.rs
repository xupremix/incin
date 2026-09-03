use crate::shapes::error::OperationKind;
use crate::tensor::device::DeviceId;
use crate::tensor::dtype::DTypeDescriptor;
#[cfg(test)]
use crate::tensor::dtype::DTypeId;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::fmt::Debug;

/// Maximum diagnostic text retained from a backend, parser, or external
/// library. Errors carry enough context to diagnose a failure without allowing
/// untrusted input to grow an error value without bound.
pub const MAX_ERROR_MESSAGE_BYTES: usize = 512;

/// Bounded diagnostic text used by typed failures.
#[derive(Clone, PartialEq, Eq)]
pub struct ErrorMessage(String);

impl ErrorMessage {
    /// Copies at most `MAX_ERROR_MESSAGE_BYTES` from `message`, preserving a
    /// valid UTF-8 boundary.
    #[must_use]
    pub fn new(message: impl AsRef<str>) -> Self {
        let message = message.as_ref();
        let mut end = message.len().min(MAX_ERROR_MESSAGE_BYTES);
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        Self(message[..end].to_string())
    }

    /// Returns the retained diagnostic text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Debug for ErrorMessage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl core::fmt::Display for ErrorMessage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ErrorMessage {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

impl From<&str> for ErrorMessage {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

/// Explicit behavior requested for a floating-point to integer conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatToIntPolicy {
    /// Require a finite, integral, in-range value.
    Exact,
    /// Explicitly discard the fractional part toward zero.
    Truncate,
    /// Explicitly clamp finite or infinite values to the destination range.
    Saturate,
}

/// Why a checked scalar conversion was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionFailure {
    /// The source was NaN or infinity and the requested policy did not define it.
    NonFinite,
    /// Exact conversion was requested for a fractional value.
    Fractional,
    /// The source is outside the destination type's representable range.
    OutOfRange,
}

impl core::fmt::Display for ConversionFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NonFinite => "non-finite value",
            Self::Fractional => "fractional value",
            Self::OutOfRange => "out-of-range value",
        })
    }
}

#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
/// Structured failure returned by descriptor executors.
pub enum BackendError {
    #[error("backend '{backend}' refused the request: {reason}")]
    /// A capability query rejected the request before launch.
    ///
    /// `backend` is mandatory. See [`StorageBackend::BACKEND_NAME`] for why a
    /// refusal that does not name its author is not worth returning.
    ///
    /// [`StorageBackend::BACKEND_NAME`]: crate::tensor::backend::StorageBackend::BACKEND_NAME
    Unsupported {
        /// Name of the backend involved.
        backend: &'static str,
        /// Structured reason the capability is unsupported.
        reason: crate::exec::capability::UnsupportedReason,
    },

    #[error("invalid input for {operation}: {reason}")]
    /// A checked handle could not be interpreted by the selected backend.
    InvalidInput {
        /// Operation involved in the failure.
        operation: crate::shapes::error::OperationKind,
        /// Bounded explanation of the failure.
        reason: &'static str,
    },

    #[error("backend execution failed for {operation}: {message}")]
    /// A device or native library failed after validation.
    Execution {
        /// Operation involved in the failure.
        operation: crate::shapes::error::OperationKind,
        /// Bounded diagnostic message.
        message: ErrorMessage,
    },

    #[error("device {device:?} was lost during execution")]
    /// The fingerprinted device disappeared or was reset.
    DeviceLost {
        /// Device the failure names.
        device: crate::tensor::device::DeviceId,
    },
}

impl BackendError {
    /// A refusal attributed to the backend that made it.
    ///
    /// This replaces a `From<UnsupportedReason>` conversion. `?` on a bare
    /// reason used to be enough to produce an unattributed error, which is how
    /// the canonical path ended up reporting "dtype Q8_0 is unsupported for
    /// zeros" without ever naming the device. Requiring the name as an
    /// argument means the only way to build one of these is at a site that
    /// knows who refused.
    #[must_use]
    pub const fn unsupported(
        backend: &'static str,
        reason: crate::exec::capability::UnsupportedReason,
    ) -> Self {
        Self::Unsupported { backend, reason }
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
/// A contribution and an accumulation fail for different reasons - one recipe
/// produced a bad value, or two individually finite contributions summed to an
/// infinity - and a report that cannot tell them apart sends the reader to the
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

    #[error("{0}")]
    /// A canonical invocation failed contract validation before any backend ran.
    ///
    /// Alongside [`Error::Policy`] and [`Error::Backend`], this preserves every
    /// [`CanonicalError`](crate::exec::CanonicalError) category at the public
    /// boundary. Without this variant the only way to surface a descriptor
    /// rejection was to render it into a string and post it as some other
    /// variant, which is how a dtype refusal ended up reported as an execution
    /// failure of an operation the caller had not asked for.
    Descriptor(crate::exec::catalog::DescriptorError),

    #[error("{0}")]
    /// A valid invocation required a support level the execution policy denies.
    Policy(crate::exec::PolicyViolation),

    #[error(transparent)]
    /// A backward pass failed (`GRD-005`).
    Backward(#[from] BackwardError),

    #[error(
        "Backend '{backend}' is unavailable in this build\n  requested: {backend}\n  rule: a backend is compiled in by a cargo feature; selecting one at \
         run time cannot enable code that was never built\n  fix: add the matching feature to your incin dependency (for example \
         features = [\"cpu\"], [\"cuda\"], [\"metal\"] or [\"wgpu\"]) and rebuild"
    )]
    /// A runtime-selected backend was not enabled in this build.
    BackendUnavailable {
        /// Name of the backend involved.
        backend: &'static str,
    },

    #[error(
        "Dtype {dtype:?} is unsupported by backend '{backend}' for '{op}'\n  operation: {op}\n  backend: {backend}\n  requested dtype: {dtype:?}\n  rule: a capability row states which dtypes an operation accepts on a \
         backend, and the executor behind it refuses the rest\n  fix: convert the operand with `.to_dtype(..)` to a dtype this row \
         advertises, or run the operation on a backend that advertises this \
         one; docs/capabilities.md lists the rows per backend"
    )]
    /// The backend or operation cannot represent the requested dtype.
    UnsupportedDType {
        /// Dtype the query asks about.
        dtype: DTypeDescriptor,
        /// Name of the backend involved.
        backend: &'static str,
        /// Operation identity string.
        op: &'static str,
    },

    #[error(
        "Precision choice {requested:?} for {role:?} role is unsupported for operation '{operation:?}', storage {storage:?} on backend '{backend}'"
    )]
    /// Requested precision choice cannot be honored by the backend.
    UnsupportedPrecision {
        /// Operation this query targets.
        operation: OperationKind,
        /// Storage dtype observed on the tensor.
        storage: DTypeDescriptor,
        /// Precision the caller requested.
        requested: DTypeDescriptor,
        /// Role (compute vs accumulator) the request applies to.
        role: crate::exec::PrecisionRole,
        /// Name of the backend involved.
        backend: &'static str,
    },

    #[error("Tensor dtype metadata {expected:?} does not match storage {got:?}")]
    /// Logical dtype differs from physical storage.
    DTypeStorageMismatch {
        /// Dtype metadata promised.
        expected: DTypeDescriptor,
        /// Dtype storage actually held.
        got: DTypeDescriptor,
    },

    #[error("{operation}: cannot convert {from:?} to {to:?}: {reason}")]
    /// A numeric conversion would silently truncate, saturate, or fabricate a
    /// value under the requested policy.
    InvalidConversion {
        /// Stable operation identity.
        operation: &'static str,
        /// Physical source dtype.
        from: DTypeDescriptor,
        /// Requested destination dtype.
        to: DTypeDescriptor,
        /// Bounded classification of the rejected value.
        reason: ConversionFailure,
    },

    #[error("{operation}: dtype mismatch: expected {expected:?}, got {actual:?}")]
    /// An operation received a dtype outside its declared contract.
    DTypeMismatch {
        /// Operation that failed validation or execution.
        operation: &'static str,
        /// Dtype metadata promised.
        expected: DTypeDescriptor,
        /// Dtype actually present.
        actual: DTypeDescriptor,
    },

    #[error("Tensor device metadata {expected:?} does not match storage {got:?}")]
    /// Logical device differs from physical storage.
    DeviceStorageMismatch {
        /// Value the contract expects.
        expected: DeviceId,
        /// Value actually present.
        got: DeviceId,
    },
    #[error("Device mismatch: left {left:?}, right {right:?}")]
    /// Inputs reside on different devices.
    DeviceMismatch {
        /// First device in the mismatch.
        left: DeviceId,
        /// Second device in the mismatch.
        right: DeviceId,
    },

    #[error("{operation}: device or placement mismatch: expected {expected:?}, got {actual:?}")]
    /// An operation received storage on a different device or placement.
    PlacementMismatch {
        /// Operation that failed validation or execution.
        operation: &'static str,
        /// Device the contract requires.
        expected: DeviceId,
        /// Device storage actually resides on.
        actual: DeviceId,
    },

    #[error("Invalid byte length: expected {expected}, got {got}")]
    /// Byte payload length does not match shape and dtype.
    /// Byte length disagrees with the element contract.
    InvalidByteLength {
        /// Value the contract expects.
        expected: usize,
        /// Value actually present.
        got: usize,
    },
    /// Byte length disagrees with the element contract.

    #[error("Invalid {backend} device ordinal {ordinal}")]
    /// Device ordinal could not be selected.
    InvalidDeviceOrdinal {
        /// Name of the backend involved.
        backend: &'static str,
        /// Zero-based position among same-kind backends.
        ordinal: usize,
    },

    #[error("{operation}: arithmetic overflow evaluating '{expression}'")]
    /// Checked non-shape arithmetic overflowed.
    ArithmeticOverflow {
        /// Operation that failed validation or execution.
        operation: &'static str,
        /// Named expression that overflowed or failed.
        expression: &'static str,
    },

    #[error("{operation}: allocation of {requested} bytes exceeds limit {limit}")]
    /// An allocation request exceeded address-space or resource limits.
    AllocationOverflow {
        /// Operation that failed validation or execution.
        operation: &'static str,
        /// Requested size or value.
        requested: u64,
        /// Configured bound that was exceeded.
        limit: u64,
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

    #[error("{operation}: invalid module or state dictionary: {reason}")]
    /// Module parameters or serialized state violate the module contract.
    InvalidModuleState {
        /// Operation that failed validation or execution.
        operation: &'static str,
        /// Bounded explanation attached to the failure.
        reason: ErrorMessage,
    },

    #[error("{operation}: malformed {artifact}: {reason}")]
    /// A model, dataset, cache, or compiled artifact is malformed.
    MalformedArtifact {
        /// Operation that failed validation or execution.
        operation: &'static str,
        /// Artifact kind being parsed.
        artifact: &'static str,
        /// Bounded explanation attached to the failure.
        reason: ErrorMessage,
    },

    #[error("{operation}: resource '{resource}' value {actual} exceeds limit {limit}")]
    /// Untrusted input exceeded a configured resource bound.
    ResourceLimit {
        /// Operation that failed validation or execution.
        operation: &'static str,
        /// Resource whose limit was exceeded.
        resource: &'static str,
        /// Observed size or value.
        actual: u64,
        /// Configured bound that was exceeded.
        limit: u64,
    },

    #[error("{operation}: I/O failure: {message}")]
    /// A filesystem or stream operation failed.
    Io {
        /// Operation that failed validation or execution.
        operation: &'static str,
        /// Bounded diagnostic message.
        message: ErrorMessage,
    },

    #[error("{operation}: internal invariant violation: {reason}")]
    /// A value that had already crossed validation contradicted its proof.
    InternalInvariant {
        /// Operation that failed validation or execution.
        operation: &'static str,
        /// Bounded explanation of the failure.
        reason: &'static str,
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
        Error::Io {
            operation: "io",
            message: ErrorMessage::new(err.to_string()),
        }
    }
}

impl From<crate::exec::CanonicalError> for Error {
    /// Carry a canonical failure across without flattening it.
    ///
    /// Every arm has an exact counterpart here, so this loses nothing: the
    /// distinction between "the request was never legal" and "the backend
    /// refused a legal request" survives, and a [`BackendError::Unsupported`]
    /// still names the backend that produced it. Callers that used to render
    /// the error with `format!` and re-post it under a guessed variant should
    /// use this instead.
    fn from(error: crate::exec::CanonicalError) -> Self {
        match error {
            crate::exec::CanonicalError::Descriptor(error) => Self::Descriptor(error),
            crate::exec::CanonicalError::Policy(error) => Self::Policy(error),
            crate::exec::CanonicalError::Backend(error) => Self::Backend(error),
        }
    }
}

impl From<crate::exec::MetaError> for Error {
    fn from(err: crate::exec::MetaError) -> Self {
        match err {
            crate::exec::MetaError::Shape(s) => Error::Shape(s),
            other => Error::InternalInvariant {
                operation: "tensor_metadata",
                reason: match other {
                    crate::exec::MetaError::InvalidAlignment { .. } => {
                        "metadata alignment is invalid"
                    }
                    crate::exec::MetaError::OutOfBounds { .. } => {
                        "metadata addresses storage outside its allocation"
                    }
                    crate::exec::MetaError::Shape(_) => {
                        "metadata shape error bypassed its structured conversion"
                    }
                },
            },
        }
    }
}

/// Converts `value` to `i64` under an explicit policy.
///
/// Exact conversion is the default used by tensor integer readback. Callers
/// must name truncation or saturation when those semantics are intended.
pub fn convert_f64_to_i64(
    operation: &'static str,
    from: DTypeDescriptor,
    value: f64,
    policy: FloatToIntPolicy,
) -> Result<i64> {
    if value.is_nan() {
        return Err(Error::InvalidConversion {
            operation,
            from,
            to: <i64 as crate::tensor::dtype::ConstDType>::DESCRIPTOR,
            reason: ConversionFailure::NonFinite,
        });
    }
    if value.is_infinite() {
        return match policy {
            FloatToIntPolicy::Saturate => Ok(if value.is_sign_negative() {
                i64::MIN
            } else {
                i64::MAX
            }),
            FloatToIntPolicy::Exact | FloatToIntPolicy::Truncate => Err(Error::InvalidConversion {
                operation,
                from,
                to: <i64 as crate::tensor::dtype::ConstDType>::DESCRIPTOR,
                reason: ConversionFailure::NonFinite,
            }),
        };
    }

    let candidate = match policy {
        FloatToIntPolicy::Exact if value.fract() != 0.0 => {
            return Err(Error::InvalidConversion {
                operation,
                from,
                to: <i64 as crate::tensor::dtype::ConstDType>::DESCRIPTOR,
                reason: ConversionFailure::Fractional,
            });
        }
        FloatToIntPolicy::Exact | FloatToIntPolicy::Truncate => value.trunc(),
        FloatToIntPolicy::Saturate => value,
    };

    // `i64::MAX as f64` rounds to 2^63, which is already one beyond the
    // inclusive integer range. Keep the upper comparison exclusive.
    const I64_MIN_F64: f64 = -9_223_372_036_854_775_808.0;
    const I64_MAX_EXCLUSIVE_F64: f64 = 9_223_372_036_854_775_808.0;
    if !(I64_MIN_F64..I64_MAX_EXCLUSIVE_F64).contains(&candidate) {
        return match policy {
            FloatToIntPolicy::Saturate => Ok(if candidate.is_sign_negative() {
                i64::MIN
            } else {
                i64::MAX
            }),
            FloatToIntPolicy::Exact | FloatToIntPolicy::Truncate => Err(Error::InvalidConversion {
                operation,
                from,
                to: <i64 as crate::tensor::dtype::ConstDType>::DESCRIPTOR,
                reason: ConversionFailure::OutOfRange,
            }),
        };
    }
    Ok(candidate as i64)
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
    fn error_message_is_bounded_at_a_utf8_boundary() {
        let ascii = ErrorMessage::new("x".repeat(MAX_ERROR_MESSAGE_BYTES + 32));
        assert_eq!(ascii.as_str().len(), MAX_ERROR_MESSAGE_BYTES);

        let unicode = ErrorMessage::new(format!("{}é", "x".repeat(MAX_ERROR_MESSAGE_BYTES - 1)));
        assert_eq!(unicode.as_str().len(), MAX_ERROR_MESSAGE_BYTES - 1);
        assert!(unicode.as_str().is_char_boundary(unicode.as_str().len()));
    }

    #[test]
    fn exact_float_to_integer_conversion_rejects_lossy_values() {
        assert_eq!(
            convert_f64_to_i64(
                "index_readback",
                DTypeId::F64.descriptor(),
                42.0,
                FloatToIntPolicy::Exact,
            )
            .unwrap(),
            42
        );

        for (value, expected) in [
            (42.5, ConversionFailure::Fractional),
            (f64::NAN, ConversionFailure::NonFinite),
            (f64::INFINITY, ConversionFailure::NonFinite),
            (f64::NEG_INFINITY, ConversionFailure::NonFinite),
            (9_223_372_036_854_775_808.0, ConversionFailure::OutOfRange),
            (-9_223_372_036_854_777_856.0, ConversionFailure::OutOfRange),
        ] {
            let err = convert_f64_to_i64(
                "index_readback",
                DTypeId::F64.descriptor(),
                value,
                FloatToIntPolicy::Exact,
            )
            .unwrap_err();
            if let Error::InvalidConversion {
                operation,
                from,
                to,
                reason,
            } = err
            {
                assert_eq!(operation, "index_readback");
                assert_eq!(from, DTypeId::F64.descriptor());
                assert_eq!(to, DTypeId::I64.descriptor());
                assert_eq!(reason, expected);
            } else {
                panic!("expected Error::InvalidConversion");
            }
        }
    }

    #[test]
    fn lossy_float_to_integer_conversion_requires_an_explicit_policy() {
        assert_eq!(
            convert_f64_to_i64(
                "explicit_truncate",
                DTypeId::F64.descriptor(),
                -42.75,
                FloatToIntPolicy::Truncate,
            )
            .unwrap(),
            -42
        );
        assert_eq!(
            convert_f64_to_i64(
                "explicit_saturate",
                DTypeId::F64.descriptor(),
                f64::INFINITY,
                FloatToIntPolicy::Saturate,
            )
            .unwrap(),
            i64::MAX
        );
        assert_eq!(
            convert_f64_to_i64(
                "explicit_saturate",
                DTypeId::F64.descriptor(),
                f64::NEG_INFINITY,
                FloatToIntPolicy::Saturate,
            )
            .unwrap(),
            i64::MIN
        );
        assert!(matches!(
            convert_f64_to_i64(
                "explicit_saturate",
                DTypeId::F64.descriptor(),
                f64::NAN,
                FloatToIntPolicy::Saturate,
            ),
            Err(Error::InvalidConversion {
                reason: ConversionFailure::NonFinite,
                ..
            })
        ));
    }

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
