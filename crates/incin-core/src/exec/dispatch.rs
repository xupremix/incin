//! The single production path from a typed operation to native execution.
//!
//! FND-004 froze one exact identity, one typed descriptor and one capability
//! row per operation, but nothing outside the test suite consumed them: the
//! stable tensor surface still called the operation-family traits directly.
//! This module is the missing production consumer, and the properties FND-005
//! requires of it are all structural rather than conventional:
//!
//! - **Support is explicit.** `B: Execute<O>` is a compile-time
//!   fact, and the exact capability row is queried before launch. A backend
//!   that has not implemented an operation cannot be asked to run it, and a
//!   backend that implements it for one dtype cannot be asked for another.
//! - **Validation precedes execution.** The descriptor is validated against
//!   the real storage metadata, not against what a caller claims.
//! - **Nothing silently returns unsupported.** There is no default method to
//!   fall through; a refusal is a typed [`UnsupportedReason`].
//! - **Outputs are derived, not accepted.** The caller never states output
//!   metadata, so it cannot fabricate any.
//! - **Capture keeps the same descriptor.** The value handed to the backend is
//!   the value a compiler would record.

use alloc::vec::Vec;
use core::fmt;

use crate::err::BackendError;
use crate::exec::capability::{
    Capabilities, CapabilityQuery, OperationIdentity, SupportLevel, UnsupportedReason,
};
use crate::exec::catalog::{Descriptor, DescriptorError, LogicalTensorMeta, Operation};
use crate::exec::context::ExecutionContext;
use crate::exec::meta::TensorMeta;
use crate::exec::policy::GradMode;

use crate::exec::request::TensorHandle;
use crate::tensor::backend::{Execute, ExecutionRequest};

/// Why a canonical invocation did not produce a value.
///
/// The two arms are kept apart on purpose. A [`DescriptorError`] means the
/// request was never legal and no backend was reached; a [`BackendError`] means
/// a legal request failed at or after launch. Collapsing them would lose the
/// distinction that decides whether the caller or the device is at fault.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalError {
    /// The invocation failed contract validation before any backend ran.
    Descriptor(DescriptorError),
    /// The backend refused or failed a validated invocation.
    Backend(BackendError),
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor(error) => write!(f, "{error}"),
            Self::Backend(error) => write!(f, "{error}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CanonicalError {}

impl From<DescriptorError> for CanonicalError {
    fn from(error: DescriptorError) -> Self {
        Self::Descriptor(error)
    }
}

impl From<BackendError> for CanonicalError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error)
    }
}

impl CanonicalError {
    /// A refusal attributed to the backend that made it.
    ///
    /// Deliberately not a `From<UnsupportedReason>` impl: see
    /// [`BackendError::unsupported`]. `?` on a bare reason must not compile,
    /// because the reason alone does not say which backend produced it.
    #[must_use]
    pub const fn unsupported(backend: &'static str, reason: UnsupportedReason) -> Self {
        Self::Backend(BackendError::unsupported(backend, reason))
    }
}

/// Read logical metadata off a checked handle.
///
/// Every field is taken from the allocation the backend will actually read, so
/// validation cannot be satisfied by metadata that describes some other tensor.
#[must_use]
pub fn logical_meta(metadata: &TensorMeta) -> LogicalTensorMeta {
    LogicalTensorMeta {
        shape: Some(metadata.shape.clone()),
        dtype: Some(metadata.dtype),
        device: Some(metadata.device),
    }
}

/// Ask the backend's exact capability registry whether `operation` may run over
/// one concrete operand.
fn admit<B: Capabilities>(
    backend: &B,
    operation: &OperationIdentity,
    metadata: &TensorMeta,
    context_training: bool,
    math_mode: crate::exec::policy::MathMode,
) -> Result<SupportLevel, UnsupportedReason> {
    let query = CapabilityQuery {
        operation: operation.clone(),
        dtype: metadata.dtype,
        layout: metadata.layout,
        rank: metadata.shape.dims().len(),
        training: context_training,
        math_mode,
    };
    match backend.support(&query) {
        SupportLevel::Unsupported(reason) => Err(reason),
        level => Ok(level),
    }
}

fn admit_operation<O, B>(
    backend: &B,
    operation: &OperationIdentity,
    metadata: &TensorMeta,
    context_training: bool,
    math_mode: crate::exec::policy::MathMode,
) -> Result<SupportLevel, UnsupportedReason>
where
    O: Operation,
    B: Capabilities + Execute<O>,
{
    let query = CapabilityQuery {
        operation: operation.clone(),
        dtype: metadata.dtype,
        layout: metadata.layout,
        rank: metadata.shape.dims().len(),
        training: context_training,
        math_mode,
    };
    let support = match operation {
        OperationIdentity::Builtin(_) => Capabilities::support(backend, &query),
        OperationIdentity::Custom(_) => Execute::<O>::supports_custom(backend, &query),
    };
    match support {
        SupportLevel::Unsupported(reason) => Err(reason),
        level => Ok(level),
    }
}

fn admit_metadata_free_operation<O, B>(
    backend: &B,
    operation: &OperationIdentity,
    training: bool,
    math_mode: crate::exec::policy::MathMode,
) -> Result<(), UnsupportedReason>
where
    O: Operation,
    B: Execute<O>,
{
    let OperationIdentity::Custom(_) = operation else {
        return Ok(());
    };
    match Execute::<O>::supports_custom_operation(backend, operation, training, math_mode) {
        SupportLevel::Unsupported(reason) => Err(reason),
        SupportLevel::Native | SupportLevel::Composed | SupportLevel::Fallback => Ok(()),
    }
}

/// Validate and run one operation on `context`'s backend.
///
/// `O` names the exact operation identity, so the descriptor, capability query
/// and `Execute` implementation are selected by the same token.
///
/// # Errors
///
/// Returns [`CanonicalError::Descriptor`] when the invocation fails contract
/// validation, and [`CanonicalError::Backend`] when the backend's capability
/// registry refuses it or execution itself fails.
pub fn execute<O, B>(
    context: &ExecutionContext<B>,
    attributes: O::Attributes,
    inputs: &[TensorHandle<'_>],
) -> Result<<B as Execute<O>>::Output, CanonicalError>
where
    O: Operation,
    B: Execute<O> + Capabilities,
{
    execute_with_payload(context, attributes, inputs, None)
}

/// Execute an operation with an optional borrowed payload kept outside its descriptor.
pub fn execute_with_payload<O, B>(
    context: &ExecutionContext<B>,
    attributes: O::Attributes,
    inputs: &[TensorHandle<'_>],
    payload: Option<&[u8]>,
) -> Result<<B as Execute<O>>::Output, CanonicalError>
where
    O: Operation,
    B: Execute<O> + Capabilities,
{
    let logical: Vec<LogicalTensorMeta> = inputs
        .iter()
        .map(|handle| logical_meta(handle.metadata()))
        .collect();

    let invocation = O::infer_invocation(attributes, logical)?;

    validate_execution_payload::<O>(&invocation, payload)?;

    let training = context.training();
    let identity = invocation.descriptor().identity();
    for handle in inputs {
        if let Err(reason) = admit_operation::<O, B>(
            context.backend(),
            identity,
            handle.metadata(),
            training,
            context.math_mode(),
        ) {
            return Err(CanonicalError::unsupported(B::BACKEND_NAME, reason));
        }
    }
    if inputs.is_empty() {
        let mut queried = false;
        for output in invocation.descriptor().outputs() {
            let (Some(dtype), Some(shape)) = (output.dtype, output.shape.as_deref()) else {
                continue;
            };
            queried = true;
            let query = CapabilityQuery {
                operation: identity.clone(),
                dtype,
                layout: crate::exec::meta::LayoutClass::Contiguous,
                rank: shape.len(),
                training,
                math_mode: context.math_mode(),
            };
            let support = match identity {
                OperationIdentity::Builtin(_) => Capabilities::support(context.backend(), &query),
                OperationIdentity::Custom(_) => {
                    <B as Execute<O>>::supports_custom(context.backend(), &query)
                }
            };
            if let SupportLevel::Unsupported(reason) = support {
                return Err(CanonicalError::unsupported(B::BACKEND_NAME, reason));
            }
        }
        if !queried {
            admit_metadata_free_operation::<O, B>(
                context.backend(),
                identity,
                training,
                context.math_mode(),
            )
            .map_err(|reason| CanonicalError::unsupported(B::BACKEND_NAME, reason))?;
        }
    }

    context
        .backend()
        .execute(ExecutionRequest {
            operation: invocation.validated(),
            inputs,
            context,
            payload,
        })
        .map_err(CanonicalError::Backend)
}

/// [`execute`], told the shape value the typed frontend was holding.
pub fn execute_shaped<O, B, S>(
    context: &ExecutionContext<B>,
    attributes: O::Attributes,
    inputs: &[TensorHandle<'_>],
    expected: &crate::shapes::ShapeValue<S>,
) -> Result<<B as Execute<O>>::Output, CanonicalError>
where
    O: Operation,
    B: Execute<O> + Capabilities,
    S: crate::prelude::Shape,
{
    execute_shaped_with_payload(context, attributes, inputs, expected, None)
}

/// [`execute_shaped`], with an optional borrowed execution payload.
pub fn execute_shaped_with_payload<O, B, S>(
    context: &ExecutionContext<B>,
    attributes: O::Attributes,
    inputs: &[TensorHandle<'_>],
    expected: &crate::shapes::ShapeValue<S>,
    payload: Option<&[u8]>,
) -> Result<<B as Execute<O>>::Output, CanonicalError>
where
    O: Operation,
    B: Execute<O> + Capabilities,
    S: crate::prelude::Shape,
{
    let logical: Vec<LogicalTensorMeta> = inputs
        .iter()
        .map(|handle| logical_meta(handle.metadata()))
        .collect();

    let invocation = O::infer_invocation_typed(attributes, logical, expected)?;

    validate_execution_payload::<O>(&invocation, payload)?;

    let training = context.training();
    let identity = invocation.descriptor().identity();
    {
        for handle in inputs {
            admit_operation::<O, B>(
                context.backend(),
                identity,
                handle.metadata(),
                training,
                context.math_mode(),
            )
            .map_err(|reason| CanonicalError::unsupported(B::BACKEND_NAME, reason))?;
        }
        // An operation with no operands would otherwise run that loop zero times
        // and reach the backend without a single capability query, which is the one
        // way through here that skips the registry entirely. Every creation
        // operation in the catalog has an operand arity of zero, so this is not a
        // hypothetical gap: it is the whole creation family.
        //
        // What such an operation must be supported *for* is the allocation it was
        // asked to produce, and the descriptor already carries that: the inferred
        // output metadata names the dtype, the device and the rank. A fresh
        // allocation is contiguous by construction, so the layout is not inferred
        // from anything, it is a fact about what the backend is being told to make.
        if inputs.is_empty() {
            let mut queried = false;
            for output in invocation.descriptor().outputs() {
                let (Some(dtype), Some(shape)) = (output.dtype, output.shape.as_deref()) else {
                    continue;
                };
                queried = true;
                let query = CapabilityQuery {
                    operation: identity.clone(),
                    dtype,
                    layout: crate::exec::meta::LayoutClass::Contiguous,
                    rank: shape.len(),
                    training,
                    math_mode: context.math_mode(),
                };
                let support = match identity {
                    OperationIdentity::Builtin(_) => {
                        Capabilities::support(context.backend(), &query)
                    }
                    OperationIdentity::Custom(_) => {
                        <B as Execute<O>>::supports_custom(context.backend(), &query)
                    }
                };
                if let SupportLevel::Unsupported(reason) = support {
                    return Err(CanonicalError::unsupported(B::BACKEND_NAME, reason));
                }
            }
            if !queried {
                admit_metadata_free_operation::<O, B>(
                    context.backend(),
                    identity,
                    training,
                    context.math_mode(),
                )
                .map_err(|reason| CanonicalError::unsupported(B::BACKEND_NAME, reason))?;
            }
        }
    }

    context
        .backend()
        .execute(ExecutionRequest {
            operation: invocation.validated(),
            inputs,
            context,
            payload,
        })
        .map_err(CanonicalError::Backend)
}

fn validate_execution_payload<O>(
    invocation: &crate::exec::catalog::ValidatedInvocation<O>,
    payload: Option<&[u8]>,
) -> Result<(), CanonicalError>
where
    O: Operation,
{
    let attributes = invocation.descriptor().attributes();
    let data =
        (attributes as &dyn core::any::Any).downcast_ref::<crate::exec::catalog::DataAttributes>();
    let operation = match invocation.descriptor().identity() {
        OperationIdentity::Builtin(operation) => operation,
        OperationIdentity::Custom(_) => return Ok(()),
    };
    match data {
        Some(data) => match payload {
            Some(bytes) if bytes.len() == data.payload.byte_len() => Ok(()),
            Some(bytes) => Err(CanonicalError::Descriptor(
                crate::exec::catalog::DescriptorError::PayloadByteLength {
                    operation: *operation,
                    expected: data.payload.byte_len(),
                    actual: bytes.len(),
                },
            )),
            None => Err(CanonicalError::Descriptor(
                crate::exec::catalog::DescriptorError::PayloadMissing {
                    operation: *operation,
                },
            )),
        },
        None if payload.is_some() => Err(CanonicalError::Descriptor(
            crate::exec::catalog::DescriptorError::UnexpectedPayload {
                operation: *operation,
            },
        )),
        None => Ok(()),
    }
}

/// The support level `operation` would resolve to for one operand, without
/// running it.
///
/// A caller that wants to choose between backends needs the same answer the
/// execution path uses, from the same registry, or the choice is made against a
/// second source of truth.
///
/// # Errors
///
/// Returns the typed reason the exact capability row refuses the operand.
pub fn support_for<O, B>(
    context: &ExecutionContext<B>,
    metadata: &TensorMeta,
) -> Result<SupportLevel, UnsupportedReason>
where
    O: crate::exec::catalog::CanonicalOperation,
    B: Capabilities + crate::tensor::backend::StorageBackend,
{
    admit(
        context.backend(),
        &OperationIdentity::Builtin(O::ID),
        metadata,
        context.training(),
        context.math_mode(),
    )
}
