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
use crate::exec::catalog::{DescriptorError, LogicalTensorMeta, Operation};
use crate::exec::context::ExecutionContext;
use crate::exec::meta::TensorMeta;
use crate::exec::policy::FallbackPolicy;

use crate::exec::request::TensorHandle;
use crate::tensor::backend::{Execute, ExecutionRequest};

/// Why a canonical invocation did not produce a value.
///
/// The variants are kept apart on purpose. A [`DescriptorError`] means the
/// request was never legal, [`PolicyViolation`] means its requested support is
/// disallowed before launch, and a [`BackendError`] means a legal request
/// failed at or after launch. Collapsing them would lose the distinction that
/// decides whether the caller, policy, or device is at fault.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalError {
    /// The invocation failed contract validation before any backend ran.
    Descriptor(DescriptorError),
    /// A valid invocation required support the execution policy does not allow.
    Policy(PolicyViolation),
    /// The backend refused or failed a validated invocation.
    Backend(BackendError),
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor(error) => write!(f, "{error}"),
            Self::Policy(error) => write!(f, "{error}"),
            Self::Backend(error) => write!(f, "{error}"),
        }
    }
}

/// A capability level that a valid invocation reported but its policy denies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyViolation {
    /// The exact built-in or custom operation whose reported support was denied.
    pub operation: OperationIdentity,
    /// The support level reported by the capability query.
    pub support: SupportLevel,
    /// The fallback policy effective for this invocation.
    pub fallback: FallbackPolicy,
}

impl fmt::Display for PolicyViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "operation {} reports {:?} support, denied by fallback policy {}",
            self.operation.display_name(),
            self.support,
            self.fallback.as_str()
        )
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

/// Convert a capability answer into canonical admission, including the
/// context's fallback policy. Keeping this in one place ensures no dispatch
/// route can reach `Execute` with a composed or transfer fallback it did not
/// explicitly permit.
fn admit_support(
    backend: &'static str,
    operation: &OperationIdentity,
    support: SupportLevel,
    fallback: FallbackPolicy,
) -> Result<(), CanonicalError> {
    match support {
        SupportLevel::Native => Ok(()),
        SupportLevel::Composed if fallback.allows_composition() => Ok(()),
        SupportLevel::Fallback if fallback.allows_transfer() => Ok(()),
        SupportLevel::Unsupported(reason) => Err(CanonicalError::unsupported(backend, reason)),
        support @ (SupportLevel::Composed | SupportLevel::Fallback) => {
            Err(CanonicalError::Policy(PolicyViolation {
                operation: operation.clone(),
                support,
                fallback,
            }))
        }
    }
}

fn admit_operation<O, B>(
    backend: &B,
    operation: &OperationIdentity,
    metadata: &TensorMeta,
    context_training: bool,
    math_mode: crate::exec::policy::MathMode,
    fallback: FallbackPolicy,
) -> Result<(), CanonicalError>
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
    admit_support(B::BACKEND_NAME, operation, support, fallback)
}

fn admit_metadata_free_operation<O, B>(
    backend: &B,
    operation: &OperationIdentity,
    training: bool,
    math_mode: crate::exec::policy::MathMode,
    fallback: FallbackPolicy,
) -> Result<(), CanonicalError>
where
    O: Operation,
    B: Execute<O>,
{
    let OperationIdentity::Custom(_) = operation else {
        return Ok(());
    };
    admit_support(
        B::BACKEND_NAME,
        operation,
        Execute::<O>::supports_custom_operation(backend, operation, training, math_mode),
        fallback,
    )
}

fn admit_invocation<O, B>(
    context: &ExecutionContext<B>,
    invocation: &crate::exec::catalog::ValidatedInvocation<O>,
    inputs: &[TensorHandle<'_>],
) -> Result<(), CanonicalError>
where
    O: Operation,
    B: Capabilities + Execute<O>,
{
    let training = context.training();
    let fallback = context.fallback();
    let identity = invocation.descriptor().identity();
    for handle in inputs {
        admit_operation::<O, B>(
            context.backend(),
            identity,
            handle.metadata(),
            training,
            context.math_mode(),
            fallback,
        )?;
    }
    if !inputs.is_empty() {
        return Ok(());
    }

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
                Execute::<O>::supports_custom(context.backend(), &query)
            }
        };
        admit_support(B::BACKEND_NAME, identity, support, fallback)?;
    }
    if !queried {
        admit_metadata_free_operation::<O, B>(
            context.backend(),
            identity,
            training,
            context.math_mode(),
            fallback,
        )?;
    }
    Ok(())
}

/// Validate and run one operation on `context`'s backend.
///
/// `O` names the exact operation identity, so the descriptor, capability query
/// and `Execute` implementation are selected by the same token.
///
/// # Errors
///
/// Returns [`CanonicalError::Descriptor`] when the invocation fails contract
/// validation, [`CanonicalError::Policy`] when its reported support is denied
/// by the effective fallback policy, and [`CanonicalError::Backend`] when the
/// backend's capability registry refuses it or execution itself fails.
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

    admit_invocation::<O, B>(context, &invocation, inputs)?;

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
///
/// The arity-1 spelling of [`execute_shaped_n`]: the expectation carries one
/// proof, and the comparison below degrades to exactly the loop it always
/// ran. The 100+ call sites across the tensor surface keep passing `S`, the
/// shape type, rather than spelling `ShapeValue<S>` at every one of them.
pub fn execute_shaped<O, B, S>(
    context: &ExecutionContext<B>,
    attributes: O::Attributes,
    inputs: &[TensorHandle<'_>],
    expected: &crate::shapes::ShapeValue<S>,
) -> Result<<B as Execute<O>>::Output, CanonicalError>
where
    O: Operation,
    B: Execute<O> + Capabilities,
    S: crate::shapes::Shape,
{
    execute_shaped_n(context, attributes, inputs, expected)
}

/// [`execute_shaped`], told the shape proofs the typed frontend was holding.
///
/// Generic over the expectation, not over one shape: a single `ShapeValue<S>`
/// and a tuple of them compare element-wise against the inferred outputs, so
/// a multi-output operation finally travels the typed path instead of
/// re-deriving its geometry frontend-side.
pub fn execute_shaped_n<O, B, E>(
    context: &ExecutionContext<B>,
    attributes: O::Attributes,
    inputs: &[TensorHandle<'_>],
    expected: &E,
) -> Result<<B as Execute<O>>::Output, CanonicalError>
where
    O: Operation,
    B: Execute<O> + Capabilities,
    E: crate::shapes::ExpectedShapes,
{
    execute_shaped_n_with_payload(context, attributes, inputs, expected, None)
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
    S: crate::shapes::Shape,
{
    execute_shaped_n_with_payload(context, attributes, inputs, expected, payload)
}

/// [`execute_shaped_n`], with an optional borrowed execution payload.
pub fn execute_shaped_n_with_payload<O, B, E>(
    context: &ExecutionContext<B>,
    attributes: O::Attributes,
    inputs: &[TensorHandle<'_>],
    expected: &E,
    payload: Option<&[u8]>,
) -> Result<<B as Execute<O>>::Output, CanonicalError>
where
    O: Operation,
    B: Execute<O> + Capabilities,
    E: crate::shapes::ExpectedShapes,
{
    let logical: Vec<LogicalTensorMeta> = inputs
        .iter()
        .map(|handle| logical_meta(handle.metadata()))
        .collect();

    let invocation = O::infer_invocation_typed(attributes, logical, expected)?;

    validate_execution_payload::<O>(&invocation, payload)?;

    admit_invocation::<O, B>(context, &invocation, inputs)?;

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
