use super::*;

/// Open operation contract for downstream static execution.
pub trait Operation: Clone + fmt::Debug + 'static {
    /// Typed attribute payload carried by this descriptor.
    type Attributes: Clone
        + fmt::Debug
        + PartialEq
        + core::any::Any
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>;

    /// Stable key identifying this operation across registries.
    const KEY: OperationKey;

    /// Infers output metadata from checked logical input metadata.
    fn infer_outputs(
        attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError>;

    /// Infer a validated invocation through the open operation contract.
    fn infer_invocation(
        attributes: Self::Attributes,
        inputs: Vec<LogicalTensorMeta>,
    ) -> Result<ValidatedInvocation<Self>, DescriptorError>
    where
        Self: Sized,
    {
        ValidatedInvocation::<Self>::infer_custom_runtime(attributes, inputs)
    }

    /// Shape-specialized form of [`Self::infer_invocation`].
    fn infer_invocation_typed<S: crate::shapes::Shape>(
        attributes: Self::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        expected: &crate::shapes::ShapeValue<S>,
    ) -> Result<ValidatedInvocation<Self>, DescriptorError>
    where
        Self: Sized,
    {
        ValidatedInvocation::<Self>::infer_custom_typed(attributes, inputs, expected)
    }
}

mod private {
    pub trait Sealed {}
}

/// A catalog operation with its exact typed attribute set.
pub trait CanonicalOperation: private::Sealed + Operation {
    /// Exact catalog identity of this operation.
    const ID: OperationKind;
}

incin_operation_catalog!(define_catalog);

macro_rules! seal_operations {
    ($(($variant:ident, $name:literal, $family:ident, $profile:ident, $attrs:ident, $min:expr, $max:expr, $legacy:literal),)*) => {$(impl private::Sealed for op::$variant {})*};
}
incin_operation_catalog!(seal_operations);

/// Concrete typed descriptor. `Descriptor<op::Add>` and
/// `Descriptor<op::Softmax>` are different, non-interchangeable Rust types.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "O::Attributes: serde::Serialize",
    deserialize = "O::Attributes: serde::Deserialize<'de>"
))]
pub struct Descriptor<O: Operation> {
    pub(super) attributes: O::Attributes,
    pub(super) inputs: Vec<LogicalTensorMeta>,
    pub(super) outputs: Vec<LogicalTensorMeta>,
    pub(super) identity: crate::exec::OperationIdentity,
    pub(super) marker: PhantomData<fn() -> O>,
}

impl<O: Operation> crate::exec::spec::ExecutionDescriptor for Descriptor<O> {
    fn output_shape(&self) -> Option<&ShapeBuf> {
        self.outputs
            .first()
            .and_then(|output| output.shape.as_ref())
    }
}

/// Supplies the graph identity used by the tracing execution adapter.
///
/// This is derived from the same canonical operation identity as capability
/// admission and descriptor execution. It is not a second operation catalog.
pub trait TraceDescriptor: crate::exec::spec::ExecutionDescriptor {
    /// Identity recorded into telemetry traces.
    fn trace_identity(&self) -> crate::exec::OperationIdentity;

    #[cfg(feature = "std")]
    /// Renders attributes for telemetry consumption.
    fn trace_attributes(
        &self,
    ) -> core::result::Result<
        alloc::collections::BTreeMap<alloc::string::String, crate::graph::AttributeValue>,
        &'static str,
    >;

    /// Serializes the descriptor body for trace replay.
    fn trace_descriptor_payload(
        &self,
    ) -> core::result::Result<Option<crate::graph::DescriptorPayload>, &'static str>;

    /// Names the output dtype for trace consumers.
    fn trace_output_dtype(
        &self,
        inputs: &[crate::exec::request::TensorHandle<'_>],
    ) -> Option<DTypeDescriptor>;
}

impl<O: Operation> TraceDescriptor for Descriptor<O> {
    fn trace_identity(&self) -> crate::exec::OperationIdentity {
        self.identity.clone()
    }

    #[cfg(feature = "std")]
    fn trace_attributes(
        &self,
    ) -> core::result::Result<
        alloc::collections::BTreeMap<alloc::string::String, crate::graph::AttributeValue>,
        &'static str,
    >
    where
        O::Attributes: serde::Serialize,
    {
        project_trace_attributes(&self.attributes)
    }

    fn trace_descriptor_payload(
        &self,
    ) -> core::result::Result<Option<crate::graph::DescriptorPayload>, &'static str> {
        #[cfg(not(feature = "std"))]
        {
            Err("descriptor capture requires the std serialization feature")
        }
        #[cfg(feature = "std")]
        {
            let payload = postcard::to_allocvec(self)
                .map_err(|_| "canonical descriptor serialization failed")?;
            Ok(Some(crate::graph::DescriptorPayload {
                schema: crate::exec::DescriptorSchemaVersion::CURRENT.get(),
                payload,
            }))
        }
    }

    fn trace_output_dtype(
        &self,
        inputs: &[crate::exec::request::TensorHandle<'_>],
    ) -> Option<DTypeDescriptor> {
        let OperationIdentity::Builtin(operation) = self.identity else {
            return self
                .outputs
                .first()
                .and_then(|output| output.dtype)
                .or_else(|| inputs.first().map(|input| input.metadata().dtype));
        };
        match operation {
            OperationKind::CmpEq
            | OperationKind::CmpNe
            | OperationKind::CmpLt
            | OperationKind::CmpLe
            | OperationKind::CmpGt
            | OperationKind::CmpGe
            | OperationKind::LogicalAnd
            | OperationKind::LogicalOr
            | OperationKind::LogicalNot => Some(DTypeId::Bool.into()),
            _ => self
                .outputs
                .first()
                .and_then(|output| output.dtype)
                .or_else(|| inputs.first().map(|input| input.metadata().dtype)),
        }
    }
}

impl<O: Operation> Descriptor<O> {
    #[must_use]
    /// Stable key of the captured operation.
    pub const fn key(&self) -> OperationKey {
        O::KEY
    }

    #[must_use]
    /// Typed attribute payload.
    pub const fn attributes(&self) -> &O::Attributes {
        &self.attributes
    }

    #[must_use]
    /// Validated output metadata.
    pub fn outputs(&self) -> &[LogicalTensorMeta] {
        &self.outputs
    }

    #[must_use]
    /// Validated input metadata.
    pub fn inputs(&self) -> &[LogicalTensorMeta] {
        &self.inputs
    }

    #[must_use]
    /// Operation identity proven at construction.
    pub const fn identity(&self) -> &crate::exec::OperationIdentity {
        &self.identity
    }
}

impl<O: CanonicalOperation> Descriptor<O>
where
    O::Attributes: AttributeContract,
{
    #[must_use]
    /// Operation this payload belongs to.
    pub const fn operation(&self) -> OperationKind {
        O::ID
    }

    /// Validate a runtime invocation and attach the resulting dynamic proof.
    ///
    /// This is the public construction seam for backend authors and other
    /// framework boundaries that need to execute an exact descriptor without
    /// reaching into frontend shape rules. Typed tensor frontends use their
    /// stronger proof path; this method intentionally records only dynamic
    /// knowledge.
    pub fn infer_runtime(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
    ) -> Result<crate::exec::Validated<Self>, DescriptorError> {
        attributes.validate(O::ID, &inputs)?;
        let outputs = O::infer_outputs(&attributes, &inputs)?;
        Ok(crate::exec::Validated::new(
            Self {
                attributes,
                inputs,
                outputs,
                identity: crate::exec::OperationIdentity::Builtin(O::ID),
                marker: PhantomData,
            },
            crate::exec::ProofLevel::Dynamic,
        ))
    }

    /// Revalidate a captured descriptor before execution.
    pub fn revalidate(&self) -> Result<crate::exec::Validated<Self>, DescriptorError> {
        self.attributes.validate(O::ID, &self.inputs)?;
        let outputs = O::infer_outputs(&self.attributes, &self.inputs)?;
        if outputs != self.outputs {
            return Err(DescriptorError::MetadataMismatch {
                operation: O::ID,
                output: 0,
                field: "outputs",
            });
        }
        Ok(crate::exec::Validated::new(
            self.clone(),
            crate::exec::ProofLevel::Dynamic,
        ))
    }
}

/// Storage-free serialized descriptor capture. The exact identity is outside
/// the payload so decoding as the wrong descriptor type fails closed.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CapturedDescriptor {
    operation: OperationKind,
    schema: u32,
    pub(super) payload: Vec<u8>,
}

#[cfg(feature = "std")]
#[derive(Debug)]
/// Failures while capturing or decoding a descriptor payload.
pub enum DescriptorCaptureError {
    /// Descriptor identity disagrees with the requested operation.
    Identity {
        /// Operation the payload claims to be.
        expected: OperationKind,
        /// Operation the caller asked for.
        actual: OperationKind,
    },
    /// A custom-operation key collided with a builtin expectation.
    CustomIdentity {
        /// Operation the payload claims to be.
        expected: OperationKind,
        /// Custom key the payload carries.
        actual: OperationKey,
    },
    /// Descriptor schema version mismatch.
    Schema {
        /// Schema version this build speaks.
        expected: u32,
        /// Schema version found in the payload.
        actual: u32,
    },
    /// Payload failed to serialize.
    Encode(postcard::Error),
    /// Payload failed to deserialize.
    Decode(postcard::Error),
}

#[cfg(feature = "std")]
impl fmt::Display for DescriptorCaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity { expected, actual } => write!(
                f,
                "captured descriptor identity {actual} does not match {expected}"
            ),
            Self::CustomIdentity { expected, actual } => write!(
                f,
                "captured descriptor identity {actual:?} does not match builtin {expected}"
            ),
            Self::Schema { expected, actual } => write!(
                f,
                "captured descriptor schema v{actual} does not match v{expected}"
            ),
            Self::Encode(error) => write!(f, "could not encode descriptor capture: {error}"),
            Self::Decode(error) => write!(f, "could not decode descriptor capture: {error}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DescriptorCaptureError {}

#[cfg(feature = "std")]
impl CapturedDescriptor {
    /// Wraps raw bytes as a capture for later decoding.
    pub fn from_payload(operation: OperationKind, schema: u32, payload: Vec<u8>) -> Self {
        Self {
            operation,
            schema,
            payload,
        }
    }

    /// Captures a validated descriptor into a portable payload.
    pub fn capture<O: CanonicalOperation>(
        descriptor: &Descriptor<O>,
    ) -> Result<Self, DescriptorCaptureError>
    where
        O::Attributes: AttributeContract,
    {
        let payload = postcard::to_allocvec(descriptor).map_err(DescriptorCaptureError::Encode)?;
        Ok(Self {
            operation: O::ID,
            schema: crate::exec::DescriptorSchemaVersion::CURRENT.get(),
            payload,
        })
    }

    #[must_use]
    /// Operation this payload belongs to.
    pub const fn operation(&self) -> OperationKind {
        self.operation
    }

    #[must_use]
    /// Schema version the payload was encoded with.
    pub const fn schema(&self) -> u32 {
        self.schema
    }

    #[must_use]
    /// Raw encoded descriptor bytes.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Decodes the payload back into a validated descriptor.
    pub fn decode<O: CanonicalOperation>(&self) -> Result<Descriptor<O>, DescriptorCaptureError>
    where
        O::Attributes: AttributeContract,
    {
        if self.operation != O::ID {
            return Err(DescriptorCaptureError::Identity {
                expected: O::ID,
                actual: self.operation,
            });
        }
        let expected = crate::exec::DescriptorSchemaVersion::CURRENT.get();
        if self.schema != expected {
            return Err(DescriptorCaptureError::Schema {
                expected,
                actual: self.schema,
            });
        }
        let descriptor: Descriptor<O> =
            postcard::from_bytes(&self.payload).map_err(DescriptorCaptureError::Decode)?;
        if descriptor.identity != crate::exec::OperationIdentity::Builtin(O::ID) {
            return match descriptor.identity {
                crate::exec::OperationIdentity::Builtin(actual) => {
                    Err(DescriptorCaptureError::Identity {
                        expected: O::ID,
                        actual,
                    })
                }
                crate::exec::OperationIdentity::Custom(actual) => {
                    Err(DescriptorCaptureError::CustomIdentity {
                        expected: O::ID,
                        actual,
                    })
                }
            };
        }
        Ok(descriptor)
    }
}
