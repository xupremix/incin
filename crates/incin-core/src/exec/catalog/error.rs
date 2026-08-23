use super::*;

/// Metadata validation error emitted before storage access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorError {
    /// Wrong number of inputs for an operation.
    Arity {
        /// Operation this query targets.
        operation: OperationKind,
        /// Inclusive range the descriptor accepts.
        expected: core::ops::RangeInclusive<usize>,
        /// Value actually observed.
        actual: usize,
    },
    /// Wrong number of outputs produced by a backend.
    OutputArity {
        /// Operation this query targets.
        operation: OperationKind,
        /// Inclusive range the descriptor accepts.
        expected: core::ops::RangeInclusive<usize>,
        /// Value actually observed.
        actual: usize,
    },
    /// An input's rank fell outside the accepted range.
    Rank {
        /// Operation this query targets.
        operation: OperationKind,
        /// Zero-based index of the offending input.
        input: usize,
        /// Inclusive range the descriptor accepts.
        expected: core::ops::RangeInclusive<usize>,
        /// Value actually observed.
        actual: usize,
    },
    /// Two inputs live on different devices.
    DeviceMismatch {
        /// Operation this query targets.
        operation: OperationKind,
        /// Zero-based index of the offending input.
        input: usize,
        /// Device the contract requires.
        expected: DeviceId,
        /// Device storage actually resides on.
        actual: DeviceId,
    },
    /// The operation has no catalog row; it cannot be validated.
    MissingCatalogEntry {
        /// Operation this query targets.
        operation: OperationKind,
    },
    /// The catalog row cannot infer what this request needs.
    MissingInference {
        /// Operation this query targets.
        operation: OperationKind,
    },
    /// An attribute value failed validation.
    InvalidAttribute {
        /// Operation this query targets.
        operation: OperationKind,
        /// Name of the rejected attribute.
        attribute: &'static str,
        /// Bounded explanation of the failure.
        reason: &'static str,
    },
    /// A shape rule rejected the geometry.
    Shape(crate::shapes::ShapeError),
    /// A declared output disagreed with computed metadata.
    MetadataMismatch {
        /// Operation this query targets.
        operation: OperationKind,
        /// Zero-based index of the disagreeing output.
        output: usize,
        /// Which metadata field disagreed.
        field: &'static str,
    },
    /// Creation payload had the wrong kind.
    PayloadKind {
        /// Operation this query targets.
        operation: OperationKind,
        /// Kind name the descriptor required.
        expected: &'static str,
    },
    /// Creation payload dtype differs from the descriptor's.
    PayloadDTypeMismatch {
        /// Operation this query targets.
        operation: OperationKind,
        /// Dtype metadata promised.
        expected: DTypeDescriptor,
        /// Dtype actually present.
        actual: DTypeDescriptor,
    },
    /// Creation payload byte length disagrees with shape math.
    PayloadByteLength {
        /// Operation this query targets.
        operation: OperationKind,
        /// Byte length the descriptor computed.
        expected: usize,
        /// Value actually observed.
        actual: usize,
    },
    /// A creation operation was dispatched without its borrowed bytes.
    PayloadMissing {
        /// Operation this query targets.
        operation: OperationKind,
    },
    /// A non-creation operation received an execution payload.
    UnexpectedPayload {
        /// Operation this query targets.
        operation: OperationKind,
    },
}

impl From<crate::shapes::ShapeError> for DescriptorError {
    fn from(error: crate::shapes::ShapeError) -> Self {
        Self::Shape(error)
    }
}

impl fmt::Display for DescriptorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arity {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation}: input arity {actual} is outside {expected:?}"
            ),
            Self::OutputArity {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation}: output arity {actual} is outside {expected:?}"
            ),
            Self::Rank {
                operation,
                input,
                expected,
                actual,
            } => write!(
                f,
                "{operation}: input {input} rank {actual} is outside {expected:?}"
            ),
            Self::DeviceMismatch {
                operation,
                input,
                expected,
                actual,
            } => write!(
                f,
                "{operation}: input {input} device {actual:?} does not match {expected:?}"
            ),
            Self::MissingCatalogEntry { operation } => write!(
                f,
                "{operation}: exact operation is absent from the canonical catalog"
            ),
            Self::MissingInference { operation } => write!(
                f,
                "{operation}: exact output metadata inference is not implemented"
            ),
            Self::InvalidAttribute {
                operation,
                attribute,
                reason,
            } => write!(f, "{operation}: invalid {attribute}: {reason}"),
            Self::Shape(error) => fmt::Display::fmt(error, f),
            Self::MetadataMismatch {
                operation,
                output,
                field,
            } => write!(
                f,
                "{operation}: output {output} {field} disagrees with inferred metadata"
            ),
            Self::PayloadKind {
                operation,
                expected,
            } => write!(f, "{operation}: expected {expected} creation payload"),
            Self::PayloadDTypeMismatch {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation}: payload dtype {actual:?} does not match {expected:?}"
            ),
            Self::PayloadByteLength {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation}: payload byte length {actual} does not match {expected}"
            ),
            Self::PayloadMissing { operation } => {
                write!(f, "{operation}: creation payload is missing")
            }
            Self::UnexpectedPayload { operation } => {
                write!(f, "{operation}: unexpected execution payload")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DescriptorError {}

/// Validation implemented by each concrete typed attribute schema.
#[cfg(feature = "std")]
pub(super) fn project_trace_attributes<T: serde::Serialize + ?Sized>(
    value: &T,
) -> core::result::Result<
    alloc::collections::BTreeMap<alloc::string::String, crate::graph::AttributeValue>,
    &'static str,
> {
    let value =
        serde_json::to_value(value).map_err(|_| "canonical attribute serialization failed")?;
    let serde_json::Value::Object(fields) = value else {
        return Ok(alloc::collections::BTreeMap::new());
    };
    let mut attributes = alloc::collections::BTreeMap::new();
    for (name, value) in fields {
        let converted = match value {
            serde_json::Value::Number(number) => number
                .as_i64()
                .map(crate::graph::AttributeValue::Int)
                .or_else(|| {
                    number
                        .as_f64()
                        .map(|value| crate::graph::AttributeValue::Float(value as f32))
                }),
            serde_json::Value::String(value) => Some(crate::graph::AttributeValue::String(value)),
            serde_json::Value::Array(values) if values.iter().all(serde_json::Value::is_number) => {
                if values.iter().all(|value| value.as_i64().is_some()) {
                    Some(crate::graph::AttributeValue::Ints(
                        values
                            .iter()
                            .filter_map(serde_json::Value::as_i64)
                            .collect(),
                    ))
                } else {
                    Some(crate::graph::AttributeValue::Floats(
                        values
                            .iter()
                            .filter_map(serde_json::Value::as_f64)
                            .map(|value| value as f32)
                            .collect(),
                    ))
                }
            }
            serde_json::Value::Bool(value) => {
                Some(crate::graph::AttributeValue::Int(i64::from(value)))
            }
            serde_json::Value::Array(_)
            | serde_json::Value::Null
            | serde_json::Value::Object(_) => None,
        };
        if let Some(value) = converted {
            attributes.insert(name, value);
        }
    }
    Ok(attributes)
}

/// Per-operation validation over its typed attribute set.
pub trait AttributeContract {
    /// Validates this attribute set against its contract, returning structured errors.
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError>;
    /// Pinned output shape, when declared.
    fn declared_shape(&self) -> Option<&[usize]> {
        None
    }
    /// Pinned dtype, when declared.
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        None
    }
    /// Pinned device, when declared.
    fn declared_device(&self) -> Option<DeviceId> {
        None
    }
    /// Axis argument, when this set carries one.
    fn axis(&self) -> Option<usize> {
        None
    }
    /// Loss reduction mode, when present.
    fn loss_reduction(&self) -> Option<LossReduction> {
        None
    }
    /// Shape-transform description, when applicable.
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        None
    }
    /// Declared output count for these inputs, when fixed.
    fn expected_output_count(&self, _inputs: &[LogicalTensorMeta]) -> Option<usize> {
        None
    }
    /// Whether a bias input participates, when declared.
    fn optional_bias(&self) -> Option<bool> {
        None
    }

    /// Projects serializable scalar and array attributes for graph consumers.
    /// The typed attribute struct remains the semantic source of truth.
    #[cfg(feature = "std")]
    fn trace_attributes(
        &self,
    ) -> core::result::Result<
        alloc::collections::BTreeMap<alloc::string::String, crate::graph::AttributeValue>,
        &'static str,
    >
    where
        Self: serde::Serialize,
    {
        project_trace_attributes(self)
    }
}
