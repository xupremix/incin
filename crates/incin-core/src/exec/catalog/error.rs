use super::*;

/// Metadata validation error emitted before storage access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorError {
    Arity {
        operation: OperationKind,
        expected: core::ops::RangeInclusive<usize>,
        actual: usize,
    },
    OutputArity {
        operation: OperationKind,
        expected: core::ops::RangeInclusive<usize>,
        actual: usize,
    },
    Rank {
        operation: OperationKind,
        input: usize,
        expected: core::ops::RangeInclusive<usize>,
        actual: usize,
    },
    DeviceMismatch {
        operation: OperationKind,
        input: usize,
        expected: DeviceId,
        actual: DeviceId,
    },
    MissingCatalogEntry {
        operation: OperationKind,
    },
    MissingInference {
        operation: OperationKind,
    },
    InvalidAttribute {
        operation: OperationKind,
        attribute: &'static str,
        reason: &'static str,
    },
    Shape(crate::shapes::ShapeError),
    MetadataMismatch {
        operation: OperationKind,
        output: usize,
        field: &'static str,
    },
    PayloadKind {
        operation: OperationKind,
        expected: &'static str,
    },
    PayloadDTypeMismatch {
        operation: OperationKind,
        expected: DTypeDescriptor,
        actual: DTypeDescriptor,
    },
    PayloadByteLength {
        operation: OperationKind,
        expected: usize,
        actual: usize,
    },
    /// A creation operation was dispatched without its borrowed bytes.
    PayloadMissing {
        operation: OperationKind,
    },
    /// A non-creation operation received an execution payload.
    UnexpectedPayload {
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

pub trait AttributeContract {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError>;
    fn declared_shape(&self) -> Option<&[usize]> {
        None
    }
    fn declared_dtype(&self) -> Option<DTypeDescriptor> {
        None
    }
    fn declared_device(&self) -> Option<DeviceId> {
        None
    }
    fn axis(&self) -> Option<usize> {
        None
    }
    fn loss_reduction(&self) -> Option<LossReduction> {
        None
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        None
    }
    fn expected_output_count(&self, _inputs: &[LogicalTensorMeta]) -> Option<usize> {
        None
    }
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
