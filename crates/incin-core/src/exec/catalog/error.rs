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

/// The edit that satisfies an operand-count contract.
fn arity_remedy(
    actual: usize,
    expected: &core::ops::RangeInclusive<usize>,
) -> alloc::string::String {
    use alloc::format;
    if actual < *expected.start() {
        format!("pass {} more operand(s)", expected.start() - actual)
    } else {
        format!("pass {} fewer operand(s)", actual - expected.end())
    }
}

/// The edit that satisfies a rank contract.
fn rank_remedy(
    actual: usize,
    expected: &core::ops::RangeInclusive<usize>,
) -> alloc::string::String {
    use alloc::format;
    if actual < *expected.start() {
        format!(
            "add {} axis/axes with `unsqueeze`, or pass an operand of rank {}",
            expected.start() - actual,
            expected.start()
        )
    } else {
        format!(
            "remove {} axis/axes with `squeeze` or `reshape`, or pass an operand of rank {}",
            actual - expected.end(),
            expected.end()
        )
    }
}

/// A concrete edit for the attribute contracts a caller trips most often.
///
/// Keyed on the attribute name the contract itself reports, so the advice
/// cannot drift from the check: a name that stops being produced stops being
/// matched, and falls back to naming where the rule lives. A generic
/// "check the attribute" would be true of every arm and useful in none.
fn attribute_remedy(attribute: &str) -> &'static str {
    match attribute {
        "index dtype" => {
            "an index operand addresses positions, so it must be an integer tensor: \
             build it as `Tensor::<_, _, i64>` rather than letting it default to a float"
        }
        "dtype" => {
            "give every operand the same dtype, converting one with `.to_dtype(..)` \
             if they genuinely differ"
        }
        "shape" | "rank" | "input shape" => {
            "adjust the operand geometry named above; the rule is enforced in this \
             operation's `AttributeContract::validate`"
        }
        "spatial parameters" => "kernel, stride, dilation and groups must each be non-zero",
        _ => "correct the attribute named above; the rule stated is the contract",
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
                "{operation}: input arity {actual} is outside {expected:?}\n  accepts: {} to {} operands\n  received: {actual}\n  fix: {}",
                expected.start(),
                expected.end(),
                arity_remedy(*actual, expected),
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
                "{operation}: input {input} rank {actual} is outside {expected:?}\n  operand {input}: rank {actual}\n  accepts: rank {} to {}\n  fix: {}",
                expected.start(),
                expected.end(),
                rank_remedy(*actual, expected),
            ),
            Self::DeviceMismatch {
                operation,
                input,
                expected,
                actual,
            } => write!(
                f,
                "{operation}: input {input} device {actual:?} does not match {expected:?}\n  operand 0 is on: {expected:?}\n  operand {input} is on: {actual:?}\n  rule: every operand of one call must live on one device;                  nothing here moves data across devices implicitly\n  fix: move one operand with `.to_device(..)` so both agree,                  then repeat the call"
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
            } => write!(
                f,
                "{operation}: invalid {attribute}: {reason}\n  attribute: {attribute}\n  rule: {reason}\n  fix: {}",
                attribute_remedy(attribute)
            ),
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
    /// One-hot depth, when this set carries one.
    fn depth(&self) -> Option<usize> {
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
