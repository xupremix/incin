use super::*;

/// One immutable row derived from the authoritative operation declaration.
pub struct OperationCatalogEntry {
    /// Exact catalog operation this row describes.
    pub operation: OperationKind,
    /// Snake_case catalog name.
    pub name: &'static str,
    /// Family grouping used for dispatch profiles.
    pub family: OperationKind,
    /// Compute profile classification.
    pub profile: SemanticProfile,
    /// Descriptor type name owning this row.
    pub descriptor: &'static str,
    /// Attribute type name for this row.
    pub attributes: &'static str,
    /// Accepted input count range.
    pub input_arity: core::ops::RangeInclusive<usize>,
    /// Produced output count range.
    pub output_arity: core::ops::RangeInclusive<usize>,
    /// Supported rank range.
    pub accepted_ranks: core::ops::RangeInclusive<usize>,
    /// How inputs combine element-wise.
    pub broadcasting: BroadcastingRule,
    /// Input dtype policy.
    pub dtype: DTypeRule,
    /// Output dtype policy.
    pub output: OutputRule,
    /// Whether all inputs must share one device.
    pub same_device: bool,
    /// Behavior on zero-element inputs.
    pub empty: EmptyRule,
    /// Numeric-domain guarantees.
    pub numeric: NumericRule,
    /// Derivative contract.
    pub gradient: GradientRule,
    /// Whether repeated runs are bitwise reproducible.
    pub deterministic: bool,
    /// Output layout behavior.
    pub layout: LayoutRule,
    /// Legacy method this row superseded.
    pub legacy_source: &'static str,
    /// Whether the op can be captured into compiled plans.
    pub capture_eligible: bool,
    /// Where the operation executes.
    pub site: ExecutionSite,
}

/// Classify one operation by where its result is produced.
///
/// The default is [`ExecutionSite::Kernel`], which is the fail-closed answer: a
/// newly declared operation is assumed to owe an executor. Being wrong that way
/// leaves an unmigrated operation visible in the migration report; being wrong
/// the other way would silently excuse it.
const fn execution_site(operation: OperationKind) -> ExecutionSite {
    match operation {
        // Each of these calls its backend method with `&mut` on a `Var<K>` and
        // returns nothing.
        OperationKind::SgdStep
        | OperationKind::AdamStep
        | OperationKind::AdamWStep => ExecutionSite::Mutation,
        OperationKind::ToDevice => ExecutionSite::DeviceTransfer,
        OperationKind::RequireGrad | OperationKind::Detach | OperationKind::Backward => {
            ExecutionSite::GraphState
        }
        // Declared by `OutputRule::HostValue` too; kept as an explicit list
        // because the two answer different questions and a future host-value
        // operation might well be executable.
        OperationKind::TensorToBytes
        | OperationKind::ToHostFloatScalar
        | OperationKind::ToHostFloatVec
        | OperationKind::ToHostIntScalar
        | OperationKind::ToHostIntVec => ExecutionSite::HostReadback,
        OperationKind::Sample | OperationKind::Rnn | OperationKind::Lstm => ExecutionSite::Composed,
        OperationKind::TensorFromData
        | OperationKind::TensorFromBytes
        | OperationKind::Zeros
        | OperationKind::Ones
        | OperationKind::UniformRandom
        | OperationKind::NormalRandom
        | OperationKind::VariableZeros
        | OperationKind::VariableOnes
        | OperationKind::VariableUniformRandom
        | OperationKind::VariableNormalRandom
        | OperationKind::Full
        | OperationKind::Arange
        | OperationKind::Linspace => ExecutionSite::Creation,
        _ => ExecutionSite::Kernel,
    }
}

const fn profile_semantics(
    profile: SemanticProfile,
) -> (
    BroadcastingRule,
    DTypeRule,
    OutputRule,
    EmptyRule,
    NumericRule,
    GradientRule,
    LayoutRule,
) {
    use SemanticProfile::*;
    match profile {
        Creation => (
            BroadcastingRule::None,
            DTypeRule::ExplicitOutput,
            OutputRule::Created,
            EmptyRule::Allowed,
            NumericRule::TypedContract,
            GradientRule::None,
            LayoutRule::FreshContiguous,
        ),
        UnaryFloat => (
            BroadcastingRule::None,
            DTypeRule::Floating,
            OutputRule::Preserve,
            EmptyRule::Allowed,
            NumericRule::IeeePropagate,
            GradientRule::Defined,
            LayoutRule::FreshContiguous,
        ),
        BinaryBroadcast => (
            BroadcastingRule::Numpy,
            DTypeRule::NumericSame,
            OutputRule::Broadcast,
            EmptyRule::Allowed,
            NumericRule::IeeePropagate,
            GradientRule::Defined,
            LayoutRule::FreshContiguous,
        ),
        Comparison => (
            BroadcastingRule::Numpy,
            DTypeRule::BooleanResult,
            OutputRule::Broadcast,
            EmptyRule::Allowed,
            NumericRule::IeeePropagate,
            GradientRule::None,
            LayoutRule::FreshContiguous,
        ),
        Logical => (
            BroadcastingRule::Numpy,
            DTypeRule::Boolean,
            OutputRule::Broadcast,
            EmptyRule::Allowed,
            NumericRule::NotApplicable,
            GradientRule::None,
            LayoutRule::FreshContiguous,
        ),
        Shape => (
            BroadcastingRule::TypedContract,
            DTypeRule::Preserve,
            OutputRule::ShapeAttributes,
            EmptyRule::Allowed,
            NumericRule::NotApplicable,
            GradientRule::Defined,
            LayoutRule::ViewWhenPossible,
        ),
        Selection => (
            BroadcastingRule::Numpy,
            DTypeRule::TypedContract,
            OutputRule::Broadcast,
            EmptyRule::Allowed,
            NumericRule::TypedContract,
            GradientRule::Piecewise,
            LayoutRule::FreshContiguous,
        ),
        Indexing => (
            BroadcastingRule::None,
            DTypeRule::TypedContract,
            OutputRule::Indexing,
            EmptyRule::TypedContract,
            NumericRule::CheckedInteger,
            GradientRule::Defined,
            LayoutRule::FreshContiguous,
        ),
        MatMul => (
            BroadcastingRule::TypedContract,
            DTypeRule::Floating,
            OutputRule::MatMul,
            EmptyRule::IdentityOrDefined,
            NumericRule::StableAccumulation,
            GradientRule::Defined,
            LayoutRule::FreshContiguous,
        ),
        Attention => (
            BroadcastingRule::TypedContract,
            DTypeRule::Floating,
            OutputRule::TypedInference,
            EmptyRule::TypedContract,
            NumericRule::StableAccumulation,
            GradientRule::Defined,
            LayoutRule::FreshContiguous,
        ),
        Reduction => (
            BroadcastingRule::None,
            DTypeRule::Floating,
            OutputRule::Reduction,
            EmptyRule::RejectedWhenReductionIsEmpty,
            NumericRule::StableAccumulation,
            GradientRule::Defined,
            LayoutRule::FreshContiguous,
        ),
        IndexReduction => (
            BroadcastingRule::None,
            DTypeRule::IndexResult,
            OutputRule::Reduction,
            EmptyRule::RejectedWhenReductionIsEmpty,
            NumericRule::IeeePropagate,
            GradientRule::None,
            LayoutRule::FreshContiguous,
        ),
        Scan => (
            BroadcastingRule::None,
            DTypeRule::Preserve,
            OutputRule::Preserve,
            EmptyRule::Allowed,
            NumericRule::StableAccumulation,
            GradientRule::Defined,
            LayoutRule::FreshContiguous,
        ),
        Normalization => (
            BroadcastingRule::TypedContract,
            DTypeRule::Floating,
            OutputRule::Preserve,
            EmptyRule::RejectedWhenReductionIsEmpty,
            NumericRule::StableAccumulation,
            GradientRule::Defined,
            LayoutRule::FreshContiguous,
        ),
        Module | Composite => (
            BroadcastingRule::TypedContract,
            DTypeRule::TypedContract,
            OutputRule::TypedInference,
            EmptyRule::TypedContract,
            NumericRule::TypedContract,
            GradientRule::Defined,
            LayoutRule::FreshContiguous,
        ),
        Loss => (
            BroadcastingRule::TypedContract,
            DTypeRule::Floating,
            OutputRule::Reduction,
            EmptyRule::RejectedWhenReductionIsEmpty,
            NumericRule::StableAccumulation,
            GradientRule::Defined,
            LayoutRule::FreshContiguous,
        ),
        Quantized => (
            BroadcastingRule::TypedContract,
            DTypeRule::Quantized,
            OutputRule::TypedInference,
            EmptyRule::TypedContract,
            NumericRule::TypedContract,
            GradientRule::None,
            LayoutRule::FreshContiguous,
        ),
        Optimizer => (
            BroadcastingRule::None,
            DTypeRule::Floating,
            OutputRule::TypedInference,
            EmptyRule::Allowed,
            NumericRule::TypedContract,
            GradientRule::None,
            LayoutRule::FreshContiguous,
        ),
        Transfer => (
            BroadcastingRule::None,
            DTypeRule::TypedContract,
            OutputRule::TypedInference,
            EmptyRule::Allowed,
            NumericRule::NotApplicable,
            GradientRule::TypedContract,
            LayoutRule::PreserveOrMaterialize,
        ),
        Autograd => (
            BroadcastingRule::None,
            DTypeRule::Preserve,
            OutputRule::Preserve,
            EmptyRule::Allowed,
            NumericRule::NotApplicable,
            GradientRule::None,
            LayoutRule::PreserveOrMaterialize,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) const fn entry(
    operation: OperationKind,
    name: &'static str,
    family: OperationKind,
    profile: SemanticProfile,
    descriptor: &'static str,
    attributes: &'static str,
    min_arity: usize,
    max_arity: usize,
    legacy_source: &'static str,
) -> OperationCatalogEntry {
    let (mut broadcasting, dtype, mut output, mut empty, numeric, mut gradient, layout) =
        profile_semantics(profile);
    let accepted_ranks = match operation {
        OperationKind::MatMulExact | OperationKind::QuantizedMatMul => 2..=usize::MAX,
        OperationKind::Dot | OperationKind::Outer => 1..=1,
        OperationKind::BatchedMatMul | OperationKind::Addmm => 2..=3,
        OperationKind::Rnn | OperationKind::Lstm => 2..=3,
        OperationKind::Conv1dExact => 2..=3,
        OperationKind::Conv2dExact
        | OperationKind::ConvTranspose2d
        | OperationKind::MaxPool2d
        | OperationKind::AvgPool2d
        | OperationKind::AdaptiveAvgPool2dExact => 3..=4,
        OperationKind::Softmax
        | OperationKind::LogSoftmax
        | OperationKind::GroupNorm
        | OperationKind::InstanceNorm
        | OperationKind::LayerNorm
        | OperationKind::BatchNorm
        | OperationKind::RmsNorm => 1..=usize::MAX,
        _ => 0..=usize::MAX,
    };
    let output_arity = match operation {
        OperationKind::TopK => 2..=2,
        OperationKind::Chunk | OperationKind::Split => 0..=usize::MAX,
        OperationKind::Rnn => 2..=2,
        OperationKind::Lstm => 3..=3,
        OperationKind::ToHostFloatScalar
        | OperationKind::ToHostFloatVec
        | OperationKind::ToHostIntScalar
        | OperationKind::ToHostIntVec
        | OperationKind::TensorToBytes => 0..=0,
        OperationKind::Backward => 0..=0,
        OperationKind::AdamStep | OperationKind::AdamWStep => 3..=3,
        _ => 1..=1,
    };
    if matches!(
        operation,
        OperationKind::BroadcastAs | OperationKind::BroadcastLeft
    ) {
        broadcasting = BroadcastingRule::ExplicitTarget;
    }
    if matches!(operation, OperationKind::ToDType) {
        output = OutputRule::ExplicitDType;
    } else if matches!(operation, OperationKind::ToDevice) {
        output = OutputRule::Preserve;
    } else if matches!(
        operation,
        OperationKind::Dot
            | OperationKind::Outer
            | OperationKind::Addmm
            | OperationKind::ScaledDotProductAttention
            | OperationKind::Linear
    ) {
        output = OutputRule::TypedInference;
    } else if matches!(operation, OperationKind::QuantizedMatMul) {
        output = OutputRule::MatMul;
    } else if matches!(
        operation,
        OperationKind::Conv1dExact
            | OperationKind::Conv2dExact
            | OperationKind::ConvTranspose2d
            | OperationKind::MaxPool2d
            | OperationKind::AvgPool2d
            | OperationKind::AdaptiveAvgPool2dExact
            | OperationKind::Rnn
            | OperationKind::Lstm
    ) {
        output = OutputRule::ShapeAttributes;
    } else if matches!(
        operation,
        OperationKind::LayerNorm
            | OperationKind::BatchNorm
            | OperationKind::GroupNorm
            | OperationKind::InstanceNorm
            | OperationKind::RmsNorm
            | OperationKind::Dropout
            | OperationKind::MaskedFill
            | OperationKind::Scatter
            | OperationKind::ScatterAdd
    ) {
        output = OutputRule::Preserve;
    } else if matches!(
        operation,
        OperationKind::ToHostFloatScalar
            | OperationKind::ToHostFloatVec
            | OperationKind::ToHostIntScalar
            | OperationKind::ToHostIntVec
            | OperationKind::TensorToBytes
    ) {
        output = OutputRule::HostValue;
    }
    if matches!(
        operation,
        OperationKind::Step
            | OperationKind::Sign
            | OperationKind::Floor
            | OperationKind::Ceil
            | OperationKind::Round
            | OperationKind::Trunc
            | OperationKind::Frac
    ) {
        gradient = GradientRule::Undefined;
    }
    if matches!(operation, OperationKind::Scatter) {
        gradient = GradientRule::Undefined;
    } else if matches!(operation, OperationKind::ScatterAdd) {
        // The derivative `scatter` cannot define is the one addition makes
        // obvious. Under last-write-wins a repeated index gives the surviving
        // write's source the whole cotangent and its rivals none, and which
        // write survives is traversal order rather than anything about the
        // values, so there is no derivative to state. Summing has no rivals:
        // every contribution reaches the output, so every contribution takes
        // the output's cotangent unchanged and the target passes its own
        // through untouched.
        gradient = GradientRule::Defined;
    } else if matches!(
        operation,
        OperationKind::TensorToBytes
            | OperationKind::ToHostFloatScalar
            | OperationKind::ToHostFloatVec
            | OperationKind::ToHostIntScalar
            | OperationKind::ToHostIntVec
    ) {
        gradient = GradientRule::None;
    } else if matches!(operation, OperationKind::ToDevice) {
        gradient = GradientRule::Defined;
    }
    if matches!(
        operation,
        OperationKind::SumAll
            | OperationKind::SumDim
            | OperationKind::SumKeepDim
            | OperationKind::ProdAll
            | OperationKind::ProdDim
            | OperationKind::Cumsum
    ) {
        empty = EmptyRule::IdentityOrDefined;
    }
    OperationCatalogEntry {
        operation,
        name,
        family,
        profile,
        descriptor,
        attributes,
        input_arity: min_arity..=max_arity,
        output_arity,
        accepted_ranks,
        broadcasting,
        dtype,
        output,
        same_device: !matches!(
            profile,
            SemanticProfile::Creation | SemanticProfile::Transfer
        ),
        empty,
        numeric,
        gradient,
        deterministic: !matches!(
            operation,
            OperationKind::UniformRandom
                | OperationKind::NormalRandom
                | OperationKind::VariableUniformRandom
                | OperationKind::VariableNormalRandom
                | OperationKind::Sample
                | OperationKind::Dropout
                | OperationKind::Scatter
                | OperationKind::TopK
                | OperationKind::Argsort
        ),
        layout,
        legacy_source,
        capture_eligible: !matches!(
            profile,
            SemanticProfile::Optimizer | SemanticProfile::Transfer
        ) && !matches!(
            operation,
            OperationKind::Sample | OperationKind::Backward
        ),
        site: execution_site(operation),
    }
}

macro_rules! define_catalog {
    ($(($variant:ident, $name:literal, $family:ident, $profile:ident, $attrs:ident, $min:expr, $max:expr, $legacy:literal),)*) => {
        /// Stable operation inventory. Its length is also the uniqueness proof:
        /// enum identities cannot repeat, and the meta-test rejects duplicate names.
        pub static OPERATION_CATALOG: &[OperationCatalogEntry] = &[
            $(entry(
                OperationKind::$variant,
                $name,
                OperationKind::$family,
                SemanticProfile::$profile,
                concat!("Descriptor<op::", stringify!($variant), ">"),
                stringify!($attrs),
                $min,
                $max,
                $legacy,
            ),)*
        ];

        /// Exact marker types used by `Descriptor<O>` and `Execute<O>`.
        pub mod op {
            $(
                #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
                /// Unit struct marker generated per descriptor variant.
                pub struct $variant;
            )*
        }

        $(
            impl Operation for op::$variant {
                type Attributes = $attrs;
                const KEY: OperationKey = OperationKey {
                    namespace: Cow::Borrowed("incin"),
                    name: Cow::Borrowed($name),
                    version: 1,
                };
                fn infer_outputs(
                    attributes: &Self::Attributes,
                    inputs: &[LogicalTensorMeta],
                ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
                    let row = catalog_entry(OperationKind::$variant).ok_or(
                        DescriptorError::MissingCatalogEntry {
                            operation: OperationKind::$variant,
                        },
                    )?;
                    infer_outputs(OperationKind::$variant, row, attributes, inputs)
                }

                fn infer_invocation(
                    attributes: Self::Attributes,
                    inputs: Vec<LogicalTensorMeta>,
                ) -> Result<ValidatedInvocation<Self>, DescriptorError> {
                    ValidatedInvocation::<Self>::infer_runtime(attributes, inputs)
                }

                fn infer_invocation_typed<S: crate::shapes::Shape>(
                    attributes: Self::Attributes,
                    inputs: Vec<LogicalTensorMeta>,
                    expected: &crate::shapes::ShapeValue<S>,
                ) -> Result<ValidatedInvocation<Self>, DescriptorError> {
                    ValidatedInvocation::<Self>::infer_typed(attributes, inputs, expected)
                }
            }

            impl CanonicalOperation for op::$variant {
                const ID: OperationKind = OperationKind::$variant;
            }
        )*
    };
}

// The `incin_operation_catalog!` invocation that expands this template lives
// in descriptor.rs (matching the original file's layout, right after
// `CanonicalOperation`), not here where it's defined - this re-export is
// what makes `define_catalog!` reachable there via `use super::*;`.
pub(crate) use define_catalog;
