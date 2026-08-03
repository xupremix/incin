//! Canonical stable-operation catalog and storage-free typed descriptors.
//!
//! Every executable semantic identity is declared exactly once in
//! `operation_catalog.rs`.  This module turns those rows into the inventory,
//! one marker (and therefore one concrete `Descriptor<Marker>` type) per
//! operation, and an owned representation suitable for tracing/capture.

// Validation is intentionally staged as readable guard clauses. Collapsing
// nested metadata-presence and predicate checks makes the fail-closed branches
// harder to audit and does not change generated code.
#![allow(clippy::collapsible_if, clippy::collapsible_match)]

use alloc::string::ToString;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

use crate::prelude::{DTypeId, DeviceId, OperationKind};

/// Broad classification only. A family is never a capability identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticProfile {
    Creation,
    UnaryFloat,
    BinaryBroadcast,
    Comparison,
    Logical,
    Mutation,
    Shape,
    Selection,
    Indexing,
    MatMul,
    Attention,
    Reduction,
    IndexReduction,
    Scan,
    Normalization,
    Module,
    Composite,
    Loss,
    Quantized,
    Optimizer,
    Transfer,
    Autograd,
}

/// How an operation combines operand extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadcastingRule {
    None,
    Numpy,
    ExplicitTarget,
    TypedContract,
}

/// Dtype legality and promotion policy recorded by the semantic contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeRule {
    Preserve,
    Floating,
    NumericSame,
    IndexResult,
    ExplicitOutput,
    Quantized,
    TypedContract,
}

/// Output metadata inference category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputRule {
    Created,
    Preserve,
    Broadcast,
    ShapeAttributes,
    Reduction,
    MatMul,
    Indexing,
    ExplicitDType,
    TypedInference,
    HostValue,
}

/// Empty tensor behavior. Empty dimensions are never silently rewritten.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmptyRule {
    Allowed,
    IdentityOrDefined,
    RejectedWhenReductionIsEmpty,
    TypedContract,
}

/// Non-finite and arithmetic behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NumericRule {
    NotApplicable,
    IeeePropagate,
    CheckedInteger,
    StableAccumulation,
    TypedContract,
}

/// Gradient contract for an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GradientRule {
    None,
    Defined,
    Piecewise,
    Undefined,
    TypedContract,
}

/// Aliasing/layout contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutRule {
    FreshContiguous,
    ViewWhenPossible,
    PreserveOrMaterialize,
    TypedContract,
    HostOnly,
}

/// One immutable row derived from the authoritative operation declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationCatalogEntry {
    pub operation: OperationKind,
    pub name: &'static str,
    pub family: OperationKind,
    pub profile: SemanticProfile,
    pub descriptor: &'static str,
    pub attributes: &'static str,
    pub input_arity: core::ops::RangeInclusive<usize>,
    pub output_arity: core::ops::RangeInclusive<usize>,
    pub accepted_ranks: core::ops::RangeInclusive<usize>,
    pub broadcasting: BroadcastingRule,
    pub dtype: DTypeRule,
    pub output: OutputRule,
    pub same_device: bool,
    pub empty: EmptyRule,
    pub numeric: NumericRule,
    pub gradient: GradientRule,
    pub deterministic: bool,
    pub layout: LayoutRule,
    pub legacy_source: &'static str,
    pub capture_eligible: bool,
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
            DTypeRule::Preserve,
            OutputRule::Broadcast,
            EmptyRule::Allowed,
            NumericRule::IeeePropagate,
            GradientRule::None,
            LayoutRule::FreshContiguous,
        ),
        Logical => (
            BroadcastingRule::Numpy,
            DTypeRule::Preserve,
            OutputRule::Broadcast,
            EmptyRule::Allowed,
            NumericRule::NotApplicable,
            GradientRule::None,
            LayoutRule::FreshContiguous,
        ),
        Mutation => (
            BroadcastingRule::None,
            DTypeRule::NumericSame,
            OutputRule::Preserve,
            EmptyRule::Allowed,
            NumericRule::IeeePropagate,
            GradientRule::Defined,
            LayoutRule::PreserveOrMaterialize,
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

const fn entry(
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
        OperationKind::MatMulExact | OperationKind::QuantizedMatMul => 2..=crate::prelude::MAX_RANK,
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
        | OperationKind::GroupNorm
        | OperationKind::InstanceNorm
        | OperationKind::LayerNorm
        | OperationKind::BatchNorm
        | OperationKind::RmsNorm => 1..=crate::prelude::MAX_RANK,
        _ => 0..=crate::prelude::MAX_RANK,
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

        /// Exact marker types used by `Descriptor<O>` and `Execute<Descriptor<O>>`.
        pub mod op {
            $(
                #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
                pub struct $variant;
            )*
        }

        $(
            impl CanonicalOperation for op::$variant {
                type Attributes = $attrs;
                const ID: OperationKind = OperationKind::$variant;
            }
        )*
    };
}

/// Logical metadata used before a backend storage handle exists.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LogicalTensorMeta {
    pub shape: Option<Vec<usize>>,
    pub dtype: Option<DTypeId>,
    pub device: Option<DeviceId>,
}

impl LogicalTensorMeta {
    #[must_use]
    pub const fn unknown() -> Self {
        Self {
            shape: None,
            dtype: None,
            device: None,
        }
    }
}

macro_rules! attributes {
    ($($name:ident { $($field:ident: $ty:ty),* $(,)? })*) => {
        $(
            #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
            pub struct $name { $(pub $field: $ty,)* }
        )*
    };
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NoAttributes;

attributes! {
    CreationAttributes { shape: Vec<usize>, dtype: DTypeId, device: DeviceId }
    FullAttributes { shape: Vec<usize>, dtype: DTypeId, device: DeviceId, value: f64 }
    ArangeAttributes { shape: Vec<usize>, dtype: DTypeId, device: DeviceId, start: f64, step: f64 }
    LinspaceAttributes { shape: Vec<usize>, dtype: DTypeId, device: DeviceId, start: f64, end: f64 }
    DistributionAttributes { shape: Vec<usize>, dtype: DTypeId, device: DeviceId, distribution: alloc::string::String, parameters: Vec<u8> }
    AxisAttributes { axis: usize }
    ScalarAttributes { value: f64 }
    ClampAttributes { min: f64, max: f64 }
    LerpAttributes { weight: f64 }
    ShapeAttributes { shape: Vec<usize> }
    RepeatAttributes { repeats: Vec<usize> }
    TransposeAttributes { first: usize, second: usize }
    NarrowAttributes { axis: usize, start: usize, length: usize }
    SliceAttributes { ranges: Vec<(usize, usize)> }
    FlattenAttributes { start_axis: usize, end_axis: usize }
    ScatterAttributes { axis: usize, duplicate_indices: DuplicateIndexRule }
    PadAttributes { padding: Vec<(usize, usize)>, value: f64 }
    DiagonalAttributes { offset: i64 }
    ChunkAttributes { chunks: usize, axis: usize }
    SplitAttributes { split_size: usize, axis: usize }
    AddmmAttributes { alpha: f64, beta: f64 }
    AttentionAttributes { scale: Option<f64>, has_mask: bool }
    UnfoldAttributes { axis: usize, size: usize, step: usize }
    PixelShuffleAttributes { upscale_factor: usize }
    GroupNormAttributes { groups: usize, epsilon: f64 }
    EpsilonAttributes { epsilon: f64 }
    DTypeAttributes { dtype: DTypeId }
    DeviceAttributes { device: DeviceId }
    IndexReductionAttributes { axis: Option<usize>, dtype: DTypeId }
    TopKAttributes { k: usize, axis: usize, largest: bool, index_dtype: DTypeId }
    ArgsortAttributes { axis: usize, descending: bool, index_dtype: DTypeId }
    NormAttributes { order: f64 }
    VarianceAttributes { unbiased: bool }
    AxisVarianceAttributes { axis: usize, unbiased: bool }
    LayerNormAttributes { normalized_shape: Vec<usize>, epsilon: f64, has_bias: bool }
    BatchNormAttributes { epsilon: f64, momentum: f64, training: bool, has_weight: bool, has_bias: bool, has_running_mean: bool, has_running_variance: bool }
    Conv1dAttributes { stride: usize, padding: usize, dilation: usize, groups: usize, has_bias: bool }
    Conv2dAttributes { stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], groups: usize, has_bias: bool }
    ConvTranspose2dAttributes { stride: [usize; 2], padding: [usize; 2], output_padding: [usize; 2], dilation: [usize; 2], groups: usize, has_bias: bool }
    Pool2dAttributes { kernel: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2] }
    AvgPool2dAttributes { kernel: [usize; 2], stride: [usize; 2], padding: [usize; 2] }
    AdaptivePool2dAttributes { output: [usize; 2] }
    LinearAttributes { has_bias: bool }
    DropoutAttributes { probability: f64, training: bool }
    RecurrentAttributes { input_size: usize, hidden_size: usize, bias_ih: bool, bias_hh: bool }
    LossAttributes { reduction: LossReduction }
    QuantizationAttributes { dtype: DTypeId }
    SgdAttributes { learning_rate: f64 }
    AdamAttributes { learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, step: usize }
    AdamWAttributes { learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64, step: usize }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DuplicateIndexRule {
    LastWriteWins,
    Reject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LossReduction {
    None,
    Mean,
    Sum,
}

mod private {
    pub trait Sealed {}
}

/// A catalog operation with its exact typed attribute set.
pub trait CanonicalOperation: private::Sealed + Clone + fmt::Debug + 'static {
    type Attributes: AttributeContract
        + Clone
        + fmt::Debug
        + PartialEq
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>;
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
pub struct Descriptor<O: CanonicalOperation> {
    attributes: O::Attributes,
    outputs: Vec<LogicalTensorMeta>,
    marker: PhantomData<fn() -> O>,
}

impl<O: CanonicalOperation> crate::exec::spec::ExecutionDescriptor for Descriptor<O> {}

impl<O: CanonicalOperation> Descriptor<O> {
    #[must_use]
    pub const fn operation(&self) -> OperationKind {
        O::ID
    }
    #[must_use]
    pub const fn attributes(&self) -> &O::Attributes {
        &self.attributes
    }
    #[must_use]
    pub fn outputs(&self) -> &[LogicalTensorMeta] {
        &self.outputs
    }
}

/// Storage-free serialized descriptor capture. The exact identity is outside
/// the payload so decoding as the wrong descriptor type fails closed.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CapturedDescriptor {
    operation: OperationKind,
    schema: u32,
    payload: Vec<u8>,
}

#[cfg(feature = "std")]
#[derive(Debug)]
pub enum DescriptorCaptureError {
    Identity {
        expected: OperationKind,
        actual: OperationKind,
    },
    Schema {
        expected: u32,
        actual: u32,
    },
    Encode(postcard::Error),
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
    pub fn capture<O: CanonicalOperation>(
        descriptor: &Descriptor<O>,
    ) -> Result<Self, DescriptorCaptureError> {
        let payload = postcard::to_allocvec(descriptor).map_err(DescriptorCaptureError::Encode)?;
        Ok(Self {
            operation: O::ID,
            schema: crate::exec::DescriptorSchemaVersion::CURRENT.get(),
            payload,
        })
    }

    #[must_use]
    pub const fn operation(&self) -> OperationKind {
        self.operation
    }

    pub fn decode<O: CanonicalOperation>(&self) -> Result<Descriptor<O>, DescriptorCaptureError> {
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
        postcard::from_bytes(&self.payload).map_err(DescriptorCaptureError::Decode)
    }
}

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
    Shape(crate::prelude::ShapeError),
    MetadataMismatch {
        operation: OperationKind,
        output: usize,
        field: &'static str,
    },
}

impl From<crate::prelude::ShapeError> for DescriptorError {
    fn from(error: crate::prelude::ShapeError) -> Self {
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
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DescriptorError {}

/// Validation implemented by each concrete typed attribute schema.
pub trait AttributeContract {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError>;
    fn declared_shape(&self) -> Option<&[usize]> {
        None
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
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
}

/// Borrowed shape attributes used by the common, storage-free inference path.
/// This is deliberately not public API: callers provide typed attributes and
/// cannot choose a transform independently of the descriptor type.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub enum ShapeTransform<'a> {
    Axis(usize),
    Transpose(usize, usize),
    Narrow {
        axis: usize,
        length: usize,
    },
    Slice(&'a [(usize, usize)]),
    Flatten {
        start: usize,
        end: usize,
    },
    Repeat(&'a [usize]),
    Pad(&'a [(usize, usize)]),
    Diagonal(i64),
    Unfold {
        axis: usize,
        size: usize,
        step: usize,
    },
    PixelShuffle(usize),
    AdaptivePool2d([usize; 2]),
    TopK {
        axis: usize,
        k: usize,
    },
    Chunk {
        chunks: usize,
        axis: usize,
    },
    Split {
        split_size: usize,
        axis: usize,
    },
    Conv1d(&'a Conv1dAttributes),
    Conv2d(&'a Conv2dAttributes),
    ConvTranspose2d(&'a ConvTranspose2dAttributes),
    Pool2d(&'a Pool2dAttributes),
    AvgPool2d(&'a AvgPool2dAttributes),
    Rnn(&'a RecurrentAttributes),
}

fn invalid(
    operation: OperationKind,
    attribute: &'static str,
    reason: &'static str,
) -> DescriptorError {
    DescriptorError::InvalidAttribute {
        operation,
        attribute,
        reason,
    }
}

fn first_shape(inputs: &[LogicalTensorMeta]) -> Option<&[usize]> {
    inputs.first()?.shape.as_deref()
}

/// An index-producing operation must declare an integer output dtype.
///
/// `argmax`, `argmin`, `topk`, and `argsort` carry their index dtype as a typed
/// attribute so the descriptor can infer the output exactly. Nothing else
/// constrained that field, so a caller could declare `F32` and have the
/// descriptor certify a floating-point "index" tensor. The family default
/// (`DTypeRule::IndexResult`) states the intent; this enforces it.
fn validate_index_dtype(
    operation: OperationKind,
    attribute: &'static str,
    dtype: DTypeId,
) -> Result<(), DescriptorError> {
    if matches!(dtype, DTypeId::U8 | DTypeId::U32 | DTypeId::I64) {
        return Ok(());
    }
    Err(invalid(
        operation,
        attribute,
        "an index output requires an integer dtype",
    ))
}

/// Reject an unbiased (Bessel-corrected) estimate over fewer than two elements.
///
/// The correction divides by `n - 1`. The `Reduction` family default only
/// rejects an *empty* domain, which is not enough here: a single-element domain
/// is non-empty and still degenerate. Refusing it in the descriptor keeps the
/// division out of every backend.
fn validate_unbiased_domain(
    operation: OperationKind,
    unbiased: bool,
    extent: Option<usize>,
) -> Result<(), DescriptorError> {
    if !unbiased {
        return Ok(());
    }
    match extent {
        Some(count) if count < 2 => Err(invalid(
            operation,
            "unbiased",
            "an unbiased variance or standard deviation requires at least two elements",
        )),
        _ => Ok(()),
    }
}

fn validate_shape(operation: OperationKind, shape: &[usize]) -> Result<(), DescriptorError> {
    crate::prelude::ShapeBuf::from_slice(shape).checked_numel(operation)?;
    Ok(())
}

macro_rules! unconstrained_attributes {
    ($($ty:ty),* $(,)?) => {$(
        impl AttributeContract for $ty {
            fn validate(&self, _operation: OperationKind, _inputs: &[LogicalTensorMeta]) -> Result<(), DescriptorError> { Ok(()) }
        }
    )*};
}

unconstrained_attributes!(NoAttributes, ScalarAttributes, LerpAttributes,);

impl AttributeContract for CreationAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for FullAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for ArangeAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)?;
        if !self.start.is_finite() || !self.step.is_finite() || self.step == 0.0 {
            return Err(invalid(
                operation,
                "step",
                "arange requires finite start and non-zero finite step",
            ));
        }
        Ok(())
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for LinspaceAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)?;
        if !self.start.is_finite() || !self.end.is_finite() {
            return Err(invalid(
                operation,
                "bounds",
                "linspace bounds must be finite",
            ));
        }
        Ok(())
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for DistributionAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)?;
        if self.distribution.is_empty() {
            return Err(invalid(
                operation,
                "distribution",
                "distribution identity must not be empty",
            ));
        }
        Ok(())
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
        Some(self.dtype)
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for ClampAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.min.is_nan() || self.max.is_nan() || self.min > self.max {
            return Err(invalid(
                operation,
                "min/max",
                "clamp requires ordered non-NaN bounds",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for ShapeAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.shape)?;
        if let Some(input) = first_shape(inputs) {
            if operation == OperationKind::ReshapeExact {
                let source =
                    crate::prelude::ShapeBuf::from_slice(input).checked_numel(operation)?;
                let target =
                    crate::prelude::ShapeBuf::from_slice(&self.shape).checked_numel(operation)?;
                if source != target {
                    return Err(invalid(
                        operation,
                        "shape",
                        "reshape must preserve the element count",
                    ));
                }
            } else if matches!(
                operation,
                OperationKind::BroadcastAs | OperationKind::BroadcastLeft
            ) {
                if input.len() > self.shape.len()
                    || input
                        .iter()
                        .rev()
                        .zip(self.shape.iter().rev())
                        .any(|(&source, &target)| source != target && source != 1)
                {
                    return Err(invalid(
                        operation,
                        "shape",
                        "source shape cannot broadcast to the explicit target",
                    ));
                }
            }
        }
        Ok(())
    }
    fn declared_shape(&self) -> Option<&[usize]> {
        Some(&self.shape)
    }
}
impl AttributeContract for RepeatAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            if shape.len() != self.repeats.len() {
                return Err(invalid(
                    operation,
                    "repeats",
                    "repeat count rank must equal input rank",
                ));
            }
            let mut output = Vec::with_capacity(shape.len());
            for (&dim, &repeat) in shape.iter().zip(&self.repeats) {
                output.push(dim.checked_mul(repeat).ok_or_else(|| {
                    invalid(
                        operation,
                        "repeats",
                        "repeated output dimension overflows usize",
                    )
                })?);
            }
            validate_shape(operation, &output)?;
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Repeat(&self.repeats))
    }
}
impl AttributeContract for AxisAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            let insertion = matches!(
                operation,
                OperationKind::UnsqueezeExact | OperationKind::StackExact
            );
            let limit = shape.len() + usize::from(insertion);
            if self.axis >= limit {
                return Err(invalid(
                    operation,
                    "axis",
                    "axis is outside the accepted rank",
                ));
            }
        }
        Ok(())
    }
    fn axis(&self) -> Option<usize> {
        Some(self.axis)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Axis(self.axis))
    }
}
impl AttributeContract for IndexReductionAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_index_dtype(operation, "dtype", self.dtype)?;
        if let (Some(axis), Some(shape)) = (self.axis, first_shape(inputs)) {
            if axis >= shape.len() {
                return Err(invalid(operation, "axis", "axis is outside the input rank"));
            }
        }
        Ok(())
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
        Some(self.dtype)
    }
    fn axis(&self) -> Option<usize> {
        self.axis
    }
}
impl AttributeContract for TransposeAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            if self.first >= shape.len() || self.second >= shape.len() {
                return Err(invalid(
                    operation,
                    "axis",
                    "transpose axis is outside the input rank",
                ));
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Transpose(self.first, self.second))
    }
}
impl AttributeContract for NarrowAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            let Some(&extent) = shape.get(self.axis) else {
                return Err(invalid(
                    operation,
                    "axis",
                    "narrow axis is outside the input rank",
                ));
            };
            let end = self.start.checked_add(self.length).ok_or_else(|| {
                invalid(operation, "start/length", "narrow endpoint overflows usize")
            })?;
            if end > extent {
                return Err(invalid(
                    operation,
                    "start/length",
                    "narrow range exceeds the input extent",
                ));
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Narrow {
            axis: self.axis,
            length: self.length,
        })
    }
}
impl AttributeContract for SliceAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            if shape.len() != self.ranges.len() {
                return Err(invalid(
                    operation,
                    "ranges",
                    "slice range count must equal input rank",
                ));
            }
            for ((start, end), extent) in self.ranges.iter().zip(shape) {
                if start > end || end > extent {
                    return Err(invalid(
                        operation,
                        "ranges",
                        "slice range must be ordered and within its extent",
                    ));
                }
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Slice(&self.ranges))
    }
}
impl AttributeContract for FlattenAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.start_axis > self.end_axis {
            return Err(invalid(
                operation,
                "axis range",
                "flatten start must not exceed end",
            ));
        }
        if let Some(shape) = first_shape(inputs) {
            if self.end_axis >= shape.len() {
                return Err(invalid(
                    operation,
                    "axis range",
                    "flatten end is outside the input rank",
                ));
            }
            validate_shape(operation, shape)?;
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Flatten {
            start: self.start_axis,
            end: self.end_axis,
        })
    }
}
impl AttributeContract for ScatterAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        AxisAttributes { axis: self.axis }.validate(operation, inputs)
    }
    fn axis(&self) -> Option<usize> {
        Some(self.axis)
    }
}
impl AttributeContract for PadAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            if shape.len() != self.padding.len() {
                return Err(invalid(
                    operation,
                    "padding",
                    "padding rank must equal input rank",
                ));
            }
            let mut output = Vec::with_capacity(shape.len());
            for (&dim, &(before, after)) in shape.iter().zip(&self.padding) {
                output.push(
                    dim.checked_add(before)
                        .and_then(|v| v.checked_add(after))
                        .ok_or_else(|| {
                            invalid(operation, "padding", "padded dimension overflows usize")
                        })?,
                );
            }
            validate_shape(operation, &output)?;
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Pad(&self.padding))
    }
}
impl AttributeContract for DiagonalAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if let Some(shape) = first_shape(inputs) {
            if !(1..=2).contains(&shape.len()) {
                return Err(invalid(
                    operation,
                    "rank",
                    "diagonal operations require rank one or two",
                ));
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Diagonal(self.offset))
    }
}
impl AttributeContract for ChunkAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.chunks == 0 {
            return Err(invalid(operation, "chunks", "chunk count must be non-zero"));
        }
        AxisAttributes { axis: self.axis }.validate(operation, inputs)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Chunk {
            chunks: self.chunks,
            axis: self.axis,
        })
    }
    fn expected_output_count(&self, inputs: &[LogicalTensorMeta]) -> Option<usize> {
        let extent = first_shape(inputs)?.get(self.axis).copied()?;
        Some(self.chunks.min(extent))
    }
}
impl AttributeContract for SplitAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.split_size == 0 {
            return Err(invalid(
                operation,
                "split_size",
                "split size must be non-zero",
            ));
        }
        AxisAttributes { axis: self.axis }.validate(operation, inputs)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Split {
            split_size: self.split_size,
            axis: self.axis,
        })
    }
    fn expected_output_count(&self, inputs: &[LogicalTensorMeta]) -> Option<usize> {
        let extent = first_shape(inputs)?.get(self.axis).copied()?;
        Some(extent.div_ceil(self.split_size))
    }
}
impl AttributeContract for AddmmAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if !self.alpha.is_finite() || !self.beta.is_finite() {
            return Err(invalid(
                operation,
                "alpha/beta",
                "addmm scaling factors must be finite",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for AttentionAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self
            .scale
            .is_some_and(|scale| !scale.is_finite() || scale <= 0.0)
        {
            return Err(invalid(
                operation,
                "scale",
                "attention scale must be positive and finite",
            ));
        }
        let expected = if self.has_mask { 4 } else { 3 };
        if inputs.len() != expected {
            return Err(invalid(
                operation,
                "has_mask",
                "attention input arity disagrees with the mask attribute",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for UnfoldAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.size == 0 || self.step == 0 {
            return Err(invalid(
                operation,
                "size/step",
                "unfold size and step must be non-zero",
            ));
        }
        AxisAttributes { axis: self.axis }.validate(operation, inputs)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Unfold {
            axis: self.axis,
            size: self.size,
            step: self.step,
        })
    }
}
impl AttributeContract for PixelShuffleAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.upscale_factor == 0 {
            return Err(invalid(
                operation,
                "upscale_factor",
                "pixel shuffle factor must be non-zero",
            ));
        }
        if let Some(shape) = first_shape(inputs) {
            if shape.len() != 4 {
                return Err(invalid(
                    operation,
                    "rank",
                    "pixel shuffle requires rank four",
                ));
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::PixelShuffle(self.upscale_factor))
    }
}
impl AttributeContract for GroupNormAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.groups == 0 {
            return Err(invalid(operation, "groups", "group count must be non-zero"));
        }
        validate_epsilon(operation, self.epsilon)?;
        if let Some(shape) = first_shape(inputs) {
            if shape.len() < 2 || shape[1] % self.groups != 0 {
                return Err(invalid(
                    operation,
                    "groups",
                    "group norm requires a channel axis divisible by the group count",
                ));
            }
        }
        Ok(())
    }
}
fn validate_epsilon(operation: OperationKind, epsilon: f64) -> Result<(), DescriptorError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(invalid(
            operation,
            "epsilon",
            "epsilon must be finite and non-negative",
        ));
    }
    Ok(())
}
impl AttributeContract for EpsilonAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_epsilon(operation, self.epsilon)?;
        match operation {
            OperationKind::InstanceNorm => {
                if first_shape(inputs).is_some_and(|shape| shape.len() != 4) {
                    return Err(invalid(
                        operation,
                        "rank",
                        "instance norm requires [batch, channels, height, width]",
                    ));
                }
            }
            OperationKind::RmsNorm => {
                if inputs.len() != 2 {
                    return Err(DescriptorError::Arity {
                        operation,
                        expected: 2..=2,
                        actual: inputs.len(),
                    });
                }
                if let (Some(input), Some(weight)) = (
                    first_shape(inputs),
                    inputs.get(1).and_then(|value| value.shape.as_deref()),
                ) {
                    if input.last() != weight.last() || weight.len() != 1 {
                        return Err(invalid(
                            operation,
                            "weight shape",
                            "RMS norm weight must match the final input extent",
                        ));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}
impl AttributeContract for LayerNormAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_shape(operation, &self.normalized_shape)?;
        validate_epsilon(operation, self.epsilon)?;
        let expected = if self.has_bias { 3 } else { 2 };
        if inputs.len() != expected {
            return Err(invalid(
                operation,
                "has_bias",
                "layer norm input arity disagrees with the bias attribute",
            ));
        }
        if let Some(input) = first_shape(inputs) {
            if self.normalized_shape.len() > input.len()
                || input[input.len() - self.normalized_shape.len()..] != self.normalized_shape
                || inputs.get(1).and_then(|value| value.shape.as_deref())
                    != Some(self.normalized_shape.as_slice())
                || self.has_bias
                    && inputs.get(2).and_then(|value| value.shape.as_deref())
                        != Some(self.normalized_shape.as_slice())
            {
                return Err(invalid(
                    operation,
                    "normalized shape",
                    "layer norm input suffix, weight, and optional bias must match",
                ));
            }
        }
        Ok(())
    }
}
impl AttributeContract for BatchNormAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_epsilon(operation, self.epsilon)?;
        if !self.momentum.is_finite() || !(0.0..=1.0).contains(&self.momentum) {
            return Err(invalid(
                operation,
                "momentum",
                "batch norm momentum must be in [0, 1]",
            ));
        }
        let expected = 1
            + usize::from(self.has_weight)
            + usize::from(self.has_bias)
            + usize::from(self.has_running_mean)
            + usize::from(self.has_running_variance);
        if inputs.len() != expected
            || self.has_running_mean != self.has_running_variance
            || !self.training && !self.has_running_mean
        {
            return Err(invalid(
                operation,
                "optional state",
                "batch norm attributes and input arity/state are inconsistent",
            ));
        }
        if let Some(input) = first_shape(inputs) {
            if input.len() < 2 {
                return Err(invalid(
                    operation,
                    "input shape",
                    "batch norm requires a channel axis",
                ));
            }
            let channels = input[1];
            for parameter in inputs.iter().skip(1) {
                if parameter.shape.as_deref() != Some(&[channels]) {
                    return Err(invalid(
                        operation,
                        "parameter shape",
                        "batch norm affine/state tensors must match the channel extent",
                    ));
                }
            }
        }
        Ok(())
    }
}

macro_rules! spatial_contract {
    ($ty:ty, $body:expr, $transform:ident, bias) => {
        impl AttributeContract for $ty {
            fn validate(
                &self,
                operation: OperationKind,
                _: &[LogicalTensorMeta],
            ) -> Result<(), DescriptorError> {
                if $body(self) {
                    Ok(())
                } else {
                    Err(invalid(
                        operation,
                        "spatial parameters",
                        "kernel, stride, dilation, and groups where present must be non-zero",
                    ))
                }
            }
            fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
                Some(ShapeTransform::$transform(self))
            }
            fn optional_bias(&self) -> Option<bool> {
                Some(self.has_bias)
            }
        }
    };
    ($ty:ty, $body:expr, $transform:ident) => {
        impl AttributeContract for $ty {
            fn validate(
                &self,
                operation: OperationKind,
                _: &[LogicalTensorMeta],
            ) -> Result<(), DescriptorError> {
                if $body(self) {
                    Ok(())
                } else {
                    Err(invalid(
                        operation,
                        "spatial parameters",
                        "kernel, stride, dilation, and groups where present must be non-zero",
                    ))
                }
            }
            fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
                Some(ShapeTransform::$transform(self))
            }
        }
    };
}
spatial_contract!(
    Conv1dAttributes,
    |a: &Conv1dAttributes| a.stride > 0 && a.dilation > 0 && a.groups > 0,
    Conv1d,
    bias
);
spatial_contract!(
    Conv2dAttributes,
    |a: &Conv2dAttributes| a.stride.iter().all(|&v| v > 0)
        && a.dilation.iter().all(|&v| v > 0)
        && a.groups > 0,
    Conv2d,
    bias
);
spatial_contract!(
    ConvTranspose2dAttributes,
    |a: &ConvTranspose2dAttributes| a.stride.iter().all(|&v| v > 0)
        && a.dilation.iter().all(|&v| v > 0)
        && a.groups > 0,
    ConvTranspose2d,
    bias
);
spatial_contract!(
    Pool2dAttributes,
    |a: &Pool2dAttributes| a.kernel.iter().all(|&v| v > 0)
        && a.stride.iter().all(|&v| v > 0)
        && a.dilation.iter().all(|&v| v > 0),
    Pool2d
);
spatial_contract!(
    AvgPool2dAttributes,
    |a: &AvgPool2dAttributes| a.kernel.iter().all(|&v| v > 0) && a.stride.iter().all(|&v| v > 0),
    AvgPool2d
);
impl AttributeContract for AdaptivePool2dAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.output.iter().all(|&extent| extent > 0) {
            Ok(())
        } else {
            Err(invalid(
                operation,
                "output",
                "adaptive pooling output extents must be non-zero",
            ))
        }
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::AdaptivePool2d(self.output))
    }
}

impl AttributeContract for TopKAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        AxisAttributes { axis: self.axis }.validate(operation, inputs)?;
        validate_index_dtype(operation, "index_dtype", self.index_dtype)?;
        if self.k == 0 {
            return Err(invalid(
                operation,
                "k",
                "top-k requires k greater than zero",
            ));
        }
        if let Some(shape) = first_shape(inputs) {
            if self.k > shape[self.axis] {
                return Err(invalid(operation, "k", "top-k exceeds the selected extent"));
            }
        }
        Ok(())
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
        Some(self.index_dtype)
    }
    fn axis(&self) -> Option<usize> {
        Some(self.axis)
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::TopK {
            axis: self.axis,
            k: self.k,
        })
    }
}
impl AttributeContract for ArgsortAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        AxisAttributes { axis: self.axis }.validate(operation, inputs)?;
        validate_index_dtype(operation, "index_dtype", self.index_dtype)
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
        Some(self.index_dtype)
    }
    fn axis(&self) -> Option<usize> {
        Some(self.axis)
    }
}
impl AttributeContract for NormAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if !self.order.is_finite() || self.order <= 0.0 {
            return Err(invalid(
                operation,
                "order",
                "norm order must be positive and finite",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for VarianceAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        // `var_all`/`std_all` reduce the whole tensor, so the corrected
        // denominator is the element count.
        validate_unbiased_domain(
            operation,
            self.unbiased,
            first_shape(inputs).map(|shape| shape.iter().product()),
        )
    }
}
impl AttributeContract for AxisVarianceAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        AxisAttributes { axis: self.axis }.validate(operation, inputs)?;
        validate_unbiased_domain(
            operation,
            self.unbiased,
            first_shape(inputs).and_then(|shape| shape.get(self.axis).copied()),
        )
    }
}
impl AttributeContract for DropoutAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if !self.probability.is_finite() || !(0.0..1.0).contains(&self.probability) {
            return Err(invalid(
                operation,
                "probability",
                "dropout probability must be in [0, 1)",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for LinearAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        let expected = if self.has_bias { 3 } else { 2 };
        if inputs.len() != expected {
            return Err(invalid(
                operation,
                "has_bias",
                "linear input arity disagrees with the bias attribute",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for RecurrentAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        if self.input_size == 0 || self.hidden_size == 0 {
            return Err(invalid(
                operation,
                "feature size",
                "recurrent input and hidden sizes must be non-zero",
            ));
        }
        if let Some(sequence) = first_shape(inputs) {
            if sequence.len() != 3 || sequence[2] != self.input_size {
                return Err(invalid(
                    operation,
                    "input shape",
                    "recurrent sequence must be [batch, sequence, input_size]",
                ));
            }
            for state in inputs.iter().skip(1) {
                if state.shape.as_deref() != Some(&[sequence[0], self.hidden_size]) {
                    return Err(invalid(
                        operation,
                        "state shape",
                        "recurrent states must be [batch, hidden_size]",
                    ));
                }
            }
        }
        Ok(())
    }
    fn shape_transform(&self) -> Option<ShapeTransform<'_>> {
        Some(ShapeTransform::Rnn(self))
    }
}
impl AttributeContract for SgdAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_learning_rate(operation, self.learning_rate)
    }
}
fn validate_learning_rate(operation: OperationKind, rate: f64) -> Result<(), DescriptorError> {
    if !rate.is_finite() || rate < 0.0 {
        return Err(invalid(
            operation,
            "learning_rate",
            "learning rate must be finite and non-negative",
        ));
    }
    Ok(())
}
impl AttributeContract for AdamAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        _: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        validate_learning_rate(operation, self.learning_rate)?;
        validate_epsilon(operation, self.epsilon)?;
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) || self.step == 0
        {
            return Err(invalid(
                operation,
                "beta/step",
                "Adam betas must be in [0, 1) and step must be non-zero",
            ));
        }
        Ok(())
    }
}
impl AttributeContract for AdamWAttributes {
    fn validate(
        &self,
        operation: OperationKind,
        inputs: &[LogicalTensorMeta],
    ) -> Result<(), DescriptorError> {
        AdamAttributes {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            step: self.step,
        }
        .validate(operation, inputs)?;
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(invalid(
                operation,
                "weight_decay",
                "weight decay must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

impl AttributeContract for DTypeAttributes {
    fn validate(&self, _: OperationKind, _: &[LogicalTensorMeta]) -> Result<(), DescriptorError> {
        Ok(())
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
        Some(self.dtype)
    }
}
impl AttributeContract for DeviceAttributes {
    fn validate(&self, _: OperationKind, _: &[LogicalTensorMeta]) -> Result<(), DescriptorError> {
        Ok(())
    }
    fn declared_device(&self) -> Option<DeviceId> {
        Some(self.device)
    }
}
impl AttributeContract for QuantizationAttributes {
    fn validate(&self, _: OperationKind, _: &[LogicalTensorMeta]) -> Result<(), DescriptorError> {
        Ok(())
    }
    fn declared_dtype(&self) -> Option<DTypeId> {
        Some(self.dtype)
    }
}
impl AttributeContract for LossAttributes {
    fn validate(&self, _: OperationKind, _: &[LogicalTensorMeta]) -> Result<(), DescriptorError> {
        Ok(())
    }
    fn loss_reduction(&self) -> Option<LossReduction> {
        Some(self.reduction)
    }
}

fn broadcast_shape(
    operation: OperationKind,
    shapes: &[&[usize]],
) -> Result<Vec<usize>, DescriptorError> {
    let rank = shapes.iter().map(|shape| shape.len()).max().unwrap_or(0);
    let mut output = vec![1; rank];
    for shape in shapes {
        for (offset, &dim) in shape.iter().rev().enumerate() {
            let axis = rank - 1 - offset;
            let current = output[axis];
            output[axis] = match (current, dim) {
                (a, b) if a == b => a,
                (1, b) => b,
                (a, 1) => a,
                _ => {
                    return Err(invalid(
                        operation,
                        "shape",
                        "input shapes are not broadcast-compatible",
                    ));
                }
            };
        }
    }
    validate_shape(operation, &output)?;
    Ok(output)
}

fn transformed_shape<A: AttributeContract>(
    operation: OperationKind,
    attributes: &A,
    inputs: &[LogicalTensorMeta],
    output_index: usize,
) -> Result<Option<Option<Vec<usize>>>, DescriptorError> {
    let Some(transform) = attributes.shape_transform() else {
        return Ok(None);
    };
    let Some(input) = inputs.first().and_then(|input| input.shape.as_deref()) else {
        return Ok(Some(None));
    };
    fn spatial_output(
        operation: OperationKind,
        input: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<usize, DescriptorError> {
        let effective = kernel
            .checked_sub(1)
            .and_then(|value| value.checked_mul(dilation))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid(operation, "spatial", "effective kernel overflows usize"))?;
        let padded = input
            .checked_add(padding)
            .and_then(|value| value.checked_add(padding))
            .ok_or_else(|| invalid(operation, "spatial", "padded extent overflows usize"))?;
        if effective > padded {
            return Err(invalid(
                operation,
                "spatial",
                "effective kernel exceeds padded input extent",
            ));
        }
        Ok((padded - effective) / stride + 1)
    }

    fn convolution_output(
        operation: OperationKind,
        input: &[usize],
        weight: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
    ) -> Result<Vec<usize>, DescriptorError> {
        let dimensions = stride.len();
        if input.len() < dimensions + 1 || weight.len() != dimensions + 2 {
            return Err(invalid(
                operation,
                "rank",
                "convolution operand rank is invalid",
            ));
        }
        let mut output = input.to_vec();
        output[input.len() - dimensions - 1] = weight[0];
        for axis in 0..dimensions {
            output[input.len() - dimensions + axis] = spatial_output(
                operation,
                input[input.len() - dimensions + axis],
                weight[weight.len() - dimensions + axis],
                stride[axis],
                padding[axis],
                dilation[axis],
            )?;
        }
        Ok(output)
    }

    let output = match transform {
        ShapeTransform::Axis(axis) => match operation {
            OperationKind::SqueezeExact => {
                if input[axis] != 1 {
                    return Err(invalid(operation, "axis", "squeeze extent must equal one"));
                }
                let mut output = input.to_vec();
                output.remove(axis);
                output
            }
            OperationKind::UnsqueezeExact => {
                let mut output = input.to_vec();
                output.insert(axis, 1);
                output
            }
            OperationKind::StackExact => {
                let mut output = input.to_vec();
                output.insert(axis, inputs.len());
                output
            }
            OperationKind::ConcatExact => {
                let mut output = input.to_vec();
                let mut extent = 0usize;
                for other in inputs {
                    let Some(shape) = other.shape.as_deref() else {
                        return Ok(Some(None));
                    };
                    if shape.len() != input.len()
                        || shape
                            .iter()
                            .enumerate()
                            .any(|(index, value)| index != axis && *value != input[index])
                    {
                        return Err(invalid(
                            operation,
                            "shape",
                            "concat inputs must match outside the concat axis",
                        ));
                    }
                    extent = extent.checked_add(shape[axis]).ok_or_else(|| {
                        invalid(operation, "shape", "concat extent overflows usize")
                    })?;
                }
                output[axis] = extent;
                output
            }
            _ => return Ok(None),
        },
        ShapeTransform::Transpose(first, second) => {
            let mut output = input.to_vec();
            output.swap(first, second);
            output
        }
        ShapeTransform::Narrow { axis, length } => {
            let mut output = input.to_vec();
            output[axis] = length;
            output
        }
        ShapeTransform::Slice(ranges) => ranges.iter().map(|(start, end)| end - start).collect(),
        ShapeTransform::Flatten { start, end } => {
            let flattened = input[start..=end]
                .iter()
                .try_fold(1usize, |value, &extent| {
                    value.checked_mul(extent).ok_or_else(|| {
                        invalid(operation, "shape", "flattened extent overflows usize")
                    })
                })?;
            let mut output = Vec::with_capacity(input.len() - (end - start));
            output.extend_from_slice(&input[..start]);
            output.push(flattened);
            output.extend_from_slice(&input[end + 1..]);
            output
        }
        ShapeTransform::Repeat(repeats) => input
            .iter()
            .zip(repeats)
            .map(|(&extent, &repeat)| {
                extent
                    .checked_mul(repeat)
                    .ok_or_else(|| invalid(operation, "repeats", "repeated extent overflows usize"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        ShapeTransform::Pad(padding) => input
            .iter()
            .zip(padding)
            .map(|(&extent, &(before, after))| {
                extent
                    .checked_add(before)
                    .and_then(|value| value.checked_add(after))
                    .ok_or_else(|| invalid(operation, "padding", "padded extent overflows usize"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        ShapeTransform::Diagonal(offset) if operation == OperationKind::Diag => {
            let displacement = usize::try_from(offset.unsigned_abs())
                .map_err(|_| invalid(operation, "offset", "diagonal offset does not fit usize"))?;
            if input.len() == 1 {
                let extent = input[0].checked_add(displacement).ok_or_else(|| {
                    invalid(operation, "offset", "diagonal extent overflows usize")
                })?;
                vec![extent, extent]
            } else {
                let rows = input[input.len() - 2];
                let columns = input[input.len() - 1];
                let extent = if offset >= 0 {
                    rows.min(columns.saturating_sub(displacement))
                } else {
                    rows.saturating_sub(displacement).min(columns)
                };
                vec![extent]
            }
        }
        ShapeTransform::Diagonal(_) => input.to_vec(),
        ShapeTransform::Unfold { axis, size, step } => {
            let extent = input[axis];
            if size > extent {
                return Err(invalid(
                    operation,
                    "size",
                    "unfold size exceeds the selected extent",
                ));
            }
            let mut output = input.to_vec();
            output[axis] = (extent - size) / step + 1;
            output.push(size);
            output
        }
        ShapeTransform::PixelShuffle(factor) => {
            let square = factor.checked_mul(factor).ok_or_else(|| {
                invalid(
                    operation,
                    "upscale_factor",
                    "squared factor overflows usize",
                )
            })?;
            if input[1] % square != 0 {
                return Err(invalid(
                    operation,
                    "channels",
                    "channel extent must be divisible by the squared upscale factor",
                ));
            }
            vec![
                input[0],
                input[1] / square,
                input[2].checked_mul(factor).ok_or_else(|| {
                    invalid(operation, "height", "pixel-shuffle height overflows usize")
                })?,
                input[3].checked_mul(factor).ok_or_else(|| {
                    invalid(operation, "width", "pixel-shuffle width overflows usize")
                })?,
            ]
        }
        ShapeTransform::AdaptivePool2d(spatial) => {
            let mut output = input.to_vec();
            let rank = output.len();
            output[rank - 2..].copy_from_slice(&spatial);
            output
        }
        ShapeTransform::TopK { axis, k } => {
            let mut output = input.to_vec();
            output[axis] = k;
            output
        }
        ShapeTransform::Chunk { chunks, axis } => {
            let extent = input[axis];
            let chunk_size = extent.div_ceil(chunks);
            let start = output_index
                .checked_mul(chunk_size)
                .ok_or_else(|| invalid(operation, "output", "chunk offset overflows usize"))?;
            if start >= extent {
                return Err(invalid(
                    operation,
                    "output",
                    "chunk output index is invalid",
                ));
            }
            let mut output = input.to_vec();
            output[axis] = (extent - start).min(chunk_size);
            output
        }
        ShapeTransform::Split { split_size, axis } => {
            let extent = input[axis];
            let start = output_index
                .checked_mul(split_size)
                .ok_or_else(|| invalid(operation, "output", "split offset overflows usize"))?;
            if start >= extent {
                return Err(invalid(
                    operation,
                    "output",
                    "split output index is invalid",
                ));
            }
            let mut output = input.to_vec();
            output[axis] = (extent - start).min(split_size);
            output
        }
        ShapeTransform::Conv1d(attributes) => {
            let Some(weight) = inputs.get(1).and_then(|value| value.shape.as_deref()) else {
                return Ok(Some(None));
            };
            if weight.len() != 3
                || weight[1]
                    .checked_mul(attributes.groups)
                    .is_none_or(|channels| channels != input[input.len() - 2])
                || inputs
                    .get(2)
                    .and_then(|value| value.shape.as_deref())
                    .is_some_and(|bias| bias != [weight[0]])
            {
                return Err(invalid(
                    operation,
                    "channels",
                    "conv1d input, grouped weight, and optional bias channels disagree",
                ));
            }
            convolution_output(
                operation,
                input,
                weight,
                &[attributes.stride],
                &[attributes.padding],
                &[attributes.dilation],
            )?
        }
        ShapeTransform::Conv2d(attributes) => {
            let Some(weight) = inputs.get(1).and_then(|value| value.shape.as_deref()) else {
                return Ok(Some(None));
            };
            if weight.len() != 4
                || weight[1]
                    .checked_mul(attributes.groups)
                    .is_none_or(|channels| channels != input[input.len() - 3])
                || inputs
                    .get(2)
                    .and_then(|value| value.shape.as_deref())
                    .is_some_and(|bias| bias != [weight[0]])
            {
                return Err(invalid(
                    operation,
                    "channels",
                    "conv2d input, grouped weight, and optional bias channels disagree",
                ));
            }
            convolution_output(
                operation,
                input,
                weight,
                &attributes.stride,
                &attributes.padding,
                &attributes.dilation,
            )?
        }
        ShapeTransform::ConvTranspose2d(attributes) => {
            let Some(weight) = inputs.get(1).and_then(|value| value.shape.as_deref()) else {
                return Ok(Some(None));
            };
            if input.len() < 3 || weight.len() != 4 {
                return Err(invalid(
                    operation,
                    "rank",
                    "transposed convolution requires rank-three/four input and rank-four weight",
                ));
            }
            if input[input.len() - 3] != weight[0]
                || inputs
                    .get(2)
                    .and_then(|value| value.shape.as_deref())
                    .is_some_and(|bias| {
                        weight[1]
                            .checked_mul(attributes.groups)
                            .is_none_or(|channels| bias != [channels])
                    })
            {
                return Err(invalid(
                    operation,
                    "channels",
                    "transposed convolution input, weight, and optional bias channels disagree",
                ));
            }
            let mut output = input.to_vec();
            let channel_axis = input.len() - 3;
            output[channel_axis] = weight[1]
                .checked_mul(attributes.groups)
                .ok_or_else(|| invalid(operation, "groups", "output channels overflow usize"))?;
            for axis in 0..2 {
                let source = input[input.len() - 2 + axis];
                let kernel = weight[weight.len() - 2 + axis];
                let extent = source
                    .checked_sub(1)
                    .and_then(|value| value.checked_mul(attributes.stride[axis]))
                    .and_then(|value| {
                        kernel
                            .checked_sub(1)
                            .and_then(|kernel| kernel.checked_mul(attributes.dilation[axis]))
                            .and_then(|kernel| value.checked_add(kernel))
                    })
                    .and_then(|value| value.checked_add(attributes.output_padding[axis]))
                    .and_then(|value| value.checked_add(1))
                    .and_then(|value| {
                        attributes.padding[axis]
                            .checked_mul(2)
                            .and_then(|padding| value.checked_sub(padding))
                    })
                    .ok_or_else(|| {
                        invalid(
                            operation,
                            "spatial",
                            "transposed convolution output extent overflows or underflows",
                        )
                    })?;
                output[input.len() - 2 + axis] = extent;
            }
            output
        }
        ShapeTransform::Pool2d(attributes) => {
            let mut output = input.to_vec();
            for axis in 0..2 {
                output[input.len() - 2 + axis] = spatial_output(
                    operation,
                    input[input.len() - 2 + axis],
                    attributes.kernel[axis],
                    attributes.stride[axis],
                    attributes.padding[axis],
                    attributes.dilation[axis],
                )?;
            }
            output
        }
        ShapeTransform::AvgPool2d(attributes) => {
            let mut output = input.to_vec();
            for axis in 0..2 {
                output[input.len() - 2 + axis] = spatial_output(
                    operation,
                    input[input.len() - 2 + axis],
                    attributes.kernel[axis],
                    attributes.stride[axis],
                    attributes.padding[axis],
                    1,
                )?;
            }
            output
        }
        ShapeTransform::Rnn(attributes) => {
            if output_index == 0 {
                vec![input[0], input[1], attributes.hidden_size]
            } else {
                vec![input[0], attributes.hidden_size]
            }
        }
    };
    validate_shape(operation, &output)?;
    Ok(Some(Some(output)))
}

fn inferred_shape<A: AttributeContract>(
    operation: OperationKind,
    row: &OperationCatalogEntry,
    attributes: &A,
    inputs: &[LogicalTensorMeta],
    output_index: usize,
) -> Result<Option<Option<Vec<usize>>>, DescriptorError> {
    let first = inputs.first().and_then(|input| input.shape.clone());
    let inferred = match row.output {
        OutputRule::Created => Some(attributes.declared_shape().map(<[usize]>::to_vec)),
        OutputRule::Preserve | OutputRule::ExplicitDType => Some(first),
        OutputRule::ShapeAttributes => {
            if let Some(shape) = attributes.declared_shape() {
                Some(Some(shape.to_vec()))
            } else {
                transformed_shape(operation, attributes, inputs, output_index)?
            }
        }
        OutputRule::Broadcast => {
            if inputs.iter().any(|input| input.shape.is_none()) {
                Some(None)
            } else {
                let shapes: Vec<&[usize]> = inputs
                    .iter()
                    .filter_map(|input| input.shape.as_deref())
                    .collect();
                Some(Some(broadcast_shape(operation, &shapes)?))
            }
        }
        OutputRule::MatMul => {
            match (
                inputs.first().and_then(|v| v.shape.as_deref()),
                inputs.get(1).and_then(|v| v.shape.as_deref()),
            ) {
                (Some(lhs), Some(rhs)) if lhs.len() >= 2 && rhs.len() >= 2 => {
                    if lhs[lhs.len() - 1] != rhs[rhs.len() - 2] {
                        return Err(invalid(
                            operation,
                            "shape",
                            "matmul contracting dimensions differ",
                        ));
                    }
                    let batch = broadcast_shape(
                        operation,
                        &[&lhs[..lhs.len() - 2], &rhs[..rhs.len() - 2]],
                    )?;
                    let mut shape = batch;
                    shape.push(lhs[lhs.len() - 2]);
                    shape.push(rhs[rhs.len() - 1]);
                    Some(Some(shape))
                }
                (None, _) | (_, None) => Some(None),
                _ => {
                    return Err(invalid(
                        operation,
                        "rank",
                        "matmul inputs require rank at least two",
                    ));
                }
            }
        }
        OutputRule::Reduction => {
            let Some(shape) = inputs.first().and_then(|input| input.shape.as_deref()) else {
                return Ok(Some(None));
            };
            if operation == OperationKind::TopK {
                return transformed_shape(operation, attributes, inputs, output_index);
            }
            if operation == OperationKind::Argsort {
                Some(Some(shape.to_vec()))
            } else if matches!(
                operation,
                OperationKind::MseLoss
                    | OperationKind::L1Loss
                    | OperationKind::BceWithLogitsLoss
                    | OperationKind::CrossEntropyLoss
            ) {
                match attributes.loss_reduction() {
                    Some(LossReduction::None) if operation == OperationKind::CrossEntropyLoss => {
                        Some(inputs.get(1).and_then(|input| input.shape.clone()))
                    }
                    Some(LossReduction::None) => Some(Some(shape.to_vec())),
                    Some(LossReduction::Mean | LossReduction::Sum) => Some(Some(Vec::new())),
                    None => None,
                }
            } else if matches!(
                operation,
                OperationKind::SumAll
                    | OperationKind::MeanAll
                    | OperationKind::MaxAll
                    | OperationKind::MinAll
                    | OperationKind::ProdAll
                    | OperationKind::Norm
                    | OperationKind::VarianceAll
                    | OperationKind::StdAll
            ) || matches!(operation, OperationKind::ArgMax | OperationKind::ArgMin)
                && attributes.axis().is_none()
            {
                Some(Some(Vec::new()))
            } else if let Some(axis) = attributes.axis() {
                if axis >= shape.len() {
                    return Err(invalid(
                        operation,
                        "axis",
                        "reduction axis is outside the input rank",
                    ));
                }
                let keep = matches!(
                    operation,
                    OperationKind::SumKeepDim
                        | OperationKind::MeanKeepDim
                        | OperationKind::MaxKeepDim
                        | OperationKind::MinKeepDim
                        | OperationKind::VarianceKeepDim
                        | OperationKind::StdKeepDim
                );
                let mut output = shape.to_vec();
                if keep {
                    output[axis] = 1;
                } else {
                    output.remove(axis);
                }
                Some(Some(output))
            } else {
                None
            }
        }
        OutputRule::HostValue => Some(None),
        OutputRule::Indexing | OutputRule::TypedInference => match operation {
            OperationKind::Gather => Some(inputs.get(1).and_then(|input| input.shape.clone())),
            OperationKind::IndexSelect => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
                attributes.axis(),
            ) {
                (Some(source), Some(indices), Some(axis)) => {
                    let count =
                        crate::prelude::ShapeBuf::from_slice(indices).checked_numel(operation)?;
                    let mut output = source.to_vec();
                    output[axis] = count;
                    Some(Some(output))
                }
                _ => Some(None),
            },
            OperationKind::EmbeddingExact => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
            ) {
                (Some(indices), Some(weight)) if weight.len() == 2 => {
                    let mut output = indices.to_vec();
                    output.push(weight[1]);
                    Some(Some(output))
                }
                (None, _) | (_, None) => Some(None),
                _ => {
                    return Err(invalid(
                        operation,
                        "weight",
                        "embedding weight must have rank two",
                    ));
                }
            },
            OperationKind::Dot => Some(Some(Vec::new())),
            OperationKind::Outer => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
            ) {
                (Some(lhs), Some(rhs)) => Some(Some(vec![lhs[0], rhs[0]])),
                _ => Some(None),
            },
            OperationKind::Addmm => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
                inputs.get(2).and_then(|input| input.shape.as_deref()),
            ) {
                (Some(addend), Some(lhs), Some(rhs)) if lhs.len() >= 2 && rhs.len() >= 2 => {
                    if lhs[lhs.len() - 1] != rhs[rhs.len() - 2] {
                        return Err(invalid(
                            operation,
                            "shape",
                            "addmm contracting dimensions differ",
                        ));
                    }
                    let mut product = broadcast_shape(
                        operation,
                        &[&lhs[..lhs.len() - 2], &rhs[..rhs.len() - 2]],
                    )?;
                    product.push(lhs[lhs.len() - 2]);
                    product.push(rhs[rhs.len() - 1]);
                    Some(Some(broadcast_shape(operation, &[addend, &product])?))
                }
                (None, _, _) | (_, None, _) | (_, _, None) => Some(None),
                _ => {
                    return Err(invalid(
                        operation,
                        "rank",
                        "addmm matrix operands require rank at least two",
                    ));
                }
            },
            OperationKind::ScaledDotProductAttention => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
                inputs.get(2).and_then(|input| input.shape.as_deref()),
            ) {
                (Some(query), Some(key), Some(value))
                    if query.len() >= 2 && key.len() >= 2 && value.len() >= 2 =>
                {
                    if query[query.len() - 1] != key[key.len() - 1]
                        || key[key.len() - 2] != value[value.len() - 2]
                    {
                        return Err(invalid(
                            operation,
                            "shape",
                            "attention query/key width or key/value sequence extents differ",
                        ));
                    }
                    let mut output = broadcast_shape(
                        operation,
                        &[
                            &query[..query.len() - 2],
                            &key[..key.len() - 2],
                            &value[..value.len() - 2],
                        ],
                    )?;
                    output.push(query[query.len() - 2]);
                    output.push(value[value.len() - 1]);
                    Some(Some(output))
                }
                (None, _, _) | (_, None, _) | (_, _, None) => Some(None),
                _ => {
                    return Err(invalid(
                        operation,
                        "rank",
                        "attention operands require rank at least two",
                    ));
                }
            },
            OperationKind::Linear => match (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
            ) {
                (Some(input), Some(weight)) if !input.is_empty() && weight.len() == 2 => {
                    if input[input.len() - 1] != weight[1]
                        || inputs
                            .get(2)
                            .and_then(|value| value.shape.as_deref())
                            .is_some_and(|bias| bias != [weight[0]])
                    {
                        return Err(invalid(
                            operation,
                            "shape",
                            "linear input width, weight, and optional bias disagree",
                        ));
                    }
                    let mut output = input.to_vec();
                    let last = output.len() - 1;
                    output[last] = weight[0];
                    Some(Some(output))
                }
                (None, _) | (_, None) => Some(None),
                _ => {
                    return Err(invalid(
                        operation,
                        "rank",
                        "linear requires non-scalar input and rank-two weight",
                    ));
                }
            },
            OperationKind::SgdStep => Some(inputs.first().and_then(|v| v.shape.clone())),
            OperationKind::AdamStep | OperationKind::AdamWStep => {
                let source = match output_index {
                    0 => 0,
                    1 => 2,
                    _ => 3,
                };
                Some(inputs.get(source).and_then(|v| v.shape.clone()))
            }
            OperationKind::Quantize
            | OperationKind::Dequantize
            | OperationKind::QuantizedMatMul => Some(first),
            _ => return Err(DescriptorError::MissingInference { operation }),
        },
    };
    Ok(inferred)
}

fn verify_outputs<A: AttributeContract>(
    operation: OperationKind,
    row: &OperationCatalogEntry,
    attributes: &A,
    inputs: &[LogicalTensorMeta],
    outputs: &[LogicalTensorMeta],
) -> Result<(), DescriptorError> {
    let first_dtype = inputs.first().and_then(|input| input.dtype);
    let is_float = |dtype: DTypeId| {
        matches!(
            dtype,
            DTypeId::BF16 | DTypeId::F16 | DTypeId::F32 | DTypeId::F64
        )
    };
    let is_integer = |dtype: DTypeId| matches!(dtype, DTypeId::U8 | DTypeId::U32 | DTypeId::I64);

    if matches!(
        row.profile,
        SemanticProfile::BinaryBroadcast
            | SemanticProfile::Comparison
            | SemanticProfile::Logical
            | SemanticProfile::Mutation
            | SemanticProfile::MatMul
    ) {
        for input in inputs.iter().skip(1) {
            if let (Some(expected), Some(actual)) = (first_dtype, input.dtype) {
                if expected != actual {
                    return Err(invalid(
                        operation,
                        "dtype",
                        "operation inputs require the same dtype",
                    ));
                }
            }
        }
    }

    let index_input = match operation {
        OperationKind::Gather | OperationKind::Scatter | OperationKind::IndexSelect => Some(1),
        OperationKind::EmbeddingExact => Some(0),
        OperationKind::CrossEntropyLoss => Some(1),
        _ => None,
    };
    let require_float = matches!(
        row.profile,
        SemanticProfile::UnaryFloat
            | SemanticProfile::MatMul
            | SemanticProfile::Attention
            | SemanticProfile::Reduction
            | SemanticProfile::Normalization
            | SemanticProfile::Loss
            | SemanticProfile::Optimizer
    ) || matches!(
        row.profile,
        SemanticProfile::Module | SemanticProfile::Composite
    ) && operation != OperationKind::EmbeddingExact;
    if require_float {
        for (index, input) in inputs.iter().enumerate() {
            if Some(index) == index_input {
                continue;
            }
            if input.dtype.is_some_and(|dtype| !is_float(dtype)) {
                return Err(invalid(
                    operation,
                    "dtype",
                    "operation requires floating-point input metadata",
                ));
            }
        }
    }
    if let Some(index) = index_input {
        if inputs
            .get(index)
            .and_then(|input| input.dtype)
            .is_some_and(|dtype| !is_integer(dtype))
        {
            return Err(invalid(
                operation,
                "index dtype",
                "index metadata requires an integer dtype",
            ));
        }
    }
    let same_dtype_pair = match operation {
        OperationKind::WhereCond => Some((1, 2)),
        OperationKind::Scatter => Some((0, 2)),
        _ => None,
    };
    if let Some((left, right)) = same_dtype_pair {
        if let (Some(expected), Some(actual)) = (
            inputs.get(left).and_then(|input| input.dtype),
            inputs.get(right).and_then(|input| input.dtype),
        ) {
            if expected != actual {
                return Err(invalid(
                    operation,
                    "dtype",
                    "value operands require the same dtype",
                ));
            }
        }
    }
    if operation == OperationKind::EmbeddingExact
        && inputs
            .get(1)
            .and_then(|input| input.dtype)
            .is_some_and(|dtype| !is_float(dtype))
    {
        return Err(invalid(
            operation,
            "weight dtype",
            "embedding weight metadata requires a floating dtype",
        ));
    }
    match operation {
        OperationKind::Quantize => {
            if first_dtype.is_some_and(|dtype| !is_float(dtype))
                || attributes.declared_dtype() != Some(DTypeId::Q8_0)
            {
                return Err(invalid(
                    operation,
                    "dtype",
                    "quantize requires floating input and q8_0 output metadata",
                ));
            }
        }
        OperationKind::Dequantize => {
            if first_dtype.is_some_and(|dtype| dtype != DTypeId::Q8_0)
                || attributes
                    .declared_dtype()
                    .is_some_and(|dtype| !is_float(dtype))
            {
                return Err(invalid(
                    operation,
                    "dtype",
                    "dequantize requires q8_0 input and floating output metadata",
                ));
            }
        }
        OperationKind::QuantizedMatMul => {
            if inputs
                .iter()
                .filter_map(|input| input.dtype)
                .any(|dtype| dtype != DTypeId::Q8_0)
            {
                return Err(invalid(
                    operation,
                    "dtype",
                    "quantized matmul requires q8_0 inputs",
                ));
            }
        }
        _ => {}
    }

    for (index, output) in outputs.iter().enumerate() {
        let expected = expected_output(operation, row, attributes, inputs, index)?;
        if output.device != expected.device {
            return Err(DescriptorError::MetadataMismatch {
                operation,
                output: index,
                field: "device",
            });
        }
        if output.dtype != expected.dtype {
            return Err(DescriptorError::MetadataMismatch {
                operation,
                output: index,
                field: "dtype",
            });
        }

        match expected.shape {
            Some(expected_shape) => {
                if output.shape != expected_shape {
                    return Err(DescriptorError::MetadataMismatch {
                        operation,
                        output: index,
                        field: "shape",
                    });
                }
            }
            // No inference branch produced an expectation. Accepting the
            // caller's shape here would let fully known inputs certify an
            // output that nothing checked, which is exactly the fabrication
            // this contract exists to prevent. Unknown input metadata
            // legitimately yields no expectation and stays unknown; known
            // inputs must fail closed instead.
            None => {
                if inputs_are_known(inputs) {
                    return Err(DescriptorError::MissingInference { operation });
                }
            }
        }
        if let Some(shape) = &output.shape {
            validate_shape(operation, shape)?;
        }
    }
    Ok(())
}

fn inputs_are_known(inputs: &[LogicalTensorMeta]) -> bool {
    !inputs.is_empty() && inputs.iter().all(|input| input.shape.is_some())
}

/// The output metadata the contract requires at `index`.
///
/// `shape` is `None` when no inference branch applies at all, and
/// `Some(None)` when a branch applies but the inputs it reads are unknown.
/// Verification and inference both read this one function, so an inferred
/// output can never disagree with a verified one.
struct ExpectedOutput {
    device: Option<DeviceId>,
    dtype: Option<DTypeId>,
    shape: Option<Option<Vec<usize>>>,
}

fn expected_output<A: AttributeContract>(
    operation: OperationKind,
    row: &OperationCatalogEntry,
    attributes: &A,
    inputs: &[LogicalTensorMeta],
    index: usize,
) -> Result<ExpectedOutput, DescriptorError> {
    let first_dtype = inputs.first().and_then(|input| input.dtype);
    let first_device = inputs.first().and_then(|input| input.device);
    let dtype = match operation {
        OperationKind::WhereCond | OperationKind::EmbeddingExact => {
            inputs.get(1).and_then(|input| input.dtype)
        }
        OperationKind::TopK if index == 0 => first_dtype,
        OperationKind::ArgMax | OperationKind::ArgMin | OperationKind::Argsort => {
            attributes.declared_dtype()
        }
        OperationKind::TopK => attributes.declared_dtype(),
        OperationKind::QuantizedMatMul => Some(DTypeId::F32),
        _ => attributes.declared_dtype().or(first_dtype),
    };
    Ok(ExpectedOutput {
        device: attributes.declared_device().or(first_device),
        dtype,
        shape: inferred_shape(operation, row, attributes, inputs, index)?,
    })
}

/// Derive the outputs an invocation must produce, instead of trusting a caller
/// to state them.
///
/// This is the entry point execution uses. A caller that never supplies output
/// metadata cannot fabricate it, so the "no output is invented" contract holds
/// by construction rather than by a comparison the caller could satisfy with a
/// lucky guess.
fn infer_outputs<A: AttributeContract>(
    operation: OperationKind,
    row: &OperationCatalogEntry,
    attributes: &A,
    inputs: &[LogicalTensorMeta],
) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
    let count = attributes
        .expected_output_count(inputs)
        .unwrap_or(*row.output_arity.start());
    let mut outputs = Vec::with_capacity(count);
    for index in 0..count {
        let expected = expected_output(operation, row, attributes, inputs, index)?;
        let shape = match expected.shape {
            Some(shape) => shape,
            None if inputs_are_known(inputs) => {
                return Err(DescriptorError::MissingInference { operation });
            }
            None => None,
        };
        outputs.push(LogicalTensorMeta {
            shape,
            dtype: expected.dtype,
            device: expected.device,
        });
    }
    Ok(outputs)
}

/// The exact rank contract for one operand role.
///
/// `OperationCatalogEntry::accepted_ranks` describes the *primary* operand, the
/// activation, the value being reduced, the tensor being reshaped. Applying it
/// to every input is wrong for any operation whose operands carry different
/// contracts: a rank-one convolution bias is not a rank-four activation, and an
/// embedding table is not the index batch that addresses it.
///
/// A role listed here overrides the primary window for that position only, so
/// widening an activation's accepted ranks can never silently widen a
/// parameter's. Roles absent from this table keep the primary window, which is
/// the correct contract for genuinely homogeneous operands (broadcast
/// arithmetic, elementwise losses, comparisons).
fn operand_ranks(
    operation: OperationKind,
    row: &OperationCatalogEntry,
    index: usize,
) -> core::ops::RangeInclusive<usize> {
    use OperationKind::*;
    let exact = |rank: usize| Some(rank..=rank);
    let override_range = match (operation, index) {
        // Convolution: the filter bank is fixed at `[c_out, c_in / groups,
        // ..spatial]`, and a bias holds one value per output channel.
        (Conv1dExact, 1) => exact(3),
        (Conv2dExact | ConvTranspose2d, 1) => exact(4),
        (Conv1dExact | Conv2dExact | ConvTranspose2d, 2) => exact(1),
        // The embedding table is always `[num_embeddings, dim]`; the indices
        // carry whatever batch geometry the caller addresses it with.
        (EmbeddingExact, 0) => Some(1..=crate::prelude::MAX_RANK),
        (EmbeddingExact, 1) => exact(2),
        // Batch-norm affine parameters and running state are per-channel
        // vectors, never activations. `BatchNormAttributes::validate` already
        // pins their extent; this pins the rank even when the extent is unknown.
        (BatchNorm, 1..=4) => exact(1),
        (RmsNorm, 1) => exact(1),
        // `Linear` weights are `[out, in]` and biases `[out]`.
        (Linear, 1) => exact(2),
        (Linear, 2) => exact(1),
        // Cross entropy consumes `[batch, classes]` logits against `[batch]`
        // class indices; the two operands are not interchangeable.
        (CrossEntropyLoss, 0) => exact(2),
        (CrossEntropyLoss, 1) => exact(1),
        // `index_select` addresses one axis with a flat index vector.
        (IndexSelect, 1) => exact(1),
        // Optimizer gradients and moment state mirror the parameter they
        // update, which the primary window already covers; the equal-shape
        // requirement is enforced separately in `validate`.
        _ => None,
    };
    override_range.unwrap_or_else(|| row.accepted_ranks.clone())
}

/// Opaque proof that exact input/output metadata was validated without storage.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedInvocation<O: CanonicalOperation> {
    validated: crate::exec::Validated<Descriptor<O>>,
    inputs: Vec<LogicalTensorMeta>,
}

impl<O: CanonicalOperation> ValidatedInvocation<O> {
    /// Internal lowering entry point. The output is supplied by the typed
    /// frontend proof; callers outside `incin-core` cannot assert it.
    pub(crate) fn validate(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        outputs: Vec<LogicalTensorMeta>,
        proof: crate::exec::ProofLevel,
    ) -> Result<Self, DescriptorError> {
        let row = catalog_entry(O::ID)
            .ok_or(DescriptorError::MissingCatalogEntry { operation: O::ID })?;
        if !row.input_arity.contains(&inputs.len()) {
            return Err(DescriptorError::Arity {
                operation: O::ID,
                expected: row.input_arity.clone(),
                actual: inputs.len(),
            });
        }
        if !row.output_arity.contains(&outputs.len()) {
            return Err(DescriptorError::OutputArity {
                operation: O::ID,
                expected: row.output_arity.clone(),
                actual: outputs.len(),
            });
        }
        attributes.validate(O::ID, &inputs)?;
        if let Some(has_bias) = attributes.optional_bias() {
            let expected = if has_bias { 3 } else { 2 };
            if inputs.len() != expected {
                return Err(DescriptorError::Arity {
                    operation: O::ID,
                    expected: expected..=expected,
                    actual: inputs.len(),
                });
            }
        }
        if row.empty == EmptyRule::RejectedWhenReductionIsEmpty {
            if let Some(shape) = inputs.first().and_then(|input| input.shape.as_deref()) {
                let reduction_is_empty = attributes
                    .axis()
                    .and_then(|axis| shape.get(axis))
                    .map_or_else(|| shape.contains(&0), |&extent| extent == 0);
                if reduction_is_empty {
                    return Err(invalid(
                        O::ID,
                        "shape",
                        "operation rejects an empty reduction domain",
                    ));
                }
            }
        }
        if let Some(expected) = attributes.expected_output_count(&inputs) {
            if outputs.len() != expected {
                return Err(DescriptorError::OutputArity {
                    operation: O::ID,
                    expected: expected..=expected,
                    actual: outputs.len(),
                });
            }
        }
        // An optimizer step reads and writes one parameter's worth of state.
        // The gradient and every moment buffer describe the same tensor, so a
        // mismatch here is a state-dictionary defect that must fail before any
        // parameter or moment is mutated rather than after a partial update.
        if matches!(
            O::ID,
            OperationKind::SgdStep | OperationKind::AdamStep | OperationKind::AdamWStep
        ) {
            if let Some(parameter) = inputs.first().and_then(|input| input.shape.as_deref()) {
                for (offset, operand) in inputs.iter().enumerate().skip(1) {
                    if let Some(shape) = operand.shape.as_deref() {
                        if shape != parameter {
                            return Err(invalid(
                                O::ID,
                                match offset {
                                    1 => "gradient shape",
                                    _ => "optimizer state shape",
                                },
                                "optimizer gradient and state must match the parameter shape",
                            ));
                        }
                    }
                }
            }
        }
        if O::ID == OperationKind::Dot {
            if let (Some(lhs), Some(rhs)) = (
                inputs.first().and_then(|input| input.shape.as_deref()),
                inputs.get(1).and_then(|input| input.shape.as_deref()),
            ) {
                if lhs[0] != rhs[0] {
                    return Err(invalid(
                        O::ID,
                        "shape",
                        "dot inputs must have equal extents",
                    ));
                }
            }
        }
        match O::ID {
            OperationKind::MaskedFill => {
                if let (Some(value), Some(mask)) = (
                    inputs.first().and_then(|input| input.shape.as_deref()),
                    inputs.get(1).and_then(|input| input.shape.as_deref()),
                ) {
                    if value != mask {
                        return Err(invalid(
                            O::ID,
                            "mask",
                            "masked_fill currently requires mask and value shapes to match",
                        ));
                    }
                }
            }
            OperationKind::Gather => {
                if let (Some(source), Some(indices), Some(axis)) = (
                    inputs.first().and_then(|input| input.shape.as_deref()),
                    inputs.get(1).and_then(|input| input.shape.as_deref()),
                    attributes.axis(),
                ) {
                    if source.len() != indices.len()
                        || indices
                            .iter()
                            .enumerate()
                            .any(|(index, &extent)| index != axis && extent > source[index])
                    {
                        return Err(invalid(
                            O::ID,
                            "index shape",
                            "gather indices must match source rank and fit non-gather extents",
                        ));
                    }
                }
            }
            OperationKind::Scatter => {
                if let (Some(target), Some(indices), Some(source), Some(axis)) = (
                    inputs.first().and_then(|input| input.shape.as_deref()),
                    inputs.get(1).and_then(|input| input.shape.as_deref()),
                    inputs.get(2).and_then(|input| input.shape.as_deref()),
                    attributes.axis(),
                ) {
                    if indices != source
                        || target.len() != indices.len()
                        || indices
                            .iter()
                            .enumerate()
                            .any(|(index, &extent)| index != axis && extent > target[index])
                    {
                        return Err(invalid(
                            O::ID,
                            "index/source shape",
                            "scatter index/source shapes must match and fit the target",
                        ));
                    }
                }
            }
            OperationKind::IndexSelect => {
                if let Some(indices) = inputs.get(1).and_then(|input| input.shape.as_deref()) {
                    if indices.len() != 1 {
                        return Err(invalid(
                            O::ID,
                            "index shape",
                            "index_select requires a rank-one index tensor",
                        ));
                    }
                }
            }
            OperationKind::EmbeddingExact => {
                if let Some(weight) = inputs.get(1).and_then(|input| input.shape.as_deref()) {
                    if weight.len() != 2 {
                        return Err(invalid(
                            O::ID,
                            "weight shape",
                            "embedding weight must have rank two",
                        ));
                    }
                }
            }
            OperationKind::MseLoss | OperationKind::L1Loss | OperationKind::BceWithLogitsLoss => {
                if let (Some(prediction), Some(target)) = (
                    inputs.first().and_then(|input| input.shape.as_deref()),
                    inputs.get(1).and_then(|input| input.shape.as_deref()),
                ) {
                    if prediction != target {
                        return Err(invalid(
                            O::ID,
                            "target shape",
                            "elementwise loss prediction and target shapes must match",
                        ));
                    }
                }
            }
            OperationKind::CrossEntropyLoss => {
                if let (Some(prediction), Some(target)) = (
                    inputs.first().and_then(|input| input.shape.as_deref()),
                    inputs.get(1).and_then(|input| input.shape.as_deref()),
                ) {
                    if prediction.len() != 2 || target.len() != 1 || target[0] != prediction[0] {
                        return Err(invalid(
                            O::ID,
                            "target shape",
                            "cross entropy requires logits [batch, classes] and targets [batch]",
                        ));
                    }
                }
            }
            _ => {}
        }
        let mut expected_device = None;
        for (index, input) in inputs.iter().enumerate() {
            if let Some(shape) = &input.shape {
                let expected = operand_ranks(O::ID, row, index);
                if !expected.contains(&shape.len()) {
                    return Err(DescriptorError::Rank {
                        operation: O::ID,
                        input: index,
                        expected,
                        actual: shape.len(),
                    });
                }
            }
            if row.same_device {
                if let Some(device) = input.device {
                    if let Some(expected) = expected_device {
                        if device != expected {
                            return Err(DescriptorError::DeviceMismatch {
                                operation: O::ID,
                                input: index,
                                expected,
                                actual: device,
                            });
                        }
                    } else {
                        expected_device = Some(device);
                    }
                }
            }
        }
        verify_outputs(O::ID, row, &attributes, &inputs, &outputs)?;
        let descriptor = Descriptor {
            attributes,
            outputs,
            marker: PhantomData,
        };
        Ok(Self {
            validated: crate::exec::Validated::new(descriptor, proof),
            inputs,
        })
    }

    /// Validate an invocation whose outputs are derived rather than supplied.
    ///
    /// The execution path uses this form. `validate` exists for a caller that
    /// already holds output metadata and needs it checked; here there is no
    /// such caller, so there is nothing to check against and nothing to forge.
    pub(crate) fn infer(
        attributes: O::Attributes,
        inputs: Vec<LogicalTensorMeta>,
        proof: crate::exec::ProofLevel,
    ) -> Result<Self, DescriptorError> {
        let row = catalog_entry(O::ID)
            .ok_or(DescriptorError::MissingCatalogEntry { operation: O::ID })?;
        attributes.validate(O::ID, &inputs)?;
        let outputs = infer_outputs(O::ID, row, &attributes, &inputs)?;
        Self::validate(attributes, inputs, outputs, proof)
    }

    #[must_use]
    pub const fn descriptor(&self) -> &Descriptor<O> {
        self.validated.descriptor()
    }
    #[must_use]
    pub const fn validated(&self) -> &crate::exec::Validated<Descriptor<O>> {
        &self.validated
    }
    #[must_use]
    pub fn inputs(&self) -> &[LogicalTensorMeta] {
        &self.inputs
    }
}

#[must_use]
pub fn catalog_entry(operation: OperationKind) -> Option<&'static OperationCatalogEntry> {
    OPERATION_CATALOG
        .iter()
        .find(|row| row.operation == operation)
}

/// Render the human-reviewed semantics inventory from the code catalog.
#[must_use]
pub fn operation_semantics_document() -> alloc::string::String {
    use core::fmt::Write as _;
    let mut out = alloc::string::String::from(
        "# Canonical operation semantics\n\nThis file is generated from `incin_core::exec::OPERATION_CATALOG`; the Rust catalog is authoritative. Families classify operations and never imply backend support. `TypedContract` and `TypedInference` refer to the exact descriptor's typed attribute validator and checked inference branch; they do not permit a backend-specific default.\n\n| ID | Descriptor | Attributes | Input/output arity | Rank | Broadcast | Dtype/output | Empty/non-finite | Gradient | Deterministic | Layout | Legacy mapping |\n|---|---|---|---|---|---|---|---|---|:--:|---|---|\n",
    );
    for row in OPERATION_CATALOG {
        let max_arity = if *row.input_arity.end() == usize::MAX {
            alloc::string::String::from("many")
        } else {
            row.input_arity.end().to_string()
        };
        let max_output_arity = if *row.output_arity.end() == usize::MAX {
            alloc::string::String::from("many")
        } else {
            row.output_arity.end().to_string()
        };
        let _ = writeln!(
            out,
            "| `{}` | `{}` | `{}` | {}-{} / {}-{} | {}-{} | `{:?}` | `{:?}` / `{:?}` | `{:?}` / `{:?}` | `{:?}` | {} | `{:?}` | `{}` |",
            row.name,
            row.descriptor,
            row.attributes,
            row.input_arity.start(),
            max_arity,
            row.output_arity.start(),
            max_output_arity,
            row.accepted_ranks.start(),
            row.accepted_ranks.end(),
            row.broadcasting,
            row.dtype,
            row.output,
            row.empty,
            row.numeric,
            row.gradient,
            if row.deterministic { "yes" } else { "no" },
            row.layout,
            row.legacy_source,
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    #[test]
    fn identities_and_names_occur_exactly_once() {
        let mut ids = BTreeSet::new();
        let mut names = BTreeSet::new();
        for row in OPERATION_CATALOG {
            assert!(row.operation.is_exact());
            assert!(
                ids.insert(row.operation),
                "duplicate identity {}",
                row.operation
            );
            assert!(names.insert(row.name), "duplicate name {}", row.name);
            assert_eq!(row.operation.name(), row.name);
        }
    }

    #[test]
    fn generated_semantics_document_covers_every_row() {
        let document = operation_semantics_document();
        for row in OPERATION_CATALOG {
            assert!(document.contains(&alloc::format!("| `{}` |", row.name)));
        }
    }

    #[test]
    fn every_typed_output_rule_is_fail_closed_or_exactly_inferred() {
        for row in OPERATION_CATALOG
            .iter()
            .filter(|row| row.output == OutputRule::TypedInference)
        {
            let inferred = matches!(
                row.operation,
                OperationKind::Dot
                    | OperationKind::Outer
                    | OperationKind::Addmm
                    | OperationKind::ScaledDotProductAttention
                    | OperationKind::Linear
                    | OperationKind::EmbeddingExact
                    | OperationKind::Quantize
                    | OperationKind::Dequantize
                    | OperationKind::SgdStep
                    | OperationKind::AdamStep
                    | OperationKind::AdamWStep
            );
            assert!(
                inferred || *row.output_arity.end() == 0,
                "{} has no typed output inference branch",
                row.operation
            );
        }
    }

    #[test]
    fn unknown_metadata_stays_unknown() {
        let invocation = ValidatedInvocation::<op::Relu>::validate(
            NoAttributes,
            vec![LogicalTensorMeta::unknown()],
            vec![LogicalTensorMeta::unknown()],
            crate::exec::ProofLevel::Dynamic,
        )
        .unwrap();
        assert_eq!(
            invocation.descriptor().outputs(),
            &[LogicalTensorMeta::unknown()]
        );
    }

    #[test]
    fn validation_rejects_wrong_arity_and_cross_device_inputs() {
        let cpu = LogicalTensorMeta {
            shape: Some(vec![2]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        let cuda = LogicalTensorMeta {
            shape: Some(vec![2]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cuda(0)),
        };
        assert!(matches!(
            ValidatedInvocation::<op::Add>::validate(
                NoAttributes,
                vec![cpu.clone()],
                vec![cpu.clone()],
                crate::exec::ProofLevel::Dynamic
            ),
            Err(DescriptorError::Arity { .. })
        ));
        assert!(matches!(
            ValidatedInvocation::<op::Add>::validate(
                NoAttributes,
                vec![cpu.clone(), cuda],
                vec![cpu],
                crate::exec::ProofLevel::Dynamic
            ),
            Err(DescriptorError::DeviceMismatch { .. })
        ));
    }

    /// Helper for the per-operand rank tests: known shape, f32, CPU.
    fn meta(shape: &[usize]) -> LogicalTensorMeta {
        typed_meta(shape, DTypeId::F32)
    }

    /// As [`meta`], for operands whose role fixes a non-float dtype.
    fn typed_meta(shape: &[usize], dtype: DTypeId) -> LogicalTensorMeta {
        LogicalTensorMeta {
            shape: Some(shape.to_vec()),
            dtype: Some(dtype),
            device: Some(DeviceId::cpu()),
        }
    }

    /// A rank-one bias must validate against a rank-four activation.
    ///
    /// The catalog's `accepted_ranks` for `Conv2dExact` is the activation
    /// window `3..=4`. Applying it to every operand rejected every biased
    /// convolution, because a bias is `[c_out]`.
    #[test]
    fn convolution_bias_is_validated_by_role_not_by_the_activation_window() {
        let attributes = |has_bias| Conv2dAttributes {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
            has_bias,
        };

        ValidatedInvocation::<op::Conv2dExact>::validate(
            attributes(false),
            vec![meta(&[1, 3, 8, 8]), meta(&[4, 3, 3, 3])],
            vec![meta(&[1, 4, 8, 8])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("an unbiased convolution validates");

        ValidatedInvocation::<op::Conv2dExact>::validate(
            attributes(true),
            vec![meta(&[1, 3, 8, 8]), meta(&[4, 3, 3, 3]), meta(&[4])],
            vec![meta(&[1, 4, 8, 8])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("a rank-one bias validates against a rank-four activation");

        // The role is exact in both directions: an activation-ranked bias is
        // still refused.
        assert!(matches!(
            ValidatedInvocation::<op::Conv2dExact>::validate(
                attributes(true),
                vec![
                    meta(&[1, 3, 8, 8]),
                    meta(&[4, 3, 3, 3]),
                    meta(&[1, 4, 1, 1])
                ],
                vec![meta(&[1, 4, 8, 8])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::Rank { input: 2, .. })
        ));
    }

    #[test]
    fn conv1d_and_transposed_convolution_bias_use_the_same_role_contract() {
        ValidatedInvocation::<op::Conv1dExact>::validate(
            Conv1dAttributes {
                stride: 1,
                padding: 1,
                dilation: 1,
                groups: 1,
                has_bias: true,
            },
            vec![meta(&[1, 3, 8]), meta(&[4, 3, 3]), meta(&[4])],
            vec![meta(&[1, 4, 8])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("conv1d accepts a rank-one bias");

        ValidatedInvocation::<op::ConvTranspose2d>::validate(
            ConvTranspose2dAttributes {
                stride: [1, 1],
                padding: [1, 1],
                output_padding: [0, 0],
                dilation: [1, 1],
                groups: 1,
                has_bias: true,
            },
            vec![meta(&[1, 3, 8, 8]), meta(&[3, 4, 3, 3]), meta(&[4])],
            vec![meta(&[1, 4, 8, 8])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("conv_transpose2d accepts a rank-one bias");

        // A rank-two weight is not a 2-D filter bank.
        assert!(matches!(
            ValidatedInvocation::<op::Conv2dExact>::validate(
                Conv2dAttributes {
                    stride: [1, 1],
                    padding: [0, 0],
                    dilation: [1, 1],
                    groups: 1,
                    has_bias: false,
                },
                vec![meta(&[1, 3, 8, 8]), meta(&[4, 3])],
                vec![meta(&[1, 4, 6, 6])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::Rank { input: 1, .. })
        ));
    }

    #[test]
    fn embedding_indices_and_weight_have_separate_rank_contracts() {
        let indices = typed_meta(&[2, 5], DTypeId::I64);
        ValidatedInvocation::<op::EmbeddingExact>::validate(
            NoAttributes,
            vec![indices.clone(), meta(&[10, 4])],
            vec![meta(&[2, 5, 4])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("rank-two integer indices address a rank-two table");

        // A rank-three table is not an embedding matrix. The typed weight rule
        // reports this before the role window is consulted; either way the
        // descriptor fails closed rather than inferring a table geometry.
        assert!(matches!(
            ValidatedInvocation::<op::EmbeddingExact>::validate(
                NoAttributes,
                vec![indices, meta(&[10, 4, 1])],
                vec![meta(&[2, 5, 4])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "weight shape",
                ..
            }) | Err(DescriptorError::Rank { input: 1, .. })
        ));

        // Indices must address something; a rank-zero scalar is not a batch,
        // and only the per-role window rejects it.
        assert!(matches!(
            ValidatedInvocation::<op::EmbeddingExact>::validate(
                NoAttributes,
                vec![typed_meta(&[], DTypeId::I64), meta(&[10, 4])],
                vec![meta(&[4])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::Rank { input: 0, .. })
        ));
    }

    #[test]
    fn batch_norm_state_is_rank_one_against_a_ranked_activation() {
        let attributes = BatchNormAttributes {
            epsilon: 1e-5,
            momentum: 0.1,
            training: false,
            has_weight: true,
            has_bias: true,
            has_running_mean: true,
            has_running_variance: true,
        };
        ValidatedInvocation::<op::BatchNorm>::validate(
            attributes.clone(),
            vec![
                meta(&[2, 3, 4, 4]),
                meta(&[3]),
                meta(&[3]),
                meta(&[3]),
                meta(&[3]),
            ],
            vec![meta(&[2, 3, 4, 4])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("rank-one affine and running state validate");

        // A rank-two affine parameter is refused. The typed per-channel extent
        // rule reports it first; the role window is the backstop that keeps the
        // rank pinned even where the extent rule cannot reach.
        assert!(matches!(
            ValidatedInvocation::<op::BatchNorm>::validate(
                attributes,
                vec![
                    meta(&[2, 3, 4, 4]),
                    meta(&[1, 3]),
                    meta(&[3]),
                    meta(&[3]),
                    meta(&[3]),
                ],
                vec![meta(&[2, 3, 4, 4])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "parameter shape",
                ..
            }) | Err(DescriptorError::Rank { input: 1, .. })
        ));
        assert_eq!(
            operand_ranks(
                OperationKind::BatchNorm,
                catalog_entry(OperationKind::BatchNorm).unwrap(),
                1
            ),
            1..=1
        );
    }

    #[test]
    fn linear_weight_and_bias_have_separate_rank_contracts() {
        ValidatedInvocation::<op::Linear>::validate(
            LinearAttributes { has_bias: true },
            vec![meta(&[2, 3]), meta(&[4, 3]), meta(&[4])],
            vec![meta(&[2, 4])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("a rank-two weight and rank-one bias validate");

        assert!(matches!(
            ValidatedInvocation::<op::Linear>::validate(
                LinearAttributes { has_bias: true },
                vec![meta(&[2, 3]), meta(&[4, 3]), meta(&[1, 4])],
                vec![meta(&[2, 4])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::Rank { input: 2, .. })
        ));

        assert!(matches!(
            ValidatedInvocation::<op::Linear>::validate(
                LinearAttributes { has_bias: false },
                vec![meta(&[2, 3]), meta(&[4, 3, 1])],
                vec![meta(&[2, 4])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::Rank { input: 1, .. })
        ));
    }

    #[test]
    fn cross_entropy_logits_and_targets_have_separate_rank_contracts() {
        ValidatedInvocation::<op::CrossEntropyLoss>::validate(
            LossAttributes {
                reduction: LossReduction::Mean,
            },
            vec![meta(&[4, 7]), typed_meta(&[4], DTypeId::I64)],
            vec![meta(&[])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("[batch, classes] logits against [batch] integer targets");

        // A rank-two target is a one-hot encoding, which this operation does
        // not consume; it must be refused rather than reinterpreted.
        assert!(matches!(
            ValidatedInvocation::<op::CrossEntropyLoss>::validate(
                LossAttributes {
                    reduction: LossReduction::Mean,
                },
                vec![meta(&[4, 7]), typed_meta(&[4, 7], DTypeId::I64)],
                vec![meta(&[])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::Rank { input: 1, .. })
                | Err(DescriptorError::InvalidAttribute { .. })
        ));
    }

    #[test]
    fn optimizer_gradient_and_state_are_validated_against_the_parameter() {
        let attributes = AdamAttributes {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step: 1,
        };
        ValidatedInvocation::<op::AdamStep>::validate(
            attributes.clone(),
            vec![meta(&[3, 4]), meta(&[3, 4]), meta(&[3, 4]), meta(&[3, 4])],
            vec![meta(&[3, 4]), meta(&[3, 4]), meta(&[3, 4])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("parameter, gradient, and both moments agree");

        // A moment buffer from a differently shaped parameter must fail before
        // any state is mutated, not after a partial update.
        assert!(matches!(
            ValidatedInvocation::<op::AdamStep>::validate(
                attributes,
                vec![meta(&[3, 4]), meta(&[3, 4]), meta(&[3, 4]), meta(&[4, 3])],
                vec![meta(&[3, 4]), meta(&[3, 4]), meta(&[3, 4])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "optimizer state shape",
                ..
            })
        ));

        assert!(matches!(
            ValidatedInvocation::<op::SgdStep>::validate(
                SgdAttributes { learning_rate: 0.1 },
                vec![meta(&[3, 4]), meta(&[3, 5])],
                vec![meta(&[3, 4])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "gradient shape",
                ..
            })
        ));
    }

    /// Family defaults are defaults; an exact operation overrides an incorrect
    /// one. These are the exceptions the FND-004 profile audit found.
    #[test]
    fn index_producing_operations_reject_a_non_integer_index_dtype() {
        // `IndexResult` is the family intent; without this guard a caller could
        // declare a float "index" dtype and have the descriptor certify it.
        assert!(matches!(
            ValidatedInvocation::<op::ArgMax>::validate(
                IndexReductionAttributes {
                    axis: Some(1),
                    dtype: DTypeId::F32,
                },
                vec![meta(&[2, 3])],
                vec![typed_meta(&[2], DTypeId::F32)],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "dtype",
                ..
            })
        ));
        ValidatedInvocation::<op::ArgMax>::validate(
            IndexReductionAttributes {
                axis: Some(1),
                dtype: DTypeId::I64,
            },
            vec![meta(&[2, 3])],
            vec![typed_meta(&[2], DTypeId::I64)],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("an integer index dtype validates");

        assert!(matches!(
            ValidatedInvocation::<op::Argsort>::validate(
                ArgsortAttributes {
                    axis: 0,
                    descending: false,
                    index_dtype: DTypeId::F64,
                },
                vec![meta(&[4])],
                vec![typed_meta(&[4], DTypeId::F64)],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "index_dtype",
                ..
            })
        ));
    }

    /// `topk` is a two-output exception: the values keep the input dtype while
    /// the indices take the declared integer dtype. Neither is the family
    /// default applied uniformly.
    #[test]
    fn topk_values_keep_the_input_dtype_while_indices_take_the_index_dtype() {
        ValidatedInvocation::<op::TopK>::validate(
            TopKAttributes {
                k: 2,
                axis: 0,
                largest: true,
                index_dtype: DTypeId::I64,
            },
            vec![meta(&[4])],
            vec![meta(&[2]), typed_meta(&[2], DTypeId::I64)],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("values in the input dtype, indices in the index dtype");

        // Indices cannot silently adopt the value dtype.
        assert!(matches!(
            ValidatedInvocation::<op::TopK>::validate(
                TopKAttributes {
                    k: 2,
                    axis: 0,
                    largest: true,
                    index_dtype: DTypeId::I64,
                },
                vec![meta(&[4])],
                vec![meta(&[2]), meta(&[2])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::MetadataMismatch {
                output: 1,
                field: "dtype",
                ..
            })
        ));
    }

    /// The `Reduction` family only rejects an *empty* domain. An unbiased
    /// estimate is also undefined on a single element, because the correction
    /// divides by `n - 1`.
    #[test]
    fn an_unbiased_estimate_rejects_a_single_element_domain() {
        assert!(matches!(
            ValidatedInvocation::<op::VarianceAll>::validate(
                VarianceAttributes { unbiased: true },
                vec![meta(&[1])],
                vec![meta(&[])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "unbiased",
                ..
            })
        ));
        // The biased estimate over the same domain is well defined.
        ValidatedInvocation::<op::VarianceAll>::validate(
            VarianceAttributes { unbiased: false },
            vec![meta(&[1])],
            vec![meta(&[])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("a biased estimate divides by n");

        assert!(matches!(
            ValidatedInvocation::<op::StdDim>::validate(
                AxisVarianceAttributes {
                    axis: 1,
                    unbiased: true,
                },
                vec![meta(&[4, 1])],
                vec![meta(&[4])],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "unbiased",
                ..
            })
        ));
    }

    /// A comparison returns a mask in the operand dtype, not a boolean tensor:
    /// `DTypeId` has no boolean member. `DTypeRule::Preserve` records that
    /// truthfully, and the output dtype must follow the input.
    #[test]
    fn a_comparison_mask_keeps_the_operand_dtype() {
        ValidatedInvocation::<op::CmpLt>::validate(
            NoAttributes,
            vec![meta(&[3]), meta(&[3])],
            vec![meta(&[3])],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("the mask is encoded in the operand dtype");

        assert!(matches!(
            ValidatedInvocation::<op::CmpLt>::validate(
                NoAttributes,
                vec![meta(&[3]), meta(&[3])],
                vec![typed_meta(&[3], DTypeId::U8)],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::MetadataMismatch { field: "dtype", .. })
        ));
    }

    /// Sum and product have an identity on an empty domain; mean and the
    /// extrema do not. The family default is the strict one, so the identity
    /// cases are the overrides.
    #[test]
    fn empty_domain_behaviour_splits_within_the_reduction_family() {
        for operation in [
            OperationKind::SumAll,
            OperationKind::ProdAll,
            OperationKind::SumDim,
            OperationKind::Cumsum,
        ] {
            assert_eq!(
                catalog_entry(operation).unwrap().empty,
                EmptyRule::IdentityOrDefined,
                "{operation} has an identity on an empty domain",
            );
        }
        for operation in [
            OperationKind::MeanAll,
            OperationKind::MaxAll,
            OperationKind::MinAll,
        ] {
            assert_eq!(
                catalog_entry(operation).unwrap().empty,
                EmptyRule::RejectedWhenReductionIsEmpty,
                "{operation} is undefined on an empty domain",
            );
        }
    }

    /// Operations whose result is not a tensor declare zero outputs, and
    /// non-differentiable and nondeterministic exceptions override their
    /// family default rather than inheriting it.
    #[test]
    fn zero_output_gradient_and_determinism_exceptions_override_their_family() {
        for operation in [
            OperationKind::ToHostFloatScalar,
            OperationKind::ToHostIntVec,
            OperationKind::TensorToBytes,
            OperationKind::Backward,
        ] {
            let row = catalog_entry(operation).unwrap();
            assert_eq!(*row.output_arity.end(), 0, "{operation} returns no tensor");
            assert_eq!(row.gradient, GradientRule::None, "{operation}");
        }

        // Piecewise-constant unary functions are float-typed but have no
        // useful gradient, unlike the rest of `UnaryFloat`.
        for operation in [
            OperationKind::Step,
            OperationKind::Sign,
            OperationKind::Floor,
            OperationKind::Round,
        ] {
            assert_eq!(
                catalog_entry(operation).unwrap().gradient,
                GradientRule::Undefined,
                "{operation}",
            );
        }

        for operation in [
            OperationKind::UniformRandom,
            OperationKind::Dropout,
            OperationKind::TopK,
            OperationKind::Argsort,
        ] {
            assert!(
                !catalog_entry(operation).unwrap().deterministic,
                "{operation} is not deterministic",
            );
        }

        // Transfer operations deliberately change device or dtype, so they are
        // the family that opts out of the same-device requirement.
        for operation in [OperationKind::ToDevice, OperationKind::ToDType] {
            assert!(
                !catalog_entry(operation).unwrap().same_device,
                "{operation} may cross devices",
            );
        }
        assert!(catalog_entry(OperationKind::Add).unwrap().same_device);
    }

    /// Known inputs must never certify an unchecked output shape.
    ///
    /// `verify_outputs` used to accept whatever shape the caller supplied
    /// whenever no inference branch produced an expectation. That let a fully
    /// known input set certify a fabricated output. Unknown inputs still
    /// legitimately produce unknown outputs; known inputs now fail closed with
    /// `MissingInference`.
    #[test]
    fn known_inputs_never_skip_output_shape_verification() {
        // `argmax` without an axis over a *known* input has an exact answer
        // (a scalar), so it is verified rather than waved through.
        assert!(matches!(
            ValidatedInvocation::<op::ArgMax>::validate(
                IndexReductionAttributes {
                    axis: None,
                    dtype: DTypeId::I64,
                },
                vec![meta(&[2, 3])],
                vec![typed_meta(&[9, 9], DTypeId::I64)],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::MetadataMismatch { field: "shape", .. })
        ));

        // Unknown input shape keeps the output shape unknown rather than
        // inventing one, and rather than failing. The index dtype is declared
        // by the attributes, so it stays known and is still verified.
        let output = LogicalTensorMeta {
            shape: None,
            dtype: Some(DTypeId::I64),
            device: None,
        };
        let unknown = ValidatedInvocation::<op::ArgMax>::validate(
            IndexReductionAttributes {
                axis: None,
                dtype: DTypeId::I64,
            },
            vec![LogicalTensorMeta::unknown()],
            vec![output.clone()],
            crate::exec::ProofLevel::Dynamic,
        )
        .expect("an unknown input shape stays unknown");
        assert_eq!(unknown.descriptor().outputs(), &[output]);
    }

    /// Every catalog row that returns a tensor must have a reachable inference
    /// branch, or declare zero outputs. A row reaching neither would be one
    /// whose outputs are only ever caller-asserted.
    #[test]
    fn every_tensor_returning_row_declares_an_inference_source() {
        for row in OPERATION_CATALOG {
            if *row.output_arity.end() == 0 {
                continue;
            }
            // `TypedInference` and `Indexing` are hand-written branches; the
            // rest derive from the declared shape, the input, or a transform.
            let has_source = match row.output {
                OutputRule::TypedInference | OutputRule::Indexing => matches!(
                    row.operation,
                    OperationKind::Gather
                        | OperationKind::Scatter
                        | OperationKind::IndexSelect
                        | OperationKind::EmbeddingExact
                        | OperationKind::Dot
                        | OperationKind::Outer
                        | OperationKind::Addmm
                        | OperationKind::ScaledDotProductAttention
                        | OperationKind::Linear
                        | OperationKind::Quantize
                        | OperationKind::Dequantize
                        | OperationKind::SgdStep
                        | OperationKind::AdamStep
                        | OperationKind::AdamWStep
                ),
                OutputRule::HostValue => false,
                _ => true,
            };
            assert!(
                has_source,
                "{} returns a tensor with no output inference source",
                row.operation,
            );
        }
    }

    #[test]
    fn attribute_bearing_descriptor_round_trips_without_storage() {
        let input = LogicalTensorMeta {
            shape: Some(vec![1, 3, 8, 8]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        let weight = LogicalTensorMeta {
            shape: Some(vec![4, 3, 3, 3]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        let output = LogicalTensorMeta {
            shape: Some(vec![1, 4, 8, 8]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        let invocation = ValidatedInvocation::<op::Conv2dExact>::validate(
            Conv2dAttributes {
                stride: [1, 1],
                padding: [1, 1],
                dilation: [1, 1],
                groups: 1,
                has_bias: false,
            },
            vec![input, weight],
            vec![output],
            crate::exec::ProofLevel::Dynamic,
        )
        .unwrap();
        let json = serde_json::to_string(invocation.descriptor()).unwrap();
        let restored: Descriptor<op::Conv2dExact> = serde_json::from_str(&json).unwrap();
        assert_eq!(&restored, invocation.descriptor());
        assert_eq!(restored.operation(), OperationKind::Conv2dExact);
        assert_eq!(restored.attributes().groups, 1);
        let captured = CapturedDescriptor::capture(invocation.descriptor()).unwrap();
        let decoded = captured.decode::<op::Conv2dExact>().unwrap();
        assert_eq!(decoded, restored);
        assert!(matches!(
            captured.decode::<op::Add>(),
            Err(DescriptorCaptureError::Identity { .. })
        ));
        let mut stale_json = serde_json::to_value(&captured).unwrap();
        stale_json["schema"] = serde_json::json!(0);
        let stale: CapturedDescriptor = serde_json::from_value(stale_json).unwrap();
        assert!(matches!(
            stale.decode::<op::Conv2dExact>(),
            Err(DescriptorCaptureError::Schema { .. })
        ));
    }

    #[test]
    fn typed_attributes_fail_before_storage_access() {
        let input = LogicalTensorMeta {
            shape: Some(vec![2, 3]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        assert!(matches!(
            ValidatedInvocation::<op::Clamp>::validate(
                ClampAttributes { min: 2.0, max: 1.0 },
                vec![input.clone()],
                vec![input.clone()],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "min/max",
                ..
            })
        ));
        assert!(matches!(
            ValidatedInvocation::<op::Softmax>::validate(
                AxisAttributes { axis: 2 },
                vec![input.clone()],
                vec![input],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "axis",
                ..
            })
        ));
        assert!(matches!(
            ValidatedInvocation::<op::AdamStep>::validate(
                AdamAttributes {
                    learning_rate: 1e-3,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    step: 0,
                },
                vec![LogicalTensorMeta::unknown(); 4],
                vec![LogicalTensorMeta::unknown(); 3],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "beta/step",
                ..
            })
        ));
    }

    #[test]
    fn inferred_metadata_cannot_be_fabricated() {
        let lhs = LogicalTensorMeta {
            shape: Some(vec![2, 1]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        let rhs = LogicalTensorMeta {
            shape: Some(vec![1, 3]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        let wrong = LogicalTensorMeta {
            shape: Some(vec![2, 1]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        assert!(matches!(
            ValidatedInvocation::<op::Add>::validate(
                NoAttributes,
                vec![lhs, rhs],
                vec![wrong],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::MetadataMismatch { field: "shape", .. })
        ));

        let created = CreationAttributes {
            shape: vec![2, 3],
            dtype: DTypeId::F32,
            device: DeviceId::cpu(),
        };
        assert!(matches!(
            ValidatedInvocation::<op::Zeros>::validate(
                created,
                vec![],
                vec![LogicalTensorMeta {
                    shape: Some(vec![2, 3]),
                    dtype: Some(DTypeId::I64),
                    device: Some(DeviceId::cpu()),
                }],
                crate::exec::ProofLevel::Static,
            ),
            Err(DescriptorError::MetadataMismatch { field: "dtype", .. })
        ));
    }

    #[test]
    fn multi_output_shapes_and_counts_are_inferred_exactly() {
        let input = LogicalTensorMeta {
            shape: Some(vec![2, 5]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        let topk_output = LogicalTensorMeta {
            shape: Some(vec![2, 3]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        let topk_indices = LogicalTensorMeta {
            dtype: Some(DTypeId::I64),
            ..topk_output.clone()
        };
        ValidatedInvocation::<op::TopK>::validate(
            TopKAttributes {
                k: 3,
                axis: 1,
                largest: true,
                index_dtype: DTypeId::I64,
            },
            vec![input.clone()],
            vec![topk_output, topk_indices],
            crate::exec::ProofLevel::Dynamic,
        )
        .unwrap();

        let output = |extent| LogicalTensorMeta {
            shape: Some(vec![2, extent]),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        ValidatedInvocation::<op::Chunk>::validate(
            ChunkAttributes { chunks: 2, axis: 1 },
            vec![input.clone()],
            vec![output(3), output(2)],
            crate::exec::ProofLevel::Dynamic,
        )
        .unwrap();
        assert!(matches!(
            ValidatedInvocation::<op::Chunk>::validate(
                ChunkAttributes { chunks: 2, axis: 1 },
                vec![input],
                vec![output(3)],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::OutputArity { .. })
        ));
    }

    #[test]
    fn recurrent_and_empty_reduction_contracts_are_storage_free() {
        let tensor = |shape| LogicalTensorMeta {
            shape: Some(shape),
            dtype: Some(DTypeId::F32),
            device: Some(DeviceId::cpu()),
        };
        ValidatedInvocation::<op::Rnn>::validate(
            RecurrentAttributes {
                input_size: 4,
                hidden_size: 6,
                bias_ih: true,
                bias_hh: true,
            },
            vec![tensor(vec![2, 3, 4]), tensor(vec![2, 6])],
            vec![tensor(vec![2, 3, 6]), tensor(vec![2, 6])],
            crate::exec::ProofLevel::Dynamic,
        )
        .unwrap();

        let empty = tensor(vec![0, 4]);
        assert!(matches!(
            ValidatedInvocation::<op::MeanAll>::validate(
                NoAttributes,
                vec![empty.clone()],
                vec![tensor(Vec::new())],
                crate::exec::ProofLevel::Dynamic,
            ),
            Err(DescriptorError::InvalidAttribute {
                attribute: "shape",
                ..
            })
        ));
        ValidatedInvocation::<op::SumAll>::validate(
            NoAttributes,
            vec![empty],
            vec![tensor(Vec::new())],
            crate::exec::ProofLevel::Dynamic,
        )
        .unwrap();
    }
}
