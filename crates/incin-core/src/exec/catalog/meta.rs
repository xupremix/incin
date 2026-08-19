use super::*;

/// Logical metadata used before a backend storage handle exists.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LogicalTensorMeta {
    pub shape: Option<ShapeBuf>,
    pub dtype: Option<DTypeDescriptor>,
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
    DataAttributes { shape: Vec<usize>, dtype: DTypeDescriptor, device: DeviceId, payload: CreationPayload }
    CreationAttributes { shape: Vec<usize>, dtype: DTypeDescriptor, device: DeviceId }
    FullAttributes { shape: Vec<usize>, dtype: DTypeDescriptor, device: DeviceId, value: f64 }
    ArangeAttributes { shape: Vec<usize>, dtype: DTypeDescriptor, device: DeviceId, start: f64, step: f64 }
    LinspaceAttributes { shape: Vec<usize>, dtype: DTypeDescriptor, device: DeviceId, start: f64, end: f64 }
    DistributionAttributes { shape: Vec<usize>, dtype: DTypeDescriptor, device: DeviceId, distribution: alloc::string::String, parameters: Vec<u8> }
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
    DTypeAttributes { dtype: DTypeDescriptor }
    DeviceAttributes { device: DeviceId }
    IndexReductionAttributes { axis: Option<usize>, dtype: DTypeDescriptor }
    TopKAttributes { k: usize, axis: usize, largest: bool, index_dtype: DTypeDescriptor }
    ArgsortAttributes { axis: usize, descending: bool, index_dtype: DTypeDescriptor }
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
    QuantizationAttributes { dtype: DTypeDescriptor }
    SgdAttributes { learning_rate: f64 }
    AdamAttributes { learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, step: usize }
    AdamWAttributes { learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64, step: usize }
}

/// Small payload metadata used by the data-creation descriptors.
///
/// `TensorFromData` records the dtype of its native source values, while
/// `TensorFromBytes` records that its bytes were supplied without a native
/// scalar type. The bytes themselves travel separately as a borrowed execution
/// payload and never become part of the semantic descriptor.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CreationPayload {
    Typed {
        byte_len: usize,
        dtype: DTypeDescriptor,
    },
    Bytes {
        byte_len: usize,
    },
}

impl CreationPayload {
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        match self {
            Self::Typed { byte_len, .. } | Self::Bytes { byte_len } => *byte_len,
        }
    }
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

/// Stable identity for an operation outside the built-in catalog.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct OperationKey {
    pub namespace: Cow<'static, str>,
    pub name: Cow<'static, str>,
    pub version: u32,
}
