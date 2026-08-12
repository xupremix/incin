use crate::prelude::*;
use alloc::collections::BTreeMap;

/// Identifies a single tensor value (an input, output, or intermediate
/// result) within a `Graph`.
pub type ValueId = usize;
/// Identifies a single operation node within a `Graph`.
pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// The kind of operation a `Node` represents — a backend-agnostic tag
/// recorded by `TracingBackend` and consumed by ONNX export (`as_str`
/// supplies the exported node's `op_type` string) and graph visualization.
pub enum OpType {
    /// An operation outside the built-in graph vocabulary.
    Custom,
    /// Index of the maximum element.
    ArgMax,
    /// Index of the minimum element.
    ArgMin,
    /// Elementwise addition.
    Add,
    /// Elementwise subtraction.
    Sub,
    /// Elementwise multiplication.
    Mul,
    /// Elementwise division.
    Div,
    /// Matrix multiplication.
    MatMul,
    /// Rectified linear unit.
    Relu,
    /// Heaviside step function.
    Step,
    /// Mish activation.
    Mish,
    /// Exponential Linear Unit.
    Elu,
    /// Gaussian Error Linear Unit.
    Gelu,
    /// Elementwise absolute value.
    Abs,
    /// Elementwise natural exponential.
    Exp,
    /// Elementwise negation.
    Neg,
    /// Elementwise square root.
    Sqrt,
    /// Elementwise natural logarithm.
    Log,
    /// Elementwise hyperbolic tangent.
    Tanh,
    /// Elementwise logistic sigmoid.
    Sigmoid,
    /// Swish/SiLU activation.
    Swish,
    /// 1-D convolution.
    Conv1d,
    /// 2-D convolution.
    Conv2d,
    /// Fully-connected (dense) layer.
    Linear,
    /// Shape reinterpretation (also used as a placeholder for
    /// `flatten`/`squeeze`, which have no dedicated variant).
    Reshape,
    /// Dimension permutation.
    Transpose,
    /// Softmax along a dimension.
    Softmax,
    /// Concatenation along an existing dimension.
    Concat,
    /// Stacking along a new dimension.
    Stack,
    /// Scalar addition.
    AddScalar,
    /// Scalar multiplication.
    MulScalar,
    /// Sum-reduce to a scalar.
    SumAll,
    /// Mean-reduce to a scalar.
    MeanAll,
    /// Max-reduce to a scalar.
    MaxAll,
    /// Min-reduce to a scalar.
    MinAll,
    /// Sum-reduce along one dimension.
    SumDim,
    /// Mean-reduce along one dimension.
    MeanDim,
    /// Max-reduce along one dimension.
    MaxDim,
    /// Min-reduce along one dimension.
    MinDim,
    /// NumPy-style broadcast.
    Broadcast,
    /// A contiguous window along one dimension.
    Narrow,
    /// 2-D max pooling.
    MaxPool2d,
    /// 2-D average pooling.
    AvgPool2d,
    /// 2-D adaptive average pooling.
    AdaptiveAvgPool2d,
    /// A strided window per dimension.
    Slice,
    /// Dtype cast.
    ToDtype,
    /// Cross-entropy loss.
    CrossEntropyLoss,
    /// Mean squared error loss.
    MseLoss,
    /// Mean absolute error loss.
    L1Loss,
    /// Binary cross-entropy from logits.
    BceWithLogitsLoss,
    /// Embedding table lookup.
    Embedding,
    /// Layer normalization.
    LayerNorm,
    /// Batch normalization.
    BatchNorm,
    /// Removes a size-1 dimension.
    Squeeze,
    /// Transposed ("deconvolution") 2-D convolution.
    ConvTranspose2d,
    /// A graph input placeholder (no computation).
    Input,
    /// A constant value baked into the graph.
    Constant,
    /// Top-k values and indices along a dimension.
    TopK,
    /// Argsort indices along a dimension.
    Argsort,
    /// Elementwise selection between two tensors driven by a mask.
    WhereCond,
    /// Gathers elements along a dimension using an index tensor of the same rank.
    Gather,
    /// Writes source elements into a copy of the target along a dimension.
    Scatter,
    /// Selects whole slices along a dimension using a 1-D index tensor.
    IndexSelect,
    /// Replaces masked positions with a scalar.
    MaskedFill,
    /// Inserts a size-1 dimension.
    Unsqueeze,
    /// Tiles a tensor a given number of times per dimension.
    Repeat,
    /// Pads each dimension with a constant value.
    Pad,
    /// Upper-triangular part, zeroing below the `k`-th diagonal.
    Triu,
    /// Lower-triangular part, zeroing above the `k`-th diagonal.
    Tril,
    /// Extracts or constructs a diagonal.
    Diag,
    /// Elementwise equality comparison.
    CmpEq,
    /// Elementwise inequality comparison.
    CmpNe,
    /// Elementwise less-than comparison.
    CmpLt,
    /// Elementwise less-than-or-equal comparison.
    CmpLe,
    /// Elementwise greater-than comparison.
    CmpGt,
    /// Elementwise greater-than-or-equal comparison.
    CmpGe,
    /// Elementwise logical conjunction.
    LogicalAnd,
    /// Elementwise logical disjunction.
    LogicalOr,
    /// Elementwise logical negation.
    LogicalNot,
    /// Scalar subtraction.
    SubScalar,
    /// Scalar division.
    DivScalar,
    /// Elementwise maximum of two tensors.
    Maximum,
    /// Elementwise minimum of two tensors.
    Minimum,
    /// Elementwise absolute difference.
    AbsDiff,
    /// Elementwise linear interpolation between two tensors.
    Lerp,
    /// Fused `beta * input + alpha * (lhs @ rhs)`.
    Addmm,
    /// Batched matrix multiplication.
    Bmm,
    /// Scaled dot-product attention.
    ScaledDotProductAttention,
    /// Sliding-window extraction along a dimension.
    Unfold,
    /// Rearranges channel depth into spatial resolution.
    PixelShuffle,
    /// Group normalization.
    GroupNorm,
    /// Instance normalization.
    InstanceNorm,
}

impl OpType {
    /// Renders this op as the string ONNX export uses for the node's
    /// `op_type` (some ops map to the closest matching ONNX standard op
    /// name, e.g. `Linear` -> `"Gemm"`, `Conv1d`/`Conv2d` -> `"Conv"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            OpType::Custom => "Custom",
            OpType::ArgMax => "ArgMax",
            OpType::ArgMin => "ArgMin",
            OpType::Add => "Add",
            OpType::Sub => "Sub",
            OpType::Mul => "Mul",
            OpType::Div => "Div",
            OpType::MatMul => "MatMul",
            OpType::Relu => "Relu",
            OpType::Step => "Step",
            OpType::Mish => "Mish",
            OpType::Elu => "Elu",
            OpType::Gelu => "Gelu",
            OpType::Abs => "Abs",
            OpType::Exp => "Exp",
            OpType::Neg => "Neg",
            OpType::Sqrt => "Sqrt",
            OpType::Log => "Log",
            OpType::Tanh => "Tanh",
            OpType::Sigmoid => "Sigmoid",
            OpType::Swish => "Swish",
            OpType::Conv1d => "Conv",
            OpType::Conv2d => "Conv",
            OpType::Linear => "Gemm",
            OpType::Reshape => "Reshape",
            OpType::Transpose => "Transpose",
            OpType::Softmax => "Softmax",
            OpType::Concat => "Concat",
            OpType::Stack => "Concat",
            OpType::AddScalar => "AddScalar",
            OpType::MulScalar => "MulScalar",
            OpType::SumAll => "SumAll",
            OpType::MeanAll => "MeanAll",
            OpType::MaxAll => "MaxAll",
            OpType::MinAll => "MinAll",
            OpType::SumDim => "SumDim",
            OpType::MeanDim => "MeanDim",
            OpType::MaxDim => "MaxDim",
            OpType::MinDim => "MinDim",
            OpType::Broadcast => "Broadcast",
            OpType::Narrow => "Narrow",
            OpType::MaxPool2d => "MaxPool2d",
            OpType::AvgPool2d => "AvgPool2d",
            OpType::AdaptiveAvgPool2d => "AdaptiveAvgPool2d",
            OpType::Slice => "Slice",
            OpType::ToDtype => "ToDtype",
            OpType::CrossEntropyLoss => "CrossEntropyLoss",
            OpType::MseLoss => "MseLoss",
            OpType::L1Loss => "L1Loss",
            OpType::BceWithLogitsLoss => "BceWithLogitsLoss",
            OpType::Embedding => "Embedding",
            OpType::LayerNorm => "LayerNorm",
            OpType::BatchNorm => "BatchNorm",
            OpType::Squeeze => "Squeeze",
            OpType::ConvTranspose2d => "ConvTranspose2d",
            OpType::Input => "Input",
            OpType::Constant => "Constant",
            OpType::TopK => "TopK",
            OpType::Argsort => "Argsort",
            OpType::WhereCond => "Where",
            // ONNX splits what most frameworks call "gather": `GatherElements`
            // indexes elementwise with a same-rank index tensor, while `Gather`
            // selects whole slices along one axis.
            OpType::Gather => "GatherElements",
            OpType::Scatter => "ScatterElements",
            OpType::IndexSelect => "Gather",
            OpType::MaskedFill => "MaskedFill",
            OpType::Unsqueeze => "Unsqueeze",
            OpType::Repeat => "Tile",
            OpType::Pad => "Pad",
            OpType::Triu => "Trilu",
            OpType::Tril => "Trilu",
            OpType::Diag => "Diag",
            OpType::CmpEq => "Equal",
            // ONNX defines no standard `NotEqual`, so this keeps its own name
            // rather than claiming a standard op that importers would reject.
            OpType::CmpNe => "CmpNe",
            OpType::CmpLt => "Less",
            OpType::CmpLe => "LessOrEqual",
            OpType::CmpGt => "Greater",
            OpType::CmpGe => "GreaterOrEqual",
            OpType::LogicalAnd => "And",
            OpType::LogicalOr => "Or",
            OpType::LogicalNot => "Not",
            OpType::SubScalar => "SubScalar",
            OpType::DivScalar => "DivScalar",
            OpType::Maximum => "Max",
            OpType::Minimum => "Min",
            OpType::AbsDiff => "AbsDiff",
            OpType::Lerp => "Lerp",
            OpType::Addmm => "Gemm",
            OpType::Bmm => "MatMul",
            OpType::ScaledDotProductAttention => "ScaledDotProductAttention",
            OpType::Unfold => "Unfold",
            OpType::PixelShuffle => "DepthToSpace",
            OpType::GroupNorm => "GroupNormalization",
            OpType::InstanceNorm => "InstanceNormalization",
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Metadata for a single tensor value in the graph — shape and dtype, but
/// not the tensor's actual data (see `Graph::initializers` for constants).
pub struct Value {
    /// This value's id.
    pub id: ValueId,
    /// The tensor's shape.
    pub shape: Vec<usize>,
    /// The tensor's element dtype.
    pub dtype: DTypeId,
    /// An optional human-readable name (e.g. for named graph inputs).
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A single operation in the graph: an `OpType` consuming `inputs` and
/// producing `outputs`, with op-specific parameters in `attributes`
/// (e.g. `conv2d`'s stride/padding, `concat`'s axis).
pub struct Node {
    /// This node's id.
    pub id: NodeId,
    /// The operation this node represents.
    pub op: OpType,
    /// Exact canonical execution identity, when the node came from typed dispatch.
    #[serde(default)]
    pub identity: Option<crate::exec::OperationIdentity>,
    /// The value ids this node consumes.
    pub inputs: Vec<ValueId>,
    /// The value ids this node produces.
    pub outputs: Vec<ValueId>,
    /// Op-specific named parameters (e.g. `"axis"`, `"strides"`).
    pub attributes: BTreeMap<String, AttributeValue>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A single named parameter value attached to a `Node` (ONNX-style
/// attribute typing: scalar or list, integer/float/string).
pub enum AttributeValue {
    /// A single integer (e.g. `"axis"`).
    Int(i64),
    /// A single float (e.g. `"epsilon"`).
    Float(f32),
    /// A single string.
    String(String),
    /// A list of integers (e.g. `"strides"`, `"pads"`, `"perm"`).
    Ints(Vec<i64>),
    /// A list of floats.
    Floats(Vec<f32>),
}

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// A serializable, backend-agnostic computation graph IR: the value/node
/// list recorded by `TracingBackend`, consumed by ONNX export and graph
/// visualization. `ValueId`/`NodeId` keys are freshly allocated per graph
/// (via `add_value`/`add_node`), not tied to any particular backend's
/// tensor identity.
pub struct Graph {
    #[serde(with = "string_key_map")]
    /// Every tensor value's metadata, keyed by id.
    pub values: BTreeMap<ValueId, Value>,
    /// Every operation, in insertion order.
    pub nodes: Vec<Node>,
    /// Value ids marked as graph inputs (via `mark_input`).
    pub inputs: Vec<ValueId>,
    /// Value ids marked as graph outputs (via `mark_output`).
    pub outputs: Vec<ValueId>,
    #[serde(with = "string_key_map")]
    /// Raw bytes for constant/weight values, keyed by value id.
    pub initializers: BTreeMap<ValueId, Vec<u8>>, // raw bytes for constants/weights
    next_value_id: usize,
    next_node_id: usize,
}

/// Serde helper: `BTreeMap` only serializes to JSON objects when keys are
/// strings, but `ValueId` is `usize` — this module stringifies keys on
/// serialize and parses them back on deserialize.
mod string_key_map {
    use alloc::collections::BTreeMap;
    use alloc::string::{String, ToString};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Serializes `map` with each key converted to its `ToString` form.
    pub fn serialize<K, V, S>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        K: ToString,
        V: Serialize,
        S: Serializer,
    {
        let string_map: alloc::collections::BTreeMap<String, &V> =
            map.iter().map(|(k, v)| (k.to_string(), v)).collect();
        string_map.serialize(serializer)
    }

    /// Deserializes a string-keyed map and parses each key back via `FromStr`.
    pub fn deserialize<'de, K, V, D>(deserializer: D) -> Result<BTreeMap<K, V>, D::Error>
    where
        K: core::str::FromStr + core::hash::Hash + Eq + core::cmp::Ord,
        K::Err: core::fmt::Display,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let string_map: alloc::collections::BTreeMap<String, V> =
            alloc::collections::BTreeMap::deserialize(deserializer)?;
        string_map
            .into_iter()
            .map(|(k, v)| {
                k.parse::<K>()
                    .map_err(serde::de::Error::custom)
                    .map(|k| (k, v))
            })
            .collect()
    }
}

impl Graph {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new tensor value's metadata, returning its freshly
    /// allocated `ValueId`.
    pub fn add_value(
        &mut self,
        shape: Vec<usize>,
        dtype: DTypeId,
        name: Option<String>,
    ) -> ValueId {
        let id = self.next_value_id;
        self.next_value_id += 1;
        self.values.insert(
            id,
            Value {
                id,
                shape,
                dtype,
                name,
            },
        );
        id
    }

    /// Appends a new operation node, returning its freshly allocated `NodeId`.
    pub fn add_node(
        &mut self,
        op: OpType,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attributes: BTreeMap<String, AttributeValue>,
    ) -> NodeId {
        self.add_node_with_identity(op, inputs, outputs, attributes, None)
    }

    /// Appends a node while retaining the exact execution identity that produced it.
    pub fn add_node_with_identity(
        &mut self,
        op: OpType,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attributes: BTreeMap<String, AttributeValue>,
        identity: Option<crate::exec::OperationIdentity>,
    ) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.push(Node {
            id,
            op,
            identity,
            inputs,
            outputs,
            attributes,
        });
        id
    }

    /// Marks `value_id` as a graph input, if not already marked.
    pub fn mark_input(&mut self, value_id: ValueId) {
        if !self.inputs.contains(&value_id) {
            self.inputs.push(value_id);
        }
    }

    /// Marks `value_id` as a graph output, if not already marked.
    pub fn mark_output(&mut self, value_id: ValueId) {
        if !self.outputs.contains(&value_id) {
            self.outputs.push(value_id);
        }
    }
}
