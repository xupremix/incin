use crate::prelude::*;
use alloc::collections::BTreeMap;

/// `ValueId`.
pub type ValueId = usize;
/// `NodeId`.
pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// `OpType`.
pub enum OpType {
    /// `ArgMax`.
    ArgMax,
    /// `ArgMin`.
    ArgMin,
    /// `Add`.
    Add,
    /// `Sub`.
    Sub,
    /// `Mul`.
    Mul,
    /// `Div`.
    Div,
    /// `MatMul`.
    MatMul,
    /// `Relu`.
    Relu,
    /// `Step`.
    Step,
    /// `Mish`.
    Mish,
    /// `Elu`.
    Elu,
    /// `Gelu`.
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
    /// `Conv1d`.
    Conv1d,
    /// `Conv2d`.
    Conv2d,
    /// `Linear`.
    Linear,
    /// `Reshape`.
    Reshape,
    /// `Transpose`.
    Transpose,
    /// `Softmax`.
    Softmax,
    /// `Concat`.
    Concat,
    /// `Stack`.
    Stack,
    /// `AddScalar`.
    AddScalar,
    /// `MulScalar`.
    MulScalar,
    /// `SumAll`.
    SumAll,
    /// `MeanAll`.
    MeanAll,
    /// `MaxAll`.
    MaxAll,
    /// `MinAll`.
    MinAll,
    /// `SumDim`.
    SumDim,
    /// `MeanDim`.
    MeanDim,
    /// `MaxDim`.
    MaxDim,
    /// `MinDim`.
    MinDim,
    /// `Broadcast`.
    Broadcast,
    /// `Narrow`.
    Narrow,
    /// `MaxPool2d`.
    MaxPool2d,
    /// `AvgPool2d`.
    AvgPool2d,
    /// `AdaptiveAvgPool2d`.
    AdaptiveAvgPool2d,
    /// `Slice`.
    Slice,
    /// `ToDtype`.
    ToDtype,
    /// `CrossEntropyLoss`.
    CrossEntropyLoss,
    /// `MseLoss`.
    MseLoss,
    /// `L1Loss`.
    L1Loss,
    /// `BceWithLogitsLoss`.
    BceWithLogitsLoss,
    /// `Embedding`.
    Embedding,
    /// `LayerNorm`.
    LayerNorm,
    /// `BatchNorm`.
    BatchNorm,
    /// `Squeeze`.
    Squeeze,
    /// `ConvTranspose2d`.
    ConvTranspose2d,
    /// `Input`.
    Input,
    /// `Constant`.
    Constant,
}

impl OpType {
    /// `as_str`.
    pub fn as_str(&self) -> &'static str {
        match self {
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// `Value`.
pub struct Value {
    /// `id`.
    pub id: ValueId,
    /// `shape`.
    pub shape: Vec<usize>,
    /// `dtype`.
    pub dtype: DTypeId,
    /// The display name of this layer node.
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// `Node`.
pub struct Node {
    /// `id`.
    pub id: NodeId,
    /// `op`.
    pub op: OpType,
    /// `inputs`.
    pub inputs: Vec<ValueId>,
    /// `outputs`.
    pub outputs: Vec<ValueId>,
    /// `attributes`.
    pub attributes: BTreeMap<String, AttributeValue>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// `AttributeValue`.
pub enum AttributeValue {
    /// `Int`.
    Int(i64),
    /// `Float`.
    Float(f32),
    /// `String`.
    String(String),
    /// `Ints`.
    Ints(Vec<i64>),
    /// `Floats`.
    Floats(Vec<f32>),
}

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// `Graph`.
pub struct Graph {
    #[serde(with = "string_key_map")]
    /// `values`.
    pub values: BTreeMap<ValueId, Value>,
    /// `nodes`.
    pub nodes: Vec<Node>,
    /// `inputs`.
    pub inputs: Vec<ValueId>,
    /// `outputs`.
    pub outputs: Vec<ValueId>,
    #[serde(with = "string_key_map")]
    /// `initializers`.
    pub initializers: BTreeMap<ValueId, Vec<u8>>, // raw bytes for constants/weights
    next_value_id: usize,
    next_node_id: usize,
}

/// `string_key_map`.
mod string_key_map {
    use alloc::collections::BTreeMap;
    use alloc::string::{String, ToString};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// `serialize`.
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

    /// `deserialize`.
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

    /// `add_value`.
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

    /// `add_node`.
    pub fn add_node(
        &mut self,
        op: OpType,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attributes: BTreeMap<String, AttributeValue>,
    ) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.push(Node {
            id,
            op,
            inputs,
            outputs,
            attributes,
        });
        id
    }

    /// `mark_input`.
    pub fn mark_input(&mut self, value_id: ValueId) {
        if !self.inputs.contains(&value_id) {
            self.inputs.push(value_id);
        }
    }

    /// `mark_output`.
    pub fn mark_output(&mut self, value_id: ValueId) {
        if !self.outputs.contains(&value_id) {
            self.outputs.push(value_id);
        }
    }
}
