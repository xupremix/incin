use crate::prelude::*;
use alloc::collections::BTreeMap;

/// Core abstraction for `ValueId` within the Kindle framework..
pub type ValueId = usize;
/// Core abstraction for `NodeId` within the Kindle framework..
pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// Core abstraction for `OpType` within the Kindle framework..
pub enum OpType {
    /// Core abstraction for `ArgMax` within the Kindle framework..
    ArgMax,
    /// Core abstraction for `ArgMin` within the Kindle framework..
    ArgMin,
    /// Core abstraction for `Add` within the Kindle framework..
    Add,
    /// Core abstraction for `Sub` within the Kindle framework..
    Sub,
    /// Core abstraction for `Mul` within the Kindle framework..
    Mul,
    /// Core abstraction for `Div` within the Kindle framework..
    Div,
    /// Core abstraction for `MatMul` within the Kindle framework..
    MatMul,
    /// Core abstraction for `Relu` within the Kindle framework..
    Relu,
    /// Core abstraction for `Step` within the Kindle framework..
    Step,
    /// Core abstraction for `Mish` within the Kindle framework..
    Mish,
    /// Core abstraction for `Elu` within the Kindle framework..
    Elu,
    /// Core abstraction for `Gelu` within the Kindle framework..
    Gelu,
    /// Core abstraction for `Conv1d` within the Kindle framework..
    Conv1d,
    /// Core abstraction for `Conv2d` within the Kindle framework..
    Conv2d,
    /// Core abstraction for `Linear` within the Kindle framework..
    Linear,
    /// Core abstraction for `Reshape` within the Kindle framework..
    Reshape,
    /// Core abstraction for `Transpose` within the Kindle framework..
    Transpose,
    /// Core abstraction for `Softmax` within the Kindle framework..
    Softmax,
    /// Core abstraction for `Concat` within the Kindle framework..
    Concat,
    /// Core abstraction for `Stack` within the Kindle framework..
    Stack,
    /// Core abstraction for `AddScalar` within the Kindle framework..
    AddScalar,
    /// Core abstraction for `MulScalar` within the Kindle framework..
    MulScalar,
    /// Core abstraction for `SumAll` within the Kindle framework..
    SumAll,
    /// Core abstraction for `MeanAll` within the Kindle framework..
    MeanAll,
    /// Core abstraction for `MaxAll` within the Kindle framework..
    MaxAll,
    /// Core abstraction for `MinAll` within the Kindle framework..
    MinAll,
    /// Core abstraction for `SumDim` within the Kindle framework..
    SumDim,
    /// Core abstraction for `MeanDim` within the Kindle framework..
    MeanDim,
    /// Core abstraction for `MaxDim` within the Kindle framework..
    MaxDim,
    /// Core abstraction for `MinDim` within the Kindle framework..
    MinDim,
    /// Core abstraction for `Broadcast` within the Kindle framework..
    Broadcast,
    /// Core abstraction for `Narrow` within the Kindle framework..
    Narrow,
    /// Core abstraction for `MaxPool2d` within the Kindle framework..
    MaxPool2d,
    /// Core abstraction for `AvgPool2d` within the Kindle framework..
    AvgPool2d,
    /// Core abstraction for `AdaptiveAvgPool2d` within the Kindle framework..
    AdaptiveAvgPool2d,
    /// Core abstraction for `Slice` within the Kindle framework..
    Slice,
    /// Core abstraction for `ToDtype` within the Kindle framework..
    ToDtype,
    /// Core abstraction for `CrossEntropyLoss` within the Kindle framework..
    CrossEntropyLoss,
    /// Core abstraction for `MseLoss` within the Kindle framework..
    MseLoss,
    /// Core abstraction for `L1Loss` within the Kindle framework..
    L1Loss,
    /// Core abstraction for `BceWithLogitsLoss` within the Kindle framework..
    BceWithLogitsLoss,
    /// Core abstraction for `Embedding` within the Kindle framework..
    Embedding,
    /// Core abstraction for `LayerNorm` within the Kindle framework..
    LayerNorm,
    /// Core abstraction for `BatchNorm` within the Kindle framework..
    BatchNorm,
    /// Core abstraction for `Squeeze` within the Kindle framework..
    Squeeze,
    /// Core abstraction for `ConvTranspose2d` within the Kindle framework..
    ConvTranspose2d,
    /// Core abstraction for `Input` within the Kindle framework..
    Input,
    /// Core abstraction for `Constant` within the Kindle framework..
    Constant,
}

impl OpType {
    /// Core abstraction for `as_str` within the Kindle framework..
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
/// Core abstraction for `Value` within the Kindle framework..
pub struct Value {
    /// Core abstraction for `id` within the Kindle framework..
    pub id: ValueId,
    /// Core abstraction for `shape` within the Kindle framework..
    pub shape: Vec<usize>,
    /// Core abstraction for `dtype` within the Kindle framework..
    pub dtype: KindleDType,
    /// Core abstraction for `name` within the Kindle framework..
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Core abstraction for `Node` within the Kindle framework..
pub struct Node {
    /// Core abstraction for `id` within the Kindle framework..
    pub id: NodeId,
    /// Core abstraction for `op` within the Kindle framework..
    pub op: OpType,
    /// Core abstraction for `inputs` within the Kindle framework..
    pub inputs: Vec<ValueId>,
    /// Core abstraction for `outputs` within the Kindle framework..
    pub outputs: Vec<ValueId>,
    /// Core abstraction for `attributes` within the Kindle framework..
    pub attributes: BTreeMap<String, AttributeValue>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Core abstraction for `AttributeValue` within the Kindle framework..
pub enum AttributeValue {
    /// Core abstraction for `Int` within the Kindle framework..
    Int(i64),
    /// Core abstraction for `Float` within the Kindle framework..
    Float(f32),
    /// Core abstraction for `String` within the Kindle framework..
    String(String),
    /// Core abstraction for `Ints` within the Kindle framework..
    Ints(Vec<i64>),
    /// Core abstraction for `Floats` within the Kindle framework..
    Floats(Vec<f32>),
}

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Core abstraction for `Graph` within the Kindle framework..
pub struct Graph {
    #[serde(with = "string_key_map")]
    /// Core abstraction for `values` within the Kindle framework..
    pub values: BTreeMap<ValueId, Value>,
    /// Core abstraction for `nodes` within the Kindle framework..
    pub nodes: Vec<Node>,
    /// Core abstraction for `inputs` within the Kindle framework..
    pub inputs: Vec<ValueId>,
    /// Core abstraction for `outputs` within the Kindle framework..
    pub outputs: Vec<ValueId>,
    #[serde(with = "string_key_map")]
    /// Core abstraction for `initializers` within the Kindle framework..
    pub initializers: BTreeMap<ValueId, Vec<u8>>, // raw bytes for constants/weights
    next_value_id: usize,
    next_node_id: usize,
}

/// Core abstraction for `string_key_map` within the Kindle framework..
mod string_key_map {
    use alloc::collections::BTreeMap;
    use alloc::string::{String, ToString};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Core abstraction for `serialize` within the Kindle framework..
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

    /// Core abstraction for `deserialize` within the Kindle framework..
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
    /// Core abstraction for `new` within the Kindle framework..
    pub fn new() -> Self {
        Self::default()
    }

    /// Core abstraction for `add_value` within the Kindle framework..
    pub fn add_value(
        &mut self,
        shape: Vec<usize>,
        dtype: KindleDType,
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

    /// Core abstraction for `add_node` within the Kindle framework..
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

    /// Core abstraction for `mark_input` within the Kindle framework..
    pub fn mark_input(&mut self, value_id: ValueId) {
        if !self.inputs.contains(&value_id) {
            self.inputs.push(value_id);
        }
    }

    /// Core abstraction for `mark_output` within the Kindle framework..
    pub fn mark_output(&mut self, value_id: ValueId) {
        if !self.outputs.contains(&value_id) {
            self.outputs.push(value_id);
        }
    }
}
