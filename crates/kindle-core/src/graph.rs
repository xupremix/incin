use crate::prelude::*;
use alloc::collections::BTreeMap;

/// Auto-generated documentation for ValueId.
pub type ValueId = usize;
/// Auto-generated documentation for NodeId.
pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
/// Auto-generated documentation for OpType.
pub enum OpType {
    /// Auto-generated documentation for ArgMax.
    ArgMax,
    /// Auto-generated documentation for ArgMin.
    ArgMin,
    /// Auto-generated documentation for Add.
    Add,
    /// Auto-generated documentation for Sub.
    Sub,
    /// Auto-generated documentation for Mul.
    Mul,
    /// Auto-generated documentation for Div.
    Div,
    /// Auto-generated documentation for MatMul.
    MatMul,
    /// Auto-generated documentation for Relu.
    Relu,
    /// Auto-generated documentation for Step.
    Step,
    /// Auto-generated documentation for Mish.
    Mish,
    /// Auto-generated documentation for Elu.
    Elu,
    /// Auto-generated documentation for Gelu.
    Gelu,
    /// Auto-generated documentation for Conv1d.
    Conv1d,
    /// Auto-generated documentation for Conv2d.
    Conv2d,
    /// Auto-generated documentation for Linear.
    Linear,
    /// Auto-generated documentation for Reshape.
    Reshape,
    /// Auto-generated documentation for Transpose.
    Transpose,
    /// Auto-generated documentation for Softmax.
    Softmax,
    /// Auto-generated documentation for Concat.
    Concat,
    /// Auto-generated documentation for Stack.
    Stack,
    /// Auto-generated documentation for AddScalar.
    AddScalar,
    /// Auto-generated documentation for MulScalar.
    MulScalar,
    /// Auto-generated documentation for SumAll.
    SumAll,
    /// Auto-generated documentation for MeanAll.
    MeanAll,
    /// Auto-generated documentation for MaxAll.
    MaxAll,
    /// Auto-generated documentation for MinAll.
    MinAll,
    /// Auto-generated documentation for SumDim.
    SumDim,
    /// Auto-generated documentation for MeanDim.
    MeanDim,
    /// Auto-generated documentation for MaxDim.
    MaxDim,
    /// Auto-generated documentation for MinDim.
    MinDim,
    /// Auto-generated documentation for Broadcast.
    Broadcast,
    /// Auto-generated documentation for Narrow.
    Narrow,
    /// Auto-generated documentation for MaxPool2d.
    MaxPool2d,
    /// Auto-generated documentation for AvgPool2d.
    AvgPool2d,
    /// Auto-generated documentation for AdaptiveAvgPool2d.
    AdaptiveAvgPool2d,
    /// Auto-generated documentation for Slice.
    Slice,
    /// Auto-generated documentation for ToDtype.
    ToDtype,
    /// Auto-generated documentation for CrossEntropyLoss.
    CrossEntropyLoss,
    /// Auto-generated documentation for MseLoss.
    MseLoss,
    /// Auto-generated documentation for L1Loss.
    L1Loss,
    /// Auto-generated documentation for BceWithLogitsLoss.
    BceWithLogitsLoss,
    /// Auto-generated documentation for Embedding.
    Embedding,
    /// Auto-generated documentation for LayerNorm.
    LayerNorm,
    /// Auto-generated documentation for BatchNorm.
    BatchNorm,
    /// Auto-generated documentation for Squeeze.
    Squeeze,
    /// Auto-generated documentation for ConvTranspose2d.
    ConvTranspose2d,
    /// Auto-generated documentation for Input.
    Input,
    /// Auto-generated documentation for Constant.
    Constant,
}

impl OpType {
    /// Auto-generated documentation for as_str.
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
/// Auto-generated documentation for Value.
pub struct Value {
    /// Auto-generated documentation for id.
    pub id: ValueId,
    /// Auto-generated documentation for shape.
    pub shape: Vec<usize>,
    /// Auto-generated documentation for dtype.
    pub dtype: KindleDType,
    /// Auto-generated documentation for name.
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Auto-generated documentation for Node.
pub struct Node {
    /// Auto-generated documentation for id.
    pub id: NodeId,
    /// Auto-generated documentation for op.
    pub op: OpType,
    /// Auto-generated documentation for inputs.
    pub inputs: Vec<ValueId>,
    /// Auto-generated documentation for outputs.
    pub outputs: Vec<ValueId>,
    /// Auto-generated documentation for attributes.
    pub attributes: BTreeMap<String, AttributeValue>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Auto-generated documentation for AttributeValue.
pub enum AttributeValue {
    /// Auto-generated documentation for Int.
    Int(i64),
    /// Auto-generated documentation for Float.
    Float(f32),
    /// Auto-generated documentation for String.
    String(String),
    /// Auto-generated documentation for Ints.
    Ints(Vec<i64>),
    /// Auto-generated documentation for Floats.
    Floats(Vec<f32>),
}

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
/// Auto-generated documentation for Graph.
pub struct Graph {
    #[serde(with = "string_key_map")]
    /// Auto-generated documentation for values.
    pub values: BTreeMap<ValueId, Value>,
    /// Auto-generated documentation for nodes.
    pub nodes: Vec<Node>,
    /// Auto-generated documentation for inputs.
    pub inputs: Vec<ValueId>,
    /// Auto-generated documentation for outputs.
    pub outputs: Vec<ValueId>,
    #[serde(with = "string_key_map")]
    /// Auto-generated documentation for initializers.
    pub initializers: BTreeMap<ValueId, Vec<u8>>, // raw bytes for constants/weights
    next_value_id: usize,
    next_node_id: usize,
}

/// Auto-generated documentation for string_key_map.
mod string_key_map {
    use alloc::collections::BTreeMap;
    use alloc::string::{String, ToString};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Auto-generated documentation for serialize.
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

    /// Auto-generated documentation for deserialize.
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
    /// Auto-generated documentation for new.
    pub fn new() -> Self {
        Self::default()
    }

    /// Auto-generated documentation for add_value.
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

    /// Auto-generated documentation for add_node.
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

    /// Auto-generated documentation for mark_input.
    pub fn mark_input(&mut self, value_id: ValueId) {
        if !self.inputs.contains(&value_id) {
            self.inputs.push(value_id);
        }
    }

    /// Auto-generated documentation for mark_output.
    pub fn mark_output(&mut self, value_id: ValueId) {
        if !self.outputs.contains(&value_id) {
            self.outputs.push(value_id);
        }
    }
}
