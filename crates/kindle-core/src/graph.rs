use crate::prelude::*;
use hashbrown::HashMap;

pub type ValueId = usize;
pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OpType {
    ArgMax,
    ArgMin,
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Relu,
    Gelu,
    Conv1d,
    Conv2d,
    Linear,
    Reshape,
    Transpose,
    Softmax,
    Concat,
    Stack,
    AddScalar,
    MulScalar,
    SumAll,
    MeanAll,
    MaxAll,
    MinAll,
    SumDim,
    MeanDim,
    MaxDim,
    MinDim,
    Broadcast,
    Narrow,
    MaxPool2d,
    AvgPool2d,
    AdaptiveAvgPool2d,
    Slice,
    ToDtype,
    CrossEntropyLoss,
    MseLoss,
    L1Loss,
    BceWithLogitsLoss,
    Embedding,
    LayerNorm,
    BatchNorm,
    Squeeze,
    ConvTranspose2d,
    Input,
    Constant,
}

impl OpType {
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
pub struct Value {
    pub id: ValueId,
    pub shape: Vec<usize>,
    pub dtype: KindleDType,
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op: OpType,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub attributes: HashMap<String, AttributeValue>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AttributeValue {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Graph {
    #[serde(with = "string_key_map")]
    pub values: HashMap<ValueId, Value>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    #[serde(with = "string_key_map")]
    pub initializers: HashMap<ValueId, Vec<u8>>, // raw bytes for constants/weights
    next_value_id: usize,
    next_node_id: usize,
}

mod string_key_map {
    use hashbrown::HashMap;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<K, V, S>(map: &HashMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        K: ToString,
        V: Serialize,
        S: Serializer,
    {
        let string_map: std::collections::HashMap<String, &V> = map.iter().map(|(k, v)| (k.to_string(), v)).collect();
        string_map.serialize(serializer)
    }

    pub fn deserialize<'de, K, V, D>(deserializer: D) -> Result<HashMap<K, V>, D::Error>
    where
        K: std::str::FromStr + std::hash::Hash + Eq,
        K::Err: std::fmt::Display,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let string_map: std::collections::HashMap<String, V> = std::collections::HashMap::deserialize(deserializer)?;
        string_map.into_iter().map(|(k, v)| {
            k.parse::<K>().map_err(serde::de::Error::custom).map(|k| (k, v))
        }).collect()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn add_node(
        &mut self,
        op: OpType,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attributes: HashMap<String, AttributeValue>,
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

    pub fn mark_input(&mut self, value_id: ValueId) {
        if !self.inputs.contains(&value_id) {
            self.inputs.push(value_id);
        }
    }

    pub fn mark_output(&mut self, value_id: ValueId) {
        if !self.outputs.contains(&value_id) {
            self.outputs.push(value_id);
        }
    }
}
