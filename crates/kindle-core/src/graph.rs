use crate::prelude::*;
use std::collections::HashMap;

pub type ValueId = usize;
pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
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

#[derive(Debug, Clone)]
pub struct Value {
    pub id: ValueId,
    pub shape: Vec<usize>,
    pub dtype: KindleDType,
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: OpType,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub attributes: HashMap<String, AttributeValue>,
}

#[derive(Debug, Clone)]
pub enum AttributeValue {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

#[derive(Debug, Default)]
pub struct Graph {
    pub values: HashMap<ValueId, Value>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub initializers: HashMap<ValueId, Vec<u8>>, // raw bytes for constants/weights
    next_value_id: usize,
    next_node_id: usize,
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
