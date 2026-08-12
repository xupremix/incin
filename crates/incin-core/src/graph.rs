use crate::exec::OperationIdentity;
use crate::prelude::{DTypeId, OperationKind};
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Identifies a tensor value in a graph.
pub type ValueId = usize;
/// Identifies an operation node in a graph.
pub type NodeId = usize;

/// Metadata for one graph value. Concrete shape vectors are retained for
/// eager tracing; compiler-facing symbolic metadata is attached by capture.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Value {
    pub id: ValueId,
    pub shape: Vec<usize>,
    pub dtype: DTypeId,
    pub name: Option<String>,
}

/// A canonical operation node. Built-in identity comes directly from the
/// operation catalog. Custom operations use their namespaced OperationKey.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub operation: OperationIdentity,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub attributes: BTreeMap<String, AttributeValue>,
}

/// A graph attribute value. The canonical typed descriptor remains the source
/// of semantic validation; this is its stable graph serialization form.
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
    pub values: BTreeMap<ValueId, Value>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    #[serde(with = "string_key_map")]
    pub initializers: BTreeMap<ValueId, Vec<u8>>,
    next_value_id: usize,
    next_node_id: usize,
}

mod string_key_map {
    use alloc::collections::BTreeMap;
    use alloc::string::{String, ToString};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<K, V, S>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        K: ToString,
        V: Serialize,
        S: Serializer,
    {
        let string_map: BTreeMap<String, &V> = map
            .iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect();
        string_map.serialize(serializer)
    }

    pub fn deserialize<'de, K, V, D>(deserializer: D) -> Result<BTreeMap<K, V>, D::Error>
    where
        K: core::str::FromStr + core::hash::Hash + Eq + Ord,
        K::Err: core::fmt::Display,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let string_map: BTreeMap<String, V> = BTreeMap::deserialize(deserializer)?;
        string_map
            .into_iter()
            .map(|(key, value)| {
                key.parse::<K>()
                    .map_err(serde::de::Error::custom)
                    .map(|key| (key, value))
            })
            .collect()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn add_node(
        &mut self,
        operation: OperationKind,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attributes: BTreeMap<String, AttributeValue>,
    ) -> NodeId {
        self.add_node_with_identity(
            OperationIdentity::Builtin(operation),
            inputs,
            outputs,
            attributes,
        )
    }

    pub fn add_node_with_identity(
        &mut self,
        operation: OperationIdentity,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
        attributes: BTreeMap<String, AttributeValue>,
    ) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.push(Node {
            id,
            operation,
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
