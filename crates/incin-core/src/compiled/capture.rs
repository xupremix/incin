//! Capture of eager execution graphs into validated IR with descriptor parity.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use crate::exec::OperationIdentity;
use crate::graph::{AttributeValue, Graph, NodeId, OpType, ValueId};
use crate::prelude::{Error, Result};

/// A single validated node in a captured graph.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapturedNode {
    /// Node identifier in the original eager graph.
    pub id: NodeId,
    /// Operation type of this node.
    pub op: OpType,
    /// Exact execution identity when the node came from typed dispatch.
    #[serde(default)]
    pub identity: Option<OperationIdentity>,
    /// Operation attributes required to reproduce its semantics.
    #[serde(default)]
    pub attributes: alloc::collections::BTreeMap<String, AttributeValue>,
    /// Value IDs consumed by this node as inputs.
    pub inputs: Vec<ValueId>,
    /// Value IDs produced by this node as outputs.
    pub outputs: Vec<ValueId>,
}

/// Metadata for a value in the captured graph.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapturedValue {
    /// Value identifier.
    pub id: ValueId,
    /// Logical shape recorded at capture time.
    pub shape: Vec<usize>,
    /// Element datatype recorded at capture time.
    pub dtype: crate::prelude::DTypeId,
    /// Optional stable display name.
    pub name: Option<String>,
    /// Whether the value is backed by captured constant bytes.
    pub initializer: bool,
}

/// Validated graph IR captured from an eager computation graph.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapturedGraph {
    /// Value IDs defined in the graph.
    pub values: Vec<CapturedValue>,
    /// Input value IDs.
    pub inputs: Vec<ValueId>,
    /// Output value IDs.
    pub outputs: Vec<ValueId>,
    /// Operation nodes in topological execution order.
    pub nodes: Vec<CapturedNode>,
}

impl CapturedGraph {
    /// Captures an eager [`Graph`] into a [`CapturedGraph`], validating input/output value linkage
    /// and topological consistency.
    pub fn capture(graph: &Graph) -> Result<Self> {
        let mut defined_values = BTreeSet::new();

        for &val_id in &graph.inputs {
            if !graph.values.contains_key(&val_id) {
                return Err(Error::Msg(String::from(
                    "graph input refers to an undefined value",
                )));
            }
            defined_values.insert(val_id);
        }

        for val_id in graph.initializers.keys() {
            if !graph.values.contains_key(val_id) {
                return Err(Error::Msg(String::from(
                    "graph initializer refers to an undefined value",
                )));
            }
        }

        for val_id in graph.values.keys() {
            if graph.initializers.contains_key(val_id) {
                defined_values.insert(*val_id);
            }
        }

        let mut nodes = Vec::with_capacity(graph.nodes.len());
        for node in &graph.nodes {
            for &in_id in &node.inputs {
                if !defined_values.contains(&in_id) {
                    return Err(Error::Msg(String::from(
                        "undefined or forward-referenced input value in eager graph node",
                    )));
                }
            }

            for &out_id in &node.outputs {
                if !graph.values.contains_key(&out_id) {
                    return Err(Error::Msg(String::from(
                        "node produces an undefined graph value",
                    )));
                }
                if !defined_values.insert(out_id) {
                    return Err(Error::Msg(String::from(
                        "graph value is produced more than once",
                    )));
                }
            }

            nodes.push(CapturedNode {
                id: node.id,
                op: node.op,
                identity: node.identity.clone(),
                attributes: node.attributes.clone(),
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
            });
        }

        for &out_id in &graph.outputs {
            if !defined_values.contains(&out_id) {
                return Err(Error::Msg(String::from(
                    "undefined output value in captured graph",
                )));
            }
        }

        let values = graph
            .values
            .values()
            .map(|value| CapturedValue {
                id: value.id,
                shape: value.shape.clone(),
                dtype: value.dtype,
                name: value.name.clone(),
                initializer: graph.initializers.contains_key(&value.id),
            })
            .collect();

        Ok(Self {
            values,
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
            nodes,
        })
    }

    /// Number of nodes in the captured graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of values defined in the captured graph.
    #[must_use]
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Returns captured metadata for a value.
    #[must_use]
    pub fn value(&self, id: ValueId) -> Option<&CapturedValue> {
        self.values.iter().find(|value| value.id == id)
    }
}
