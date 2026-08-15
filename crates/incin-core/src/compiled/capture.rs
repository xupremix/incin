//! Capture of eager execution graphs into validated IR with descriptor parity.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use crate::exec::{ExecutionSite, OperationIdentity};
use crate::graph::{Graph, NodeId, Value, ValueId};
use crate::err::{Error, Result};

/// A single validated node in a captured graph.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapturedNode {
    /// Node identifier in the original eager graph.
    pub id: NodeId,
    /// Operation type of this node.
    pub operation: OperationIdentity,
    pub execution_site: Option<ExecutionSite>,
    pub attributes: alloc::collections::BTreeMap<String, crate::graph::AttributeValue>,
    pub descriptor_payload: Option<crate::graph::DescriptorPayload>,
    /// Value IDs consumed by this node as inputs.
    pub inputs: Vec<ValueId>,
    /// Value IDs produced by this node as outputs.
    pub outputs: Vec<ValueId>,
}

/// Validated graph IR captured from an eager computation graph.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapturedGraph {
    /// Value IDs defined in the graph.
    pub values: Vec<ValueId>,
    /// Captured value metadata used by compilation and guards.
    pub value_metadata: alloc::collections::BTreeMap<ValueId, Value>,
    /// Serialized bytes for graph-owned constant values.
    #[serde(default)]
    pub initializers: alloc::collections::BTreeMap<ValueId, Vec<u8>>,
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
                    "graph input is missing value metadata",
                )));
            }
            if graph.initializers.contains_key(&val_id) {
                return Err(Error::Msg(String::from(
                    "graph input cannot also be an initializer",
                )));
            }
            if !defined_values.insert(val_id) {
                return Err(Error::Msg(String::from("duplicate graph input value")));
            }
        }

        for val_id in graph.initializers.keys() {
            if !graph.values.contains_key(val_id) {
                return Err(Error::Msg(String::from(
                    "initializer is missing value metadata",
                )));
            }
            if !defined_values.insert(*val_id) {
                return Err(Error::Msg(String::from(
                    "initializer overlaps a graph input",
                )));
            }
        }

        let mut nodes = Vec::with_capacity(graph.nodes.len());
        for (expected_id, node) in graph.nodes.iter().enumerate() {
            if node.id != expected_id {
                return Err(Error::Msg(String::from(
                    "graph nodes are not in canonical topological order",
                )));
            }
            for &in_id in &node.inputs {
                if !defined_values.contains(&in_id) {
                    return Err(Error::Msg(String::from(
                        "undefined input value in eager graph node",
                    )));
                }
            }

            for &out_id in &node.outputs {
                if !graph.values.contains_key(&out_id) {
                    return Err(Error::Msg(String::from(
                        "graph node output is missing value metadata",
                    )));
                }
                if !defined_values.insert(out_id) {
                    return Err(Error::Msg(String::from(
                        "graph node defines a value more than once",
                    )));
                }
            }

            nodes.push(CapturedNode {
                id: node.id,
                operation: node.operation.clone(),
                execution_site: node.execution_site,
                attributes: node.attributes.clone(),
                descriptor_payload: node.descriptor_payload.clone(),
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

        let values: Vec<ValueId> = graph.values.keys().copied().collect();

        Ok(Self {
            values,
            value_metadata: graph.values.clone(),
            initializers: graph.initializers.clone(),
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
}
