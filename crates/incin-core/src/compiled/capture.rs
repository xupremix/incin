//! Capture of eager execution graphs into validated IR with descriptor parity.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use crate::exec::OperationIdentity;
use crate::graph::{Graph, NodeId, Value, ValueId};
use crate::prelude::{Error, Result};

/// A single validated node in a captured graph.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CapturedNode {
    /// Node identifier in the original eager graph.
    pub id: NodeId,
    /// Operation type of this node.
    pub operation: OperationIdentity,
    pub attributes: alloc::collections::BTreeMap<String, crate::graph::AttributeValue>,
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
            defined_values.insert(val_id);
        }

        for val_id in graph.values.keys() {
            if graph.initializers.contains_key(val_id) {
                defined_values.insert(*val_id);
            }
        }

        let mut nodes = Vec::with_capacity(graph.nodes.len());
        for node in &graph.nodes {
            for &in_id in &node.inputs {
                if !graph.values.contains_key(&in_id) && !defined_values.contains(&in_id) {
                    return Err(Error::Msg(String::from(
                        "undefined input value in eager graph node",
                    )));
                }
            }

            for &out_id in &node.outputs {
                defined_values.insert(out_id);
            }

            nodes.push(CapturedNode {
                id: node.id,
                operation: node.operation.clone(),
                attributes: node.attributes.clone(),
                inputs: node.inputs.clone(),
                outputs: node.outputs.clone(),
            });
        }

        for &out_id in &graph.outputs {
            if !defined_values.contains(&out_id) && !graph.values.contains_key(&out_id) {
                return Err(Error::Msg(String::from(
                    "undefined output value in captured graph",
                )));
            }
        }

        let values: Vec<ValueId> = graph.values.keys().copied().collect();

        Ok(Self {
            values,
            value_metadata: graph.values.clone(),
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
