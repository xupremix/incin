//! Safe kernel fusion pass and backward hook integration for compiled graphs.

use alloc::vec::Vec;

use crate::compiled::capture::{CapturedGraph, CapturedNode};
use crate::prelude::OperationKind;
use crate::prelude::Result;

fn builtin_operation(identity: &crate::exec::OperationIdentity) -> Option<OperationKind> {
    match identity {
        crate::exec::OperationIdentity::Builtin(operation) => Some(*operation),
        crate::exec::OperationIdentity::Custom(_) => None,
    }
}

/// Describes why two operations may not be fused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionBlocker {
    /// A graph output value separates the two ops.
    GraphOutput,
    /// The ops have incompatible types for fusion.
    IncompatibleOps,
    /// The consuming op has multiple producers.
    MultipleProducers,
}

/// A candidate pair of adjacent nodes that may be fused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionCandidate {
    /// Index of the producer node in the topological order.
    pub producer_idx: usize,
    /// Index of the consumer node in the topological order.
    pub consumer_idx: usize,
    /// The producer op type.
    pub producer_op: OperationKind,
    /// The consumer op type.
    pub consumer_op: OperationKind,
}

/// A fused kernel: replaces a chain of nodes with a single entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedKernel {
    /// Indices of the original nodes fused into this kernel.
    pub source_node_indices: Vec<usize>,
    /// The leading op type of the fused kernel.
    pub primary_op: OperationKind,
}

/// Safe fusion pass that identifies fusable chains and produces fused kernels.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FusionPass;

impl FusionPass {
    /// Determines whether two adjacent ops are fusable.
    #[must_use]
    fn can_fuse(producer: OperationKind, consumer: OperationKind) -> bool {
        // Only fuse pointwise chains for safety
        use OperationKind::*;
        let is_pointwise = |op| {
            matches!(
                op,
                Add | Sub
                    | Mul
                    | Div
                    | Relu
                    | Gelu
                    | Sigmoid
                    | Tanh
                    | Swish
                    | Neg
                    | Abs
                    | Exp
                    | Sqrt
                    | Log
            )
        };
        is_pointwise(producer) && is_pointwise(consumer)
    }

    /// Identifies fusion candidates in a captured graph.
    pub fn find_candidates(&self, graph: &CapturedGraph) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            if i + 1 >= graph.nodes.len() {
                break;
            }
            let next = &graph.nodes[i + 1];

            // Check that the producer's output is consumed by the next node only
            let producer_outputs = &node.outputs;
            let consumer_inputs = &next.inputs;

            // Output is used as input to the next and not a graph output
            let is_chained = producer_outputs
                .iter()
                .any(|out_id| consumer_inputs.contains(out_id));
            let output_is_graph_output = producer_outputs
                .iter()
                .any(|out_id| graph.outputs.contains(out_id));

            let Some(producer) = builtin_operation(&node.operation) else {
                continue;
            };
            let Some(consumer) = builtin_operation(&next.operation) else {
                continue;
            };
            if is_chained && !output_is_graph_output && Self::can_fuse(producer, consumer) {
                candidates.push(FusionCandidate {
                    producer_idx: i,
                    consumer_idx: i + 1,
                    producer_op: producer,
                    consumer_op: consumer,
                });
            }
        }

        candidates
    }

    /// Applies fusion candidates to produce a new [`CapturedGraph`] with fused kernels.
    pub fn apply(
        &self,
        graph: &CapturedGraph,
        candidates: &[FusionCandidate],
    ) -> Result<(CapturedGraph, Vec<FusedKernel>)> {
        if candidates.is_empty() {
            return Ok((graph.clone(), Vec::new()));
        }

        let fused_pairs: alloc::collections::BTreeSet<usize> =
            candidates.iter().map(|c| c.consumer_idx).collect();

        let mut new_nodes: Vec<CapturedNode> = Vec::new();
        let mut kernels: Vec<FusedKernel> = Vec::new();

        let mut i = 0;
        while i < graph.nodes.len() {
            // Check if this node is a fusion consumer (producer was already merged)
            if fused_pairs.contains(&i) {
                // Already merged, skip because it was incorporated into the previous kernel
                i += 1;
                continue;
            }

            // Check if this node is a fusion producer
            let fusion = candidates.iter().find(|c| c.producer_idx == i);
            if let Some(cand) = fusion {
                let producer = &graph.nodes[cand.producer_idx];
                let consumer = &graph.nodes[cand.consumer_idx];
                // Fused node: takes producer's inputs, produces consumer's outputs
                new_nodes.push(CapturedNode {
                    id: producer.id,
                    operation: producer.operation.clone(),
                    attributes: producer.attributes.clone(),
                    inputs: producer.inputs.clone(),
                    outputs: consumer.outputs.clone(),
                });
                kernels.push(FusedKernel {
                    source_node_indices: alloc::vec![cand.producer_idx, cand.consumer_idx],
                    primary_op: builtin_operation(&producer.operation).ok_or_else(|| {
                        crate::err::Error::Msg(alloc::string::String::from(
                            "custom operations cannot be fused",
                        ))
                    })?,
                });
                i += 1;
                continue;
            }

            new_nodes.push(graph.nodes[i].clone());
            i += 1;
        }

        let fused_graph = CapturedGraph {
            values: graph.values.clone(),
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
            nodes: new_nodes,
        };

        Ok((fused_graph, kernels))
    }
}
