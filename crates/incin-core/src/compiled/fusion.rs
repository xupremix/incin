//! Inspection-only fusion analysis for compiled graphs.
//!
//! Executable fused lowering is unavailable in the preview CPU evaluator, so
//! applying candidates fails closed.

use alloc::vec::Vec;

use crate::compiled::capture::CapturedGraph;
use crate::err::Result;
use crate::shapes::error::OperationKind;

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

/// Fusion analysis that identifies candidate chains; applying them fails closed.
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

    /// Identifies adjacent pointwise heuristic candidates in a captured graph.
    ///
    /// This does not prove that an intermediate value has no other consumers;
    /// no executable fusion lowering is currently available.
    pub fn find_candidates(&self, graph: &CapturedGraph) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            if i + 1 >= graph.nodes.len() {
                break;
            }
            let next = &graph.nodes[i + 1];

            // Check only that an adjacent node consumes a producer output. This
            // is a heuristic candidate, not an exclusive-consumer proof.
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

    /// Returns the unchanged graph when there are no candidates.
    ///
    /// Non-empty candidate sets fail closed until executable fused-descriptor lowering exists.
    pub fn apply(
        &self,
        graph: &CapturedGraph,
        candidates: &[FusionCandidate],
    ) -> Result<(CapturedGraph, Vec<FusedKernel>)> {
        if candidates.is_empty() {
            return Ok((graph.clone(), Vec::new()));
        }

        let _ = (graph, candidates);
        Err(crate::err::Error::Msg(
            "compiled fusion has no executable fused descriptor lowering".into(),
        ))
    }
}
