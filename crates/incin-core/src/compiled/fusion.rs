//! Inspection-only fusion analysis for compiled graphs.
//!
//! Executable fused lowering is unavailable in the preview CPU evaluator, so
//! applying candidates fails closed.

use alloc::vec::Vec;

use crate::compiled::capture::CapturedGraph;
use crate::err::Result;
use crate::graph::ValueId;
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

    /// Every node index that reads `value` as an input.
    ///
    /// A node that reads the same value twice (`x + x`) counts once: the
    /// question fusion asks is how many *nodes* still need the value to exist,
    /// not how many times one node mentions it. A fused body can refer to its
    /// operand as often as it likes.
    fn consumer_nodes(graph: &CapturedGraph, value: ValueId) -> Vec<usize> {
        graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.inputs.contains(&value))
            .map(|(index, _)| index)
            .collect()
    }

    /// Identifies pointwise pairs where fusing provably cannot lose a value.
    ///
    /// Fusing a producer into its consumer makes the intermediate disappear, so
    /// it is only legal when nothing else needs that intermediate. Three things
    /// have to hold, and the previous heuristic checked only the last:
    ///
    /// 1. The producer has exactly one output. A multi-output node has no
    ///    single intermediate to eliminate.
    /// 2. That output has exactly one consuming node. This is the proof the
    ///    module's doc comment said was missing -- the old scan paired node `i`
    ///    with node `i + 1` by position and never counted readers, so a value
    ///    feeding both node `i + 1` and node `i + 5` was fused anyway and the
    ///    second reader was left referring to something that no longer existed.
    /// 3. The output does not escape as a graph output, since a caller outside
    ///    the graph is a consumer the edges do not show.
    ///
    /// Adjacency is no longer assumed either. The consumer is found by
    /// following the edge, so a producer whose consumer sits further down the
    /// topological order is still a candidate, and two unrelated neighbours no
    /// longer look like one.
    pub fn find_candidates(&self, graph: &CapturedGraph) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();

        for (producer_idx, node) in graph.nodes.iter().enumerate() {
            let Some(producer) = builtin_operation(&node.operation) else {
                continue;
            };
            // One output, or there is no single intermediate to remove.
            let [produced] = node.outputs[..] else {
                continue;
            };
            // Escapes the graph: an external consumer the edges cannot show.
            if graph.outputs.contains(&produced) {
                continue;
            }
            // Exactly one node still needs it.
            let consumers = Self::consumer_nodes(graph, produced);
            let [consumer_idx] = consumers[..] else {
                continue;
            };
            let Some(consumer) = builtin_operation(&graph.nodes[consumer_idx].operation) else {
                continue;
            };
            if !Self::can_fuse(producer, consumer) {
                continue;
            }
            // Nodes are in topological order, so a consumer must follow its
            // producer. A violation means the graph is malformed rather than
            // that this pair is unfusable, and silently fusing it would reorder
            // execution.
            if consumer_idx <= producer_idx {
                continue;
            }
            candidates.push(FusionCandidate {
                producer_idx,
                consumer_idx,
                producer_op: producer,
                consumer_op: consumer,
            });
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
