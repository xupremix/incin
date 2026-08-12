//! Safe kernel fusion pass and backward hook integration for compiled graphs.

use alloc::vec::Vec;

use crate::compiled::capture::CapturedGraph;
use crate::graph::OpType;
use crate::prelude::Result;

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
    pub producer_op: OpType,
    /// The consumer op type.
    pub consumer_op: OpType,
}

/// A fused kernel: replaces a chain of nodes with a single entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedKernel {
    /// Indices of the original nodes fused into this kernel.
    pub source_node_indices: Vec<usize>,
    /// The leading op type of the fused kernel.
    pub primary_op: OpType,
}

/// Safe fusion pass that identifies fusable chains and produces fused kernels.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FusionPass;

impl FusionPass {
    /// Determines whether two adjacent ops are fusable.
    #[must_use]
    fn can_fuse(producer: OpType, consumer: OpType) -> bool {
        // Only fuse pointwise chains for safety
        use OpType::*;
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
    ///
    /// Fusion is disabled until the compiled IR can represent the complete
    /// ordered operation sequence and all consumer inputs.
    pub fn find_candidates(&self, graph: &CapturedGraph) -> Vec<FusionCandidate> {
        let _ = graph;
        Vec::new()
    }

    /// Applies fusion candidates to produce a new [`CapturedGraph`] with fused kernels.
    pub fn apply(
        &self,
        graph: &CapturedGraph,
        candidates: &[FusionCandidate],
    ) -> Result<(CapturedGraph, Vec<FusedKernel>)> {
        let _ = candidates;
        graph.validate()?;
        Ok((graph.clone(), Vec::new()))
    }
}
