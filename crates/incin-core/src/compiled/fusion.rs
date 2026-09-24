//! Proven fusion analysis and admission for compiled graphs.
//!
//! [`FusionPass`] discovers candidate pointwise pairs, proves each chain's
//! intermediates exclusively consumed (tape, topological order, one geometry,
//! float dtype, template boundary budget), and admits the proven groups while
//! refusing everything else by name. Emitting an admitted group as one
//! executable kernel is a backend's job behind [`FusedKernelLowering`]; core
//! never promises a lowering it cannot name.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::compiled::alloc::SavedTensorSet;
use crate::compiled::capture::CapturedGraph;
use crate::err::Result;
use crate::graph::ValueId;
use crate::shapes::error::OperationKind;
use crate::tensor::dtype::{DTypeDescriptor, DTypeId};

fn builtin_operation(identity: &crate::exec::OperationIdentity) -> Option<OperationKind> {
    match identity {
        crate::exec::OperationIdentity::Builtin(operation) => Some(*operation),
        crate::exec::OperationIdentity::Custom(_) => None,
    }
}

/// The fusion class an operation belongs to, after TVM `FuseOps`-style
/// classification inside a dataflow region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionPatternKind {
    /// Element-wise pointwise work that the scalar templates can lower.
    ElemWise,
    /// Anything else; never fused by this pass.
    Opaque,
}

/// Classifies `op` for fusion. The allow-list is the CMP-005 pointwise set:
/// only these operations may appear in a proven chain.
#[must_use]
pub const fn fusion_pattern_kind(op: OperationKind) -> FusionPatternKind {
    use OperationKind::*;
    match op {
        Add | Sub | Mul | Div | Relu | Gelu | Sigmoid | Tanh | Swish | Neg | Abs | Exp | Sqrt
        | Log => FusionPatternKind::ElemWise,
        _ => FusionPatternKind::Opaque,
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
    /// An operation is not in the element-wise fusion class.
    NotElementwise,
    /// The intermediate value is not proven exclusively consumed by this pair.
    NotProvenExclusive,
    /// The intermediate is saved for backward and must survive the forward pass.
    SavedForBackward,
    /// The chain spans values with differing shapes or dtypes.
    ShapeMismatch,
    /// The consumer does not follow its producer in topological order.
    OutOfTopologicalOrder,
    /// The chain is too short to be worth emitting as one kernel.
    InsufficientBenefit,
    /// The chain's values are not a builtin float the scalar templates compute.
    UnsupportedDtype,
    /// The chain does not have the one or two boundary operands the pointwise
    /// templates hold.
    BoundaryCount,
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

/// Why one candidate edge or one assembled chain was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupRefusal {
    /// A single producer/consumer edge failed a gate.
    Link {
        /// Index of the producer node.
        producer_idx: usize,
        /// Index of the consumer node.
        consumer_idx: usize,
        /// The gate that refused the edge.
        blocker: FusionBlocker,
    },
    /// A multi-node chain failed a group-level gate.
    Chain {
        /// The node indices that were considered.
        node_indices: Vec<usize>,
        /// The gate that refused the chain.
        blocker: FusionBlocker,
    },
}

/// The result of planning provable fusion groups from candidate edges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedGroups {
    /// Vertex-disjoint chains safe to lower as one kernel.
    pub groups: Vec<FusedKernel>,
    /// Every edge or chain that was considered and refused, with the reason.
    pub refused: Vec<GroupRefusal>,
}

/// A backend's contract for emitting one proven fusion group as one kernel.
///
/// [`FusionPass::apply`] admits only groups that are exclusive, absent from
/// the backward tape, topologically ordered, geometrically uniform, float
/// typed, and within the two-operand boundary budget — but an admitted group
/// is still just node indices in a [`CapturedGraph`]. Turning it into a
/// single executable unit is backend representation work, and this trait is
/// where core hands that job over.
///
/// Implementations must re-validate rather than trust, exactly as
/// [`FusionPass::plan_groups`] re-runs [`FusionPass::check_link`] on
/// hand-fed candidates: a group that bypassed admission, or a saved set that
/// changed since, has to be refused, never lowered.
pub trait FusedKernelLowering {
    /// The single executable unit this backend emits for a group.
    type Kernel;

    /// Lowers `group` into one kernel covering exactly its node chain.
    ///
    /// # Errors
    ///
    /// Returns the first gate that refuses the group. A failed call never
    /// yields a partial kernel.
    fn lower_group(
        &self,
        graph: &CapturedGraph,
        group: &FusedKernel,
        saved: &SavedTensorSet,
    ) -> Result<Self::Kernel>;
}

/// Fusion analysis that proves pointwise chains exclusive, then admits the
/// proven groups or refuses the rest by name.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FusionPass;

impl FusionPass {
    /// Determines whether two adjacent ops are fusable.
    #[must_use]
    fn can_fuse(producer: OperationKind, consumer: OperationKind) -> bool {
        matches!(fusion_pattern_kind(producer), FusionPatternKind::ElemWise)
            && matches!(fusion_pattern_kind(consumer), FusionPatternKind::ElemWise)
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

    /// Re-validates one producer/consumer edge against every fusion gate.
    ///
    /// This is the single source of truth for link legality: `find_candidates`
    /// discovers candidate edges, and `plan_groups` re-runs this check before
    /// accepting any of them, so a hand-fed pair cannot bypass a gate.
    ///
    /// The gates, in order:
    ///
    /// 1. Both nodes exist and carry builtin identities.
    /// 2. Both ops classify as [`FusionPatternKind::ElemWise`].
    /// 3. The producer has exactly one output — a multi-output node has no
    ///    single intermediate to eliminate.
    /// 4. That output does not escape as a graph output (an external consumer
    ///    the edge list cannot show).
    /// 5. The output has exactly one consuming node, and it is `consumer_idx`.
    ///    This is the exclusivity proof: fusing the producer away is only safe
    ///    when nothing else still needs the intermediate.
    /// 6. The consumer sits strictly after the producer in topological order;
    ///    a violation means the graph is malformed rather than unfusable.
    ///
    /// # Errors
    ///
    /// Returns the [`FusionBlocker`] for the first gate the pair fails.
    pub fn check_link(
        &self,
        graph: &CapturedGraph,
        producer_idx: usize,
        consumer_idx: usize,
    ) -> core::result::Result<(), FusionBlocker> {
        let producer_node = graph
            .nodes
            .get(producer_idx)
            .ok_or(FusionBlocker::OutOfTopologicalOrder)?;
        let consumer_node = graph
            .nodes
            .get(consumer_idx)
            .ok_or(FusionBlocker::OutOfTopologicalOrder)?;
        let producer =
            builtin_operation(&producer_node.operation).ok_or(FusionBlocker::IncompatibleOps)?;
        let consumer =
            builtin_operation(&consumer_node.operation).ok_or(FusionBlocker::IncompatibleOps)?;
        if !Self::can_fuse(producer, consumer) {
            return Err(FusionBlocker::NotElementwise);
        }
        let [produced] = producer_node.outputs[..] else {
            return Err(FusionBlocker::NotProvenExclusive);
        };
        if graph.outputs.contains(&produced) {
            return Err(FusionBlocker::GraphOutput);
        }
        let consumers = Self::consumer_nodes(graph, produced);
        if consumers.len() > 1 {
            return Err(FusionBlocker::MultipleProducers);
        }
        let [found] = consumers[..] else {
            return Err(FusionBlocker::NotProvenExclusive);
        };
        if found != consumer_idx {
            return Err(FusionBlocker::NotProvenExclusive);
        }
        if consumer_idx <= producer_idx {
            return Err(FusionBlocker::OutOfTopologicalOrder);
        }
        Ok(())
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
    ///
    /// Every structural prefilter here exists only to *discover* the consumer
    /// index; the gates themselves are re-checked by [`Self::check_link`].
    pub fn find_candidates(&self, graph: &CapturedGraph) -> Vec<FusionCandidate> {
        let mut candidates = Vec::new();

        for (producer_idx, node) in graph.nodes.iter().enumerate() {
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
            if self.check_link(graph, producer_idx, consumer_idx).is_err() {
                continue;
            }
            let Some(producer_op) = builtin_operation(&node.operation) else {
                continue;
            };
            let Some(consumer_op) = builtin_operation(&graph.nodes[consumer_idx].operation) else {
                continue;
            };
            candidates.push(FusionCandidate {
                producer_idx,
                consumer_idx,
                producer_op,
                consumer_op,
            });
        }

        candidates
    }

    /// Assembles candidate edges into vertex-disjoint fusion groups.
    ///
    /// Each edge is re-validated through [`Self::check_link`], so a pair that
    /// was constructed by hand rather than by [`Self::find_candidates`] faces
    /// the same gates. Two further gates apply at plan time:
    ///
    /// * an intermediate listed in `saved` cannot disappear — fusing it away
    ///   would silently change what the backward tape can replay;
    /// * when two edges compete for the same consumer (first-wins on a sorted
    ///   order), the loser is refused rather than merged. Sharing a node as
    ///   one edge's consumer and the next edge's producer is a chain, not a
    ///   conflict, and is what the path assembly below joins.
    ///
    /// Accepted edges are then assembled into maximal ascending paths, and a
    /// path whose values do not share one shape and dtype is refused whole: a
    /// single pointwise kernel iterates one geometry. Two lowering gates then
    /// run so a returned group is one a backend can actually emit: the chain
    /// must be a builtin float the scalar templates compute, and its distinct
    /// boundary operands must fit the one-or-two the pointwise templates hold.
    #[must_use]
    pub fn plan_groups(
        &self,
        graph: &CapturedGraph,
        candidates: &[FusionCandidate],
        saved: &SavedTensorSet,
    ) -> PlannedGroups {
        struct Edge {
            producer_idx: usize,
            consumer_idx: usize,
            producer_op: OperationKind,
        }

        let mut groups = Vec::new();
        let mut refused = Vec::new();

        let mut seen = BTreeSet::new();
        let mut unique: Vec<&FusionCandidate> = Vec::new();
        for candidate in candidates {
            if seen.insert((candidate.producer_idx, candidate.consumer_idx)) {
                unique.push(candidate);
            }
        }
        unique.sort_by_key(|candidate| (candidate.producer_idx, candidate.consumer_idx));

        let mut used_consumers = BTreeSet::new();
        let mut accepted: Vec<Edge> = Vec::new();
        for candidate in unique {
            if let Err(blocker) =
                self.check_link(graph, candidate.producer_idx, candidate.consumer_idx)
            {
                refused.push(GroupRefusal::Link {
                    producer_idx: candidate.producer_idx,
                    consumer_idx: candidate.consumer_idx,
                    blocker,
                });
                continue;
            }
            let value = graph.nodes[candidate.producer_idx].outputs[0];
            if saved.contains(value) {
                refused.push(GroupRefusal::Link {
                    producer_idx: candidate.producer_idx,
                    consumer_idx: candidate.consumer_idx,
                    blocker: FusionBlocker::SavedForBackward,
                });
                continue;
            }
            // First-wins on the consumer only. A node consumed by one accepted
            // edge and producing the next is the chain junction itself;
            // claiming it twice as a *consumer* is the conflict that would put
            // one node into two groups.
            if used_consumers.contains(&candidate.consumer_idx) {
                refused.push(GroupRefusal::Link {
                    producer_idx: candidate.producer_idx,
                    consumer_idx: candidate.consumer_idx,
                    blocker: FusionBlocker::MultipleProducers,
                });
                continue;
            }
            used_consumers.insert(candidate.consumer_idx);
            accepted.push(Edge {
                producer_idx: candidate.producer_idx,
                consumer_idx: candidate.consumer_idx,
                producer_op: candidate.producer_op,
            });
        }

        // Chain accepted edges into maximal ascending paths. A node is a path
        // start when no accepted edge feeds it.
        let mut out_edge: BTreeMap<usize, usize> = BTreeMap::new();
        for (index, edge) in accepted.iter().enumerate() {
            out_edge.insert(edge.producer_idx, index);
        }
        let consumed: BTreeSet<usize> = accepted.iter().map(|edge| edge.consumer_idx).collect();
        let mut visited: BTreeSet<usize> = BTreeSet::new();
        for edge in &accepted {
            if consumed.contains(&edge.producer_idx) {
                continue;
            }
            let mut path = Vec::new();
            let primary_op = edge.producer_op;
            let mut cursor = edge.producer_idx;
            loop {
                if !visited.insert(cursor) {
                    break;
                }
                path.push(cursor);
                let Some(next) = out_edge.get(&cursor).copied() else {
                    break;
                };
                cursor = accepted[next].consumer_idx;
            }
            if path.len() < 2 {
                refused.push(GroupRefusal::Chain {
                    node_indices: path,
                    blocker: FusionBlocker::InsufficientBenefit,
                });
                continue;
            }
            if let Err(blocker) = Self::chain_geometry_ok(graph, &path) {
                refused.push(GroupRefusal::Chain {
                    node_indices: path,
                    blocker,
                });
                continue;
            }
            if let Err(blocker) = Self::chain_lowering_ok(graph, &path) {
                refused.push(GroupRefusal::Chain {
                    node_indices: path,
                    blocker,
                });
                continue;
            }
            groups.push(FusedKernel {
                source_node_indices: path,
                primary_op,
            });
        }

        PlannedGroups { groups, refused }
    }

    /// Every value a path touches must share one shape and dtype.
    fn chain_geometry_ok(
        graph: &CapturedGraph,
        path: &[usize],
    ) -> core::result::Result<(), FusionBlocker> {
        let mut reference: Option<(Vec<usize>, DTypeDescriptor)> = None;
        let mut check = |value: ValueId| -> core::result::Result<(), FusionBlocker> {
            let meta = graph
                .value_metadata
                .get(&value)
                .ok_or(FusionBlocker::ShapeMismatch)?;
            match &reference {
                None => {
                    reference = Some((meta.shape.clone(), meta.dtype));
                    Ok(())
                }
                Some((shape, dtype)) => {
                    if *shape == meta.shape && *dtype == meta.dtype {
                        Ok(())
                    } else {
                        Err(FusionBlocker::ShapeMismatch)
                    }
                }
            }
        };
        for &node_idx in path {
            let node = graph
                .nodes
                .get(node_idx)
                .ok_or(FusionBlocker::OutOfTopologicalOrder)?;
            for &value in node.inputs.iter().chain(node.outputs.iter()) {
                check(value)?;
            }
        }
        Ok(())
    }

    /// The chain must be a builtin float and fit the two-operand templates.
    ///
    /// [`Self::chain_geometry_ok`] already proved every value shares one
    /// descriptor, so one probe decides the dtype. The boundary walk mirrors
    /// what a lowering re-does: each step reads the previous step's output
    /// (the spine) or an external value, the distinct external values are the
    /// emitted kernel's operands, and the pointwise templates hold one or
    /// two. Refusing here keeps [`Self::apply`] from promising a group a
    /// backend [`FusedKernelLowering`] implementation would have to refuse.
    fn chain_lowering_ok(
        graph: &CapturedGraph,
        path: &[usize],
    ) -> core::result::Result<(), FusionBlocker> {
        let probe = path
            .first()
            .and_then(|&index| graph.nodes.get(index))
            .and_then(|node| node.outputs.first())
            .copied()
            .ok_or(FusionBlocker::OutOfTopologicalOrder)?;
        let meta = graph
            .value_metadata
            .get(&probe)
            .ok_or(FusionBlocker::ShapeMismatch)?;
        let dtype = meta
            .dtype
            .builtin_id()
            .ok_or(FusionBlocker::UnsupportedDtype)?;
        if !matches!(
            dtype,
            DTypeId::F16 | DTypeId::BF16 | DTypeId::F32 | DTypeId::F64
        ) {
            return Err(FusionBlocker::UnsupportedDtype);
        }

        let mut boundaries = BTreeSet::new();
        for (step_index, &index) in path.iter().enumerate() {
            let node = graph
                .nodes
                .get(index)
                .ok_or(FusionBlocker::OutOfTopologicalOrder)?;
            let spine = if step_index == 0 {
                None
            } else {
                Some(
                    graph.nodes[path[step_index - 1]]
                        .outputs
                        .first()
                        .copied()
                        .ok_or(FusionBlocker::NotProvenExclusive)?,
                )
            };
            for &input in &node.inputs {
                if spine != Some(input) {
                    boundaries.insert(input);
                }
            }
        }
        if !(1..=2).contains(&boundaries.len()) {
            return Err(FusionBlocker::BoundaryCount);
        }
        Ok(())
    }

    /// Runs fusion admission and returns the graph plus every named outcome.
    ///
    /// Candidates are planned through [`Self::plan_groups`], so each returned
    /// group has proven exclusive consumption of its intermediates, absence
    /// from `saved`, topological order, one shared shape and dtype, a float
    /// dtype, and the one-or-two boundary budget — every property a backend
    /// lowering re-checks through [`FusedKernelLowering`]. Candidates that
    /// fail a gate land in [`PlannedGroups::refused`] with their
    /// [`FusionBlocker`]: fail-closed per group, never a blanket error.
    ///
    /// The graph itself is returned structurally unchanged: admitted nodes
    /// still execute as their original op chain through the compiled plan
    /// path, and emitting a group as one kernel is the backend's
    /// [`FusedKernelLowering`] job.
    ///
    /// # Errors
    ///
    /// Admission is infallible today; the [`Result`] keeps the seam open for
    /// failures that cannot be attributed to one named group.
    pub fn apply(
        &self,
        graph: &CapturedGraph,
        candidates: &[FusionCandidate],
        saved: &SavedTensorSet,
    ) -> Result<(CapturedGraph, PlannedGroups)> {
        Ok((graph.clone(), self.plan_groups(graph, candidates, saved)))
    }
}
