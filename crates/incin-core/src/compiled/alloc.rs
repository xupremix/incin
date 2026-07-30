//! Liveness analysis and buffer allocation planning for compiled graphs.

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::compiled::capture::CapturedGraph;
use crate::graph::ValueId;
use crate::prelude::Result;

/// The liveness interval of a value in the compiled graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LivenessInterval {
    /// Index of the node that defines this value.
    pub def_node: usize,
    /// Index of the last node that uses this value.
    pub last_use_node: usize,
}

impl LivenessInterval {
    /// Returns `true` if this interval overlaps with another.
    #[must_use]
    pub fn overlaps_with(&self, other: &Self) -> bool {
        self.def_node <= other.last_use_node && other.def_node <= self.last_use_node
    }
}

/// Maps each value to its liveness interval in a compiled graph.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LivenessMap {
    intervals: BTreeMap<ValueId, LivenessInterval>,
}

impl LivenessMap {
    /// Computes a liveness map from a captured graph.
    #[must_use]
    pub fn compute(graph: &CapturedGraph) -> Self {
        let mut intervals: BTreeMap<ValueId, LivenessInterval> = BTreeMap::new();

        // Input values are live from the beginning
        for &input in &graph.inputs {
            intervals.insert(
                input,
                LivenessInterval {
                    def_node: 0,
                    last_use_node: 0,
                },
            );
        }

        // Walk nodes and track def/use
        for (node_idx, node) in graph.nodes.iter().enumerate() {
            // Outputs are defined at this node
            for &out_id in &node.outputs {
                intervals
                    .entry(out_id)
                    .or_insert(LivenessInterval {
                        def_node: node_idx,
                        last_use_node: node_idx,
                    })
                    .last_use_node = node_idx;
            }
            // Inputs used at this node — extend their last use
            for &in_id in &node.inputs {
                if let Some(interval) = intervals.get_mut(&in_id)
                    && node_idx > interval.last_use_node
                {
                    interval.last_use_node = node_idx;
                }
            }
        }

        // Outputs are live until the end of the graph
        for &out_id in &graph.outputs {
            if let Some(interval) = intervals.get_mut(&out_id) {
                interval.last_use_node = graph.nodes.len();
            }
        }

        Self { intervals }
    }

    /// Returns the liveness interval for a given value, if known.
    #[must_use]
    pub fn get(&self, value_id: ValueId) -> Option<LivenessInterval> {
        self.intervals.get(&value_id).copied()
    }

    /// Returns all values in the liveness map.
    #[must_use]
    pub fn values(&self) -> Vec<ValueId> {
        self.intervals.keys().copied().collect()
    }

    /// Finds pairs of values whose liveness intervals do not overlap (alias candidates).
    #[must_use]
    pub fn alias_candidates(&self) -> Vec<(ValueId, ValueId)> {
        let ids: Vec<ValueId> = self.intervals.keys().copied().collect();
        let mut candidates = Vec::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a = ids[i];
                let b = ids[j];
                let ia = &self.intervals[&a];
                let ib = &self.intervals[&b];
                if !ia.overlaps_with(ib) {
                    candidates.push((a, b));
                }
            }
        }
        candidates
    }

    /// Extends liveness intervals to account for saved tensors kept alive by autograd.
    ///
    /// Saved tensors must remain live until the backward pass (represented by
    /// `backward_end_node`), even if they appear unused after the forward node.
    pub fn extend_for_saved_tensors(
        &mut self,
        saved: &SavedTensorSet,
        backward_end_node: usize,
    ) {
        for &vid in &saved.values {
            if let Some(interval) = self.intervals.get_mut(&vid)
                && interval.last_use_node < backward_end_node
            {
                interval.last_use_node = backward_end_node;
            }

        }
    }
}

/// A set of value IDs representing tensors saved for the backward pass.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct SavedTensorSet {
    /// Values stashed by forward ops for use in backward recipes.
    pub values: BTreeSet<ValueId>,
}

impl SavedTensorSet {
    /// Creates a new empty `SavedTensorSet`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks a value as saved.
    pub fn save(&mut self, value_id: ValueId) {
        self.values.insert(value_id);
    }

    /// Returns `true` if the value is in this set.
    #[must_use]
    pub fn contains(&self, value_id: ValueId) -> bool {
        self.values.contains(&value_id)
    }
}


/// Allocation slot for a buffer, identified by an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct BufferSlot(pub usize);

/// A memory plan assigning each value to a buffer slot.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryPlan {
    /// Mapping from value ID to buffer slot.
    pub assignments: BTreeMap<ValueId, BufferSlot>,
    /// Estimated peak number of simultaneously live slots.
    pub peak_live_slots: usize,
}

/// Allocation planner producing a `MemoryPlan` from liveness analysis.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AllocationPlanner;

impl AllocationPlanner {
    /// Produces a `MemoryPlan` from a `LivenessMap`.
    pub fn plan(&self, liveness: &LivenessMap, graph: &CapturedGraph) -> Result<MemoryPlan> {
        let mut assignments: BTreeMap<ValueId, BufferSlot> = BTreeMap::new();
        let mut free_slots: Vec<BufferSlot> = Vec::new();
        let mut next_slot: usize = 0;
        let mut peak_live_slots: usize = 0;
        let mut currently_live: BTreeSet<ValueId> = BTreeSet::new();

        // Pre-assign slots to graph inputs
        for &input in &graph.inputs {
            let slot = BufferSlot(next_slot);
            next_slot += 1;
            assignments.insert(input, slot);
            currently_live.insert(input);
        }
        peak_live_slots = peak_live_slots.max(currently_live.len());

        let node_count = graph.nodes.len();

        for node_idx in 0..=node_count {
            // Release inputs whose last use is before (or at) this node
            let to_free: Vec<ValueId> = currently_live
                .iter()
                .filter(|&&vid| {
                    liveness
                        .get(vid)
                        .is_some_and(|iv| iv.last_use_node < node_idx)
                })
                .copied()
                .collect();
            for vid in to_free {
                currently_live.remove(&vid);
                if let Some(&slot) = assignments.get(&vid) {
                    free_slots.push(slot);
                }
            }

            // Allocate output slots for this node
            for node in graph.nodes.iter().filter(|n| n.id == node_idx) {
                for &out_id in &node.outputs {
                    let slot = if let Some(s) = free_slots.pop() {
                        s
                    } else {
                        let s = BufferSlot(next_slot);
                        next_slot += 1;
                        s
                    };
                    assignments.insert(out_id, slot);
                    currently_live.insert(out_id);
                }
            }

            peak_live_slots = peak_live_slots.max(currently_live.len());
        }

        Ok(MemoryPlan {
            assignments,
            peak_live_slots,
        })
    }
}
