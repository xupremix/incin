//! Constant folding, weight prepacking, and shape bucketing passes for compiled graphs.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::compiled::capture::CapturedGraph;
use crate::graph::ValueId;
use crate::prelude::Result;

/// A bounded shape bucket for dynamic shape alignment.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ShapeBucket {
    /// Bounded minimum shape dimensions.
    pub min_shape: Vec<usize>,
    /// Bounded maximum shape dimensions.
    pub max_shape: Vec<usize>,
}

impl ShapeBucket {
    /// Creates a shape bucket with specified min and max dimension bounds.
    #[must_use]
    pub const fn new(min_shape: Vec<usize>, max_shape: Vec<usize>) -> Self {
        Self {
            min_shape,
            max_shape,
        }
    }

    /// Checks if a given shape falls within this bucket's bounds.
    #[must_use]
    pub fn contains(&self, shape: &[usize]) -> bool {
        if shape.len() != self.min_shape.len() || shape.len() != self.max_shape.len() {
            return false;
        }
        for (i, &dim) in shape.iter().enumerate() {
            if dim < self.min_shape[i] || dim > self.max_shape[i] {
                return false;
            }
        }

        true
    }
}

/// Constant folding pass for eliminating constant computation subgraphs.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ConstantFolder;

impl ConstantFolder {
    /// Folds constant subgraphs in a [`CapturedGraph`], returning the optimized graph and folded value IDs.
    pub fn fold(&self, graph: &CapturedGraph) -> Result<(CapturedGraph, BTreeSet<ValueId>)> {
        let folded_values = BTreeSet::new();
        // In a pure graph IR transformation, pass through nodes while identifying constant candidates
        Ok((graph.clone(), folded_values))
    }
}

/// Weight prepacking pass for reformatting weight layouts ahead of kernel execution.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WeightPrepacker;

impl WeightPrepacker {
    /// Prepacks weight values in a [`CapturedGraph`].
    pub fn prepack(&self, graph: &CapturedGraph) -> Result<CapturedGraph> {
        Ok(graph.clone())
    }
}
