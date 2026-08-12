//! Constant folding, weight prepacking, and shape bucketing passes for compiled graphs.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::compiled::capture::CapturedGraph;
use crate::graph::ValueId;
use crate::prelude::{Error, Result};

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
    pub fn new(min_shape: Vec<usize>, max_shape: Vec<usize>) -> Result<Self> {
        if min_shape.len() != max_shape.len() {
            return Err(Error::Msg(
                "shape bucket bounds must have equal rank".into(),
            ));
        }
        if min_shape
            .iter()
            .zip(max_shape.iter())
            .any(|(min, max)| min > max)
        {
            return Err(Error::Msg(
                "shape bucket minimum cannot exceed its maximum".into(),
            ));
        }
        Ok(Self {
            min_shape,
            max_shape,
        })
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
    /// Rejects constant folding until the pass performs a real transformation.
    pub fn fold(&self, graph: &CapturedGraph) -> Result<(CapturedGraph, BTreeSet<ValueId>)> {
        let _ = graph;
        Err(Error::UnsupportedBackendOperation {
            op: "compiled.constant_fold",
            backend: "compiled-prototype",
        })
    }
}

/// Weight prepacking pass for reformatting weight layouts ahead of kernel execution.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WeightPrepacker;

impl WeightPrepacker {
    /// Rejects weight prepacking until the pass performs a real transformation.
    pub fn prepack(&self, graph: &CapturedGraph) -> Result<CapturedGraph> {
        let _ = graph;
        Err(Error::UnsupportedBackendOperation {
            op: "compiled.weight_prepack",
            backend: "compiled-prototype",
        })
    }
}
