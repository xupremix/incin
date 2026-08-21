//! Inspection-only constant-folding, prepacking, and shape-bucketing types.
//!
//! Folding and prepacking have no executable lowering in the preview CPU
//! evaluator and therefore fail closed.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::compiled::capture::CapturedGraph;
use crate::err::{Error, Result};
use crate::graph::ValueId;

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

/// Inspection-only constant-folding pass; execution requests fail closed.
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

/// Inspection-only weight-prepacking pass; execution requests fail closed.
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
