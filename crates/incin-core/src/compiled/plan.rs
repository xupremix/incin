//! Immutable compiled execution plans and dynamic runtime guards.

use alloc::string::String;
use alloc::vec::Vec;

use crate::compiled::capture::CapturedGraph;
use crate::graph::ValueId;
use crate::prelude::{DTypeId, Error, Result};

/// Operation fusion strategy for graph compilation.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum FusionPolicy {
    /// Kernel fusion enabled (default).
    #[default]
    Enabled,
    /// Kernel fusion disabled for debugging or exact execution tracing.
    Disabled,
}

/// Dynamic shape handling policy.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum DynamicShapePolicy {
    /// Runtime shape and dtype verification using guard checks (default).
    #[default]
    Guarded,
    /// Strict static shape validation matching compiled dimensions exactly.
    Strict,
}

/// Options controlling graph compilation.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct CompileOptions {
    /// Kernel fusion policy.
    pub fusion: FusionPolicy,
    /// Dynamic shape handling policy.
    pub dynamic_shapes: DynamicShapePolicy,
}

impl CompileOptions {
    /// Creates default compilation options.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            fusion: FusionPolicy::Enabled,
            dynamic_shapes: DynamicShapePolicy::Guarded,
        }
    }
}

/// A dynamic runtime guard verifying shape and datatype expectations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ShapeGuard {
    /// Value ID in the captured graph.
    pub value_id: ValueId,
    /// Expected tensor shape.
    pub expected_shape: Vec<usize>,
    /// Expected element datatype.
    pub expected_dtype: DTypeId,
}

impl ShapeGuard {
    /// Creates a new guard for a value ID with expected shape and datatype.
    #[must_use]
    pub fn new(value_id: ValueId, expected_shape: Vec<usize>, expected_dtype: DTypeId) -> Self {
        Self {
            value_id,
            expected_shape,
            expected_dtype,
        }
    }

    /// Verifies actual runtime shape and datatype against this guard.
    pub fn check(&self, actual_shape: &[usize], actual_dtype: DTypeId) -> Result<()> {
        if self.expected_dtype != actual_dtype {
            return Err(Error::DTypeStorageMismatch {
                expected: self.expected_dtype.into(),
                got: actual_dtype.into(),
            });
        }
        if self.expected_shape != actual_shape {
            return Err(Error::Msg(alloc::format!(
                "shape guard failed for input value {}: expected {:?}, got {:?}",
                self.value_id,
                self.expected_shape,
                actual_shape
            )));
        }
        Ok(())
    }
}

/// An immutable compiled execution plan built from a captured graph.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompiledPlan {
    /// The captured IR graph.
    pub graph: CapturedGraph,
    /// Compilation options used.
    pub options: CompileOptions,
    /// Dynamic shape guards for graph inputs.
    pub input_guards: Vec<ShapeGuard>,
}

impl CompiledPlan {
    /// Creates a compiled plan from a captured graph and options, initializing input guards.
    #[must_use]
    pub fn compile(graph: CapturedGraph, options: CompileOptions) -> Self {
        let input_guards = graph
            .inputs
            .iter()
            .filter_map(|&in_id| {
                graph
                    .value(in_id)
                    .map(|value| ShapeGuard::new(in_id, value.shape.clone(), value.dtype))
            })
            .collect();

        Self {
            graph,
            options,
            input_guards,
        }
    }

    /// Validates dynamic guards for provided inputs.
    pub fn verify_input(&self, index: usize, shape: &[usize], dtype: DTypeId) -> Result<()> {
        let guard = self.input_guards.get(index).ok_or_else(|| {
            Error::Msg(alloc::format!(
                "compiled input index {} is out of range; expected {} inputs",
                index,
                self.input_guards.len()
            ))
        })?;
        guard.check(shape, dtype)
    }
}
