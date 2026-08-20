//! Runtime workload values the analytical hybrid planner scores strategies
//! against.

use super::*;

/// Runtime workload values needed by the analytical planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridWorkload {
    pub(super) batch_size: usize,
    pub(super) tensor_shard_extent: usize,
    pub(super) parameter_elements: usize,
    pub(super) activation_elements_per_microbatch: usize,
    pub(super) microbatches: usize,
    pub(super) optimizer_state_copies: usize,
    pub(super) device_capacity_bytes: [usize; 2],
}

impl HybridWorkload {
    /// Build a checked workload.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        batch_size: usize,
        tensor_shard_extent: usize,
        parameter_elements: usize,
        activation_elements_per_microbatch: usize,
        microbatches: usize,
        optimizer_state_copies: usize,
        device_capacity_bytes: [usize; 2],
    ) -> Result<Self, HybridPlanError> {
        if batch_size == 0 {
            return Err(HybridPlanError::ZeroWorkloadField {
                field: WorkloadField::BatchSize,
            });
        }
        if tensor_shard_extent == 0 {
            return Err(HybridPlanError::ZeroWorkloadField {
                field: WorkloadField::TensorShardExtent,
            });
        }
        if parameter_elements == 0 {
            return Err(HybridPlanError::ZeroWorkloadField {
                field: WorkloadField::ParameterElements,
            });
        }
        if activation_elements_per_microbatch == 0 {
            return Err(HybridPlanError::ZeroWorkloadField {
                field: WorkloadField::ActivationElements,
            });
        }
        if microbatches == 0 {
            return Err(HybridPlanError::ZeroWorkloadField {
                field: WorkloadField::Microbatches,
            });
        }
        if device_capacity_bytes[0] == 0 {
            return Err(HybridPlanError::ZeroDeviceCapacity { rank: 0 });
        }
        if device_capacity_bytes[1] == 0 {
            return Err(HybridPlanError::ZeroDeviceCapacity { rank: 1 });
        }
        Ok(Self {
            batch_size,
            tensor_shard_extent,
            parameter_elements,
            activation_elements_per_microbatch,
            microbatches,
            optimizer_state_copies,
            device_capacity_bytes,
        })
    }

    /// Global batch size.
    #[must_use]
    pub const fn batch_size(self) -> usize {
        self.batch_size
    }

    /// Dimension a TP=2 candidate would split.
    #[must_use]
    pub const fn tensor_shard_extent(self) -> usize {
        self.tensor_shard_extent
    }

    /// Total trainable parameter elements.
    #[must_use]
    pub const fn parameter_elements(self) -> usize {
        self.parameter_elements
    }

    /// Boundary activation elements for one microbatch.
    #[must_use]
    pub const fn activation_elements_per_microbatch(self) -> usize {
        self.activation_elements_per_microbatch
    }

    /// Microbatches in one step.
    #[must_use]
    pub const fn microbatches(self) -> usize {
        self.microbatches
    }

    /// Optimizer-state tensors with parameter cardinality.
    #[must_use]
    pub const fn optimizer_state_copies(self) -> usize {
        self.optimizer_state_copies
    }

    /// Physical capacity of each rank's device.
    #[must_use]
    pub const fn device_capacity_bytes(self) -> [usize; 2] {
        self.device_capacity_bytes
    }
}

/// Logical workload field named by a structured planning error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadField {
    /// Global batch.
    BatchSize,
    /// TP shard dimension.
    TensorShardExtent,
    /// Trainable parameters.
    ParameterElements,
    /// Boundary activation.
    ActivationElements,
    /// Pipeline microbatches.
    Microbatches,
}

/// Floating dtypes supported by every initial two-rank strategy.
pub trait HybridPlanDType: PipelineDType {}

impl HybridPlanDType for f32 {}
impl HybridPlanDType for f64 {}
impl HybridPlanDType for f16 {}
impl HybridPlanDType for bf16 {}
impl HybridPlanDType for Dyn {}

/// Runtime counterpart of [`HybridPlanDType`].
pub const fn validate_hybrid_plan_dtype(dtype: DTypeId) -> Result<(), HybridPlanError> {
    match dtype {
        DTypeId::BF16 | DTypeId::F16 | DTypeId::F32 | DTypeId::F64 => Ok(()),
        DTypeId::U8 | DTypeId::U32 | DTypeId::I64 | DTypeId::Q8_0 | DTypeId::Bool => {
            Err(HybridPlanError::UnsupportedDType { dtype })
        }
    }
}
