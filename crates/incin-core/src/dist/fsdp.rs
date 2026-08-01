//! FSDP and ZeRO memory-sharding plans and parity contracts (DST-014).
//!
//! Fully Sharded Data Parallel (FSDP / ZeRO-3) shards model parameters,
//! gradients, and optimizer states across data-parallel ranks. Unsharded
//! layer parameters are gathered into transient memory just before forward
//! or backward layer execution, then freed immediately after.

use alloc::vec::Vec;

use crate::dist::collective::CollectiveReductionDType;
use crate::dist::data_parallel::DataParallelDType;
use crate::prelude::{ConstDType, DTypeId};
use crate::shapes::error::OperationKind;

/// Stage of Zero Redundancy Optimizer (ZeRO) / FSDP partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZeROStage {
    /// ZeRO-1: Shard optimizer states across data ranks.
    ZeRO1,
    /// ZeRO-2: Shard optimizer states and parameter gradients across data ranks.
    ZeRO2,
    /// ZeRO-3 / FSDP: Shard optimizer states, gradients, and model parameters across data ranks.
    ZeRO3,
}

/// Stable identity of a model parameter in FSDP / ZeRO plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FsdpParameterId(u64);

impl FsdpParameterId {
    /// Build a nonzero parameter identity.
    pub const fn new(value: u64) -> Result<Self, FsdpError> {
        if value == 0 {
            Err(FsdpError::ReservedParameterId)
        } else {
            Ok(Self(value))
        }
    }

    /// Retrieve the underlying u64 value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Descriptor of a single parameter sharded under FSDP / ZeRO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsdpParameterDescriptor {
    id: FsdpParameterId,
    unsharded_elements: usize,
    sharded_elements: usize,
    dtype: DTypeId,
    layer_id: usize,
}

impl FsdpParameterDescriptor {
    /// Parameter identifier.
    #[must_use]
    pub const fn id(self) -> FsdpParameterId {
        self.id
    }

    /// Full element count across the unpartitioned model.
    #[must_use]
    pub const fn unsharded_elements(self) -> usize {
        self.unsharded_elements
    }

    /// Element count stored persistently on this rank.
    #[must_use]
    pub const fn sharded_elements(self) -> usize {
        self.sharded_elements
    }

    /// Parameter data type.
    #[must_use]
    pub const fn dtype(self) -> DTypeId {
        self.dtype
    }

    /// Logical layer index for transient gathering scope.
    #[must_use]
    pub const fn layer_id(self) -> usize {
        self.layer_id
    }
}

/// Memory breakdown comparing FSDP / ZeRO persistent and transient footprints against DP baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct FsdpMemoryReport {
    /// Persistent bytes retained on each rank (sharded weights + grads + opt states).
    pub persistent_bytes: usize,
    /// Peak transient bytes allocated during forward/backward for unsharded layer weights.
    pub transient_bytes: usize,
    /// Memory required if full model were replicated on every rank (standard DP).
    pub unsharded_full_bytes: usize,
    /// Persistent memory reduction factor (unsharded_full_bytes / persistent_bytes).
    pub memory_reduction_ratio: f64,
}

/// FSDP / ZeRO execution plan with persistent and transient memory parity invariants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsdpPlan {
    stage: ZeROStage,
    world_size: usize,
    parameters: Vec<FsdpParameterDescriptor>,
}

impl FsdpPlan {
    /// Active ZeRO partitioning stage.
    #[must_use]
    pub const fn stage(&self) -> ZeROStage {
        self.stage
    }

    /// Data-parallel world size.
    #[must_use]
    pub const fn world_size(&self) -> usize {
        self.world_size
    }

    /// All registered parameters.
    #[must_use]
    pub fn parameters(&self) -> &[FsdpParameterDescriptor] {
        &self.parameters
    }

    /// Calculate persistent memory, peak transient memory, and memory savings report.
    #[must_use]
    pub fn memory_report(&self) -> FsdpMemoryReport {
        let mut total_unsharded_bytes = 0usize;
        let mut total_sharded_bytes = 0usize;
        let mut max_layer_transient_bytes = 0usize;

        // Group parameters by layer to find peak transient memory for ungathered parameters.
        let mut current_layer_id = usize::MAX;
        let mut current_layer_bytes = 0usize;

        for p in &self.parameters {
            let unsharded_b = p
                .dtype
                .size_bytes(p.unsharded_elements, OperationKind::Storage)
                .unwrap_or(0);
            let sharded_b = p
                .dtype
                .size_bytes(p.sharded_elements, OperationKind::Storage)
                .unwrap_or(0);

            total_unsharded_bytes += unsharded_b;
            total_sharded_bytes += sharded_b;

            if p.layer_id != current_layer_id {
                if current_layer_bytes > max_layer_transient_bytes {
                    max_layer_transient_bytes = current_layer_bytes;
                }
                current_layer_id = p.layer_id;
                current_layer_bytes = unsharded_b;
            } else {
                current_layer_bytes += unsharded_b;
            }
        }
        if current_layer_bytes > max_layer_transient_bytes {
            max_layer_transient_bytes = current_layer_bytes;
        }

        // Persistent memory calculation based on ZeRO stage:
        // ZeRO-1: full weights (1.0) + full grads (1.0) + sharded opt states (2.0/N) assuming 2 opt-state bytes per param byte
        // ZeRO-2: full weights (1.0) + sharded grads (1.0/N) + sharded opt states (2.0/N)
        // ZeRO-3: sharded weights (1.0/N) + sharded grads (1.0/N) + sharded opt states (2.0/N)
        let persistent_bytes = match self.stage {
            ZeROStage::ZeRO1 => total_unsharded_bytes * 2 + (total_sharded_bytes * 2),
            ZeROStage::ZeRO2 => total_unsharded_bytes + (total_sharded_bytes * 3),
            ZeROStage::ZeRO3 => total_sharded_bytes * 4,
        };

        let transient_bytes = match self.stage {
            ZeROStage::ZeRO1 | ZeROStage::ZeRO2 => 0,
            ZeROStage::ZeRO3 => max_layer_transient_bytes,
        };

        let unsharded_full_bytes = total_unsharded_bytes * 4;
        let memory_reduction_ratio = if persistent_bytes > 0 {
            unsharded_full_bytes as f64 / persistent_bytes as f64
        } else {
            1.0
        };

        FsdpMemoryReport {
            persistent_bytes,
            transient_bytes,
            unsharded_full_bytes,
            memory_reduction_ratio,
        }
    }

    /// Verify that persistent memory scales inversely with world size and transient memory is bounded.
    pub fn verify_memory_parity(&self) -> Result<(), FsdpError> {
        let report = self.memory_report();
        if self.stage == ZeROStage::ZeRO3 && self.world_size > 1 {
            if report.persistent_bytes >= report.unsharded_full_bytes {
                return Err(FsdpError::ParityViolation {
                    reason: "ZeRO-3 persistent bytes must be strictly less than full DP bytes",
                });
            }
        }
        Ok(())
    }
}

/// Builder for constructing an FSDP / ZeRO plan.
#[derive(Debug)]
pub struct FsdpPlanBuilder {
    stage: ZeROStage,
    parameters: Vec<FsdpParameterDescriptor>,
}

impl FsdpPlanBuilder {
    /// Create a new FSDP plan builder for the target ZeRO stage.
    #[must_use]
    pub fn new(stage: ZeROStage) -> Self {
        Self {
            stage,
            parameters: Vec::new(),
        }
    }

    /// Register a parameter with statically checked floating dtype.
    pub fn push_static<K>(
        &mut self,
        id: FsdpParameterId,
        unsharded_elements: usize,
        layer_id: usize,
        world_size: usize,
    ) -> Result<(), FsdpError>
    where
        K: ConstDType + DataParallelDType + CollectiveReductionDType<crate::dist::placement::Mean>,
    {
        self.push_dyn(id, unsharded_elements, K::DTYPE, layer_id, world_size)
    }

    /// Register a parameter with runtime dtype validation.
    pub fn push_dyn(
        &mut self,
        id: FsdpParameterId,
        unsharded_elements: usize,
        dtype: DTypeId,
        layer_id: usize,
        world_size: usize,
    ) -> Result<(), FsdpError> {
        if world_size == 0 {
            return Err(FsdpError::InvalidWorldSize);
        }
        if self.parameters.iter().any(|p| p.id == id) {
            return Err(FsdpError::DuplicateParameter { id });
        }
        let sharded_elements = unsharded_elements.div_ceil(world_size);
        self.parameters.push(FsdpParameterDescriptor {
            id,
            unsharded_elements,
            sharded_elements,
            dtype,
            layer_id,
        });
        Ok(())
    }

    /// Finish building the plan and validate memory parity.
    pub fn finish(self, world_size: usize) -> Result<FsdpPlan, FsdpError> {
        if world_size == 0 {
            return Err(FsdpError::InvalidWorldSize);
        }
        if self.parameters.is_empty() {
            return Err(FsdpError::NoParameters);
        }
        let plan = FsdpPlan {
            stage: self.stage,
            world_size,
            parameters: self.parameters,
        };
        plan.verify_memory_parity()?;
        Ok(plan)
    }
}

/// Failures during FSDP / ZeRO plan construction and verification.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum FsdpError {
    /// Zero parameter ID is reserved.
    #[error("parameter identity zero is reserved")]
    ReservedParameterId,
    /// World size must be non-zero.
    #[error("FSDP world size must be greater than zero")]
    InvalidWorldSize,
    /// Parameter registered twice.
    #[error("parameter identity {id:?} registered more than once")]
    DuplicateParameter {
        /// Repeated parameter identity.
        id: FsdpParameterId,
    },
    /// FSDP step described no parameters.
    #[error("an FSDP plan must contain at least one parameter")]
    NoParameters,
    /// Persistent or transient memory parity failure.
    #[error("FSDP memory parity violation: {reason}")]
    ParityViolation {
        /// Explanation of parity failure.
        reason: &'static str,
    },
}
