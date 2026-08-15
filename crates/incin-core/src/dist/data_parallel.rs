//! Typed two-rank data-parallel gradient plans.
//!
//! Each rank evaluates a different batch shard against a replicated model.
//! Local parameter gradients are therefore partial means; an all-reduce with
//! mean semantics turns each one into the replicated full-batch gradient.
//! The static API admits only floating dtypes and exactly `DP=2`. [`Dyn`]
//! follows the same path after runtime dtype validation.

use alloc::vec::Vec;

use half::{bf16, f16};
use typenum::{U1, U2};

use crate::dist::collective::{CollectiveError, CollectiveReductionDType, StreamId};
use crate::dist::mesh::{Data, DeviceMesh, MeshAxis, MeshSpec, Pipeline, TensorParallel};
use crate::dist::placement::{Mean, Partial, PlacementKind, Replicated};
use crate::dist::plan::{
    CollectivePlan, CollectivePlanBuilder, CollectiveTag, PlanError, SequenceToken,
};
use crate::shapes::Dyn;
use crate::tensor::dtype::{BuiltinDType, ConstDType, DTypeId};

/// Exactly two data replicas and no tensor or pipeline partitioning.
pub type TwoRankDataParallel = MeshSpec<Data<U2>, TensorParallel<U1>, Pipeline<U1>>;

/// Dtypes whose mean reduction is valid for data-parallel gradients.
///
/// Static integer and block-quantized dtypes intentionally have no
/// implementation. [`Dyn`] is admitted so
/// [`validate_data_parallel_dtype`] can enforce the same rule at runtime.
pub trait DataParallelDType: CollectiveReductionDType<Mean> {}

impl DataParallelDType for f32 {}
impl DataParallelDType for f64 {}
impl DataParallelDType for f16 {}
impl DataParallelDType for bf16 {}
impl DataParallelDType for Dyn {}

/// Runtime counterpart of [`DataParallelDType`].
pub const fn validate_data_parallel_dtype(dtype: DTypeId) -> Result<(), DataParallelError> {
    match dtype {
        DTypeId::BF16 | DTypeId::F16 | DTypeId::F32 | DTypeId::F64 => Ok(()),
        DTypeId::U8 | DTypeId::U32 | DTypeId::I64 | DTypeId::Q8_0 | DTypeId::Bool => {
            Err(DataParallelError::UnsupportedGradientDType { dtype })
        }
    }
}

/// Stable identity of a model parameter's gradient.
///
/// The identity must be derived from stable model structure, not an address or
/// process-local hash. It is included in collective preflight, so ranks that
/// swap two equally shaped parameters still disagree before launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GradientId(u64);

impl GradientId {
    /// Build a nonzero stable identity.
    pub const fn new(value: u64) -> Result<Self, DataParallelError> {
        if value == 0 {
            Err(DataParallelError::ReservedGradientId)
        } else {
            Ok(Self(value))
        }
    }

    /// Numeric identity included in the collective plan hash.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One parameter gradient in data-parallel launch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientDescriptor {
    id: GradientId,
    elements: usize,
    dtype: DTypeId,
    sequence: SequenceToken,
}

impl GradientDescriptor {
    /// Stable model-parameter identity.
    #[must_use]
    pub const fn id(self) -> GradientId {
        self.id
    }

    /// Logical elements in this rank's gradient.
    #[must_use]
    pub const fn elements(self) -> usize {
        self.elements
    }

    /// Static or runtime-resolved gradient dtype.
    #[must_use]
    pub const fn dtype(self) -> DTypeId {
        self.dtype
    }

    /// Position in the collective plan.
    #[must_use]
    pub const fn sequence(self) -> SequenceToken {
        self.sequence
    }
}

/// Immutable DP=2 plan and its model-gradient identities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataParallelPlan {
    collective: CollectivePlan,
    gradients: Vec<GradientDescriptor>,
}

impl DataParallelPlan {
    /// Transport-neutral collective plan consumed by reference or NCCL.
    #[must_use]
    pub const fn collective_plan(&self) -> &CollectivePlan {
        &self.collective
    }

    /// Ordered parameter gradients.
    #[must_use]
    pub fn gradients(&self) -> &[GradientDescriptor] {
        &self.gradients
    }

    /// Consume the wrapper for transport bootstrap.
    #[must_use]
    pub fn into_collective_plan(self) -> CollectivePlan {
        self.collective
    }
}

/// Builder for exactly two data-parallel ranks.
pub struct DataParallelPlanBuilder<'a> {
    inner: CollectivePlanBuilder<'a, TwoRankDataParallel>,
    rank: usize,
    previous: Option<SequenceToken>,
    gradients: Vec<GradientDescriptor>,
}

impl<'a> DataParallelPlanBuilder<'a> {
    /// Begin a rank-local plan on a physically bound DP=2 mesh.
    #[must_use]
    pub fn new(mesh: &'a DeviceMesh<TwoRankDataParallel>, rank: usize) -> Self {
        Self {
            inner: CollectivePlanBuilder::new(mesh),
            rank,
            previous: None,
            gradients: Vec::new(),
        }
    }

    /// Append a statically typed floating gradient.
    pub fn push_static<K>(
        &mut self,
        id: GradientId,
        elements: usize,
        stream: StreamId,
    ) -> Result<SequenceToken, DataParallelError>
    where
        K: ConstDType + BuiltinDType + DataParallelDType,
    {
        let rank = self.rank;
        self.push_common(id, elements, K::DTYPE, |inner, tag, dependency| {
            inner.push_static_tagged::<
                K,
                Partial<TwoRankDataParallel, Mean>,
                Replicated<TwoRankDataParallel>,
            >(
                tag,
                MeshAxis::Data,
                rank,
                elements,
                stream,
                dependency,
            )
        })
    }

    /// Append a runtime-selected gradient with the same floating-dtype rule.
    pub fn push_dyn(
        &mut self,
        id: GradientId,
        elements: usize,
        dtype: DTypeId,
        stream: StreamId,
    ) -> Result<SequenceToken, DataParallelError> {
        validate_data_parallel_dtype(dtype)?;
        let rank = self.rank;
        self.push_common(id, elements, dtype, |inner, tag, dependency| {
            inner.push_dyn_tagged(
                tag,
                MeshAxis::Data,
                rank,
                elements,
                dtype,
                PlacementKind::Partial {
                    reduction: crate::exec::ReduceOp::Mean,
                },
                PlacementKind::Replicated,
                stream,
                dependency,
            )
        })
    }

    fn push_common(
        &mut self,
        id: GradientId,
        elements: usize,
        dtype: DTypeId,
        append: impl FnOnce(
            &mut CollectivePlanBuilder<'a, TwoRankDataParallel>,
            CollectiveTag,
            Option<SequenceToken>,
        ) -> Result<SequenceToken, PlanError>,
    ) -> Result<SequenceToken, DataParallelError> {
        if self.gradients.iter().any(|gradient| gradient.id == id) {
            return Err(DataParallelError::DuplicateGradient { id });
        }
        let sequence = append(&mut self.inner, CollectiveTag::new(id.get()), self.previous)?;
        self.gradients.push(GradientDescriptor {
            id,
            elements,
            dtype,
            sequence,
        });
        self.previous = Some(sequence);
        Ok(sequence)
    }

    /// Freeze the plan. A training step with no parameter gradients is an
    /// explicit error rather than a vacuous distributed success.
    pub fn finish(self) -> Result<DataParallelPlan, DataParallelError> {
        if self.gradients.is_empty() {
            return Err(DataParallelError::NoGradients);
        }
        Ok(DataParallelPlan {
            collective: self.inner.finish(),
            gradients: self.gradients,
        })
    }
}

/// Failure while building a DP=2 gradient plan.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum DataParallelError {
    /// Zero is reserved for generic unlabelled collectives.
    #[error("gradient identity zero is reserved for unlabelled collectives")]
    ReservedGradientId,
    /// The same parameter appeared twice in one step.
    #[error("gradient identity {id:?} appears more than once")]
    DuplicateGradient {
        /// Repeated parameter identity.
        id: GradientId,
    },
    /// A data-parallel step described no gradients.
    #[error("a data-parallel step must contain at least one gradient")]
    NoGradients,
    /// Runtime `Dyn` selected a dtype whose mean has no gradient semantics.
    #[error("dtype {dtype:?} cannot represent a data-parallel mean gradient")]
    UnsupportedGradientDType {
        /// Rejected runtime dtype.
        dtype: DTypeId,
    },
    /// The underlying collective plan was invalid.
    #[error(transparent)]
    Plan(#[from] PlanError),
    /// Shared collective validation failed.
    #[error(transparent)]
    Collective(#[from] CollectiveError),
}
