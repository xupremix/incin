//! Typed two-rank tensor-parallel collective plans.
//!
//! Column-parallel linear and head-parallel attention produce shards that may
//! be gathered into a replicated tensor. Row-parallel linear produces local
//! partial sums that must be all-reduced. This module fixes those three
//! semantics for exactly `TP=2`, keeps operation identity in collective
//! preflight, and gives static and [`Dyn`] callers matching validation paths.

use alloc::vec::Vec;

use half::{bf16, f16};
use typenum::{U1, U2};

use crate::dist::collective::{CollectiveError, CollectiveReductionDType, StreamId};
use crate::dist::mesh::{Data, DeviceMesh, MeshAxis, MeshSpec, Pipeline, TensorParallel};
use crate::dist::placement::{Partial, PlacementAxis, PlacementKind, Replicated, Sharded, Sum};
use crate::dist::plan::{
    CollectivePlan, CollectivePlanBuilder, CollectiveTag, PlanError, SequenceToken,
};
use crate::dist::rule::ShardDivisible;
use crate::shapes::Dyn;
use crate::tensor::dtype::{BuiltinDType, ConstDType, DTypeId};

/// Exactly two tensor-parallel ranks and no data or pipeline partitioning.
pub type TwoRankTensorParallel = MeshSpec<Data<U1>, TensorParallel<U2>, Pipeline<U1>>;

/// Floating dtypes supported by the first tensor-parallel linear/attention path.
///
/// Static integer and block-quantized dtypes intentionally have no
/// implementation. [`Dyn`] is admitted so
/// [`validate_tensor_parallel_dtype`] can enforce the same policy at runtime.
pub trait TensorParallelDType: CollectiveReductionDType<Sum> {}

impl TensorParallelDType for f32 {}
impl TensorParallelDType for f64 {}
impl TensorParallelDType for f16 {}
impl TensorParallelDType for bf16 {}
impl TensorParallelDType for Dyn {}

/// Compile-time proof that an extent divides evenly across two TP ranks.
pub trait TwoWayShard: ShardDivisible<U2> {}

impl<T> TwoWayShard for T where T: ShardDivisible<U2> {}

/// Runtime counterpart of [`TensorParallelDType`].
pub const fn validate_tensor_parallel_dtype(dtype: DTypeId) -> Result<(), TensorParallelError> {
    match dtype {
        DTypeId::BF16 | DTypeId::F16 | DTypeId::F32 | DTypeId::F64 => Ok(()),
        DTypeId::U8 | DTypeId::U32 | DTypeId::I64 | DTypeId::Q8_0 | DTypeId::Bool => {
            Err(TensorParallelError::UnsupportedTensorDType { dtype })
        }
    }
}

/// Which dimension is being statically or dynamically partitioned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorParallelDimension {
    /// A column-parallel linear layer's output-feature dimension.
    OutputFeatures,
    /// A row-parallel linear layer's contraction/input-feature dimension.
    InputFeatures,
    /// An attention layer's head dimension.
    AttentionHeads,
}

/// Validate and project one runtime extent onto two equal shards.
pub const fn validate_two_way_extent(
    dimension: TensorParallelDimension,
    extent: usize,
) -> Result<usize, TensorParallelError> {
    if !extent.is_multiple_of(2) {
        Err(TensorParallelError::NonDivisible {
            dimension,
            extent,
            ranks: 2,
        })
    } else {
        Ok(extent / 2)
    }
}

/// Stable identity of one tensor-parallel semantic operation.
///
/// The upper two bits are rejected because the plan tag reserves two low bits
/// for the operation kind. This makes column and attention gathers with equal
/// shapes distinguishable without relying on process-local hashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorParallelId(u64);

impl TensorParallelId {
    const MAX: u64 = u64::MAX >> 2;

    /// Build a nonzero identity that can be encoded losslessly in a plan tag.
    pub const fn new(value: u64) -> Result<Self, TensorParallelError> {
        if value == 0 {
            Err(TensorParallelError::ReservedOperationId)
        } else if value > Self::MAX {
            Err(TensorParallelError::OperationIdTooLarge {
                maximum: Self::MAX,
                found: value,
            })
        } else {
            Ok(Self(value))
        }
    }

    /// Numeric caller-defined identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Tensor-parallel communication attached to one layer operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorParallelCollective {
    /// Gather output-feature shards from a column-parallel linear.
    ColumnOutputGather {
        /// Tensor axis partitioned across ranks.
        tensor_axis: usize,
    },
    /// Sum row-parallel local products into a replicated output.
    RowOutputSum,
    /// Gather independently computed attention-head shards.
    AttentionHeadGather {
        /// Tensor axis containing the attention heads.
        tensor_axis: usize,
    },
}

impl TensorParallelCollective {
    const fn tag_code(self) -> u64 {
        match self {
            Self::ColumnOutputGather { .. } => 1,
            Self::RowOutputSum => 2,
            Self::AttentionHeadGather { .. } => 3,
        }
    }

    /// Stable semantic tag included in collective preflight.
    #[must_use]
    pub const fn plan_tag(self, id: TensorParallelId) -> CollectiveTag {
        CollectiveTag::new((id.get() << 2) | self.tag_code())
    }
}

/// One tensor-parallel operation in launch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorParallelDescriptor {
    id: TensorParallelId,
    collective: TensorParallelCollective,
    local_elements: usize,
    global_elements: usize,
    dtype: DTypeId,
    sequence: SequenceToken,
}

impl TensorParallelDescriptor {
    /// Stable layer/operation identity.
    #[must_use]
    pub const fn id(self) -> TensorParallelId {
        self.id
    }

    /// Tensor-parallel semantic operation.
    #[must_use]
    pub const fn collective(self) -> TensorParallelCollective {
        self.collective
    }

    /// Elements supplied by this rank.
    #[must_use]
    pub const fn local_elements(self) -> usize {
        self.local_elements
    }

    /// Elements in the replicated logical result.
    #[must_use]
    pub const fn global_elements(self) -> usize {
        self.global_elements
    }

    /// Static or runtime-resolved compute/storage dtype.
    #[must_use]
    pub const fn dtype(self) -> DTypeId {
        self.dtype
    }

    /// Position in the transport plan.
    #[must_use]
    pub const fn sequence(self) -> SequenceToken {
        self.sequence
    }
}

/// Immutable TP=2 plan and its semantic layer operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorParallelPlan {
    collective: CollectivePlan,
    operations: Vec<TensorParallelDescriptor>,
}

impl TensorParallelPlan {
    /// Transport-neutral collective plan.
    #[must_use]
    pub const fn collective_plan(&self) -> &CollectivePlan {
        &self.collective
    }

    /// Ordered tensor-parallel operations.
    #[must_use]
    pub fn operations(&self) -> &[TensorParallelDescriptor] {
        &self.operations
    }

    /// Consume the wrapper for transport bootstrap.
    #[must_use]
    pub fn into_collective_plan(self) -> CollectivePlan {
        self.collective
    }
}

/// Builder for exactly two tensor-parallel ranks.
pub struct TensorParallelPlanBuilder<'a> {
    inner: CollectivePlanBuilder<'a, TwoRankTensorParallel>,
    rank: usize,
    previous: Option<SequenceToken>,
    operations: Vec<TensorParallelDescriptor>,
}

impl<'a> TensorParallelPlanBuilder<'a> {
    /// Begin a rank-local plan on a physically bound TP=2 mesh.
    #[must_use]
    pub fn new(mesh: &'a DeviceMesh<TwoRankTensorParallel>, rank: usize) -> Self {
        Self {
            inner: CollectivePlanBuilder::new(mesh),
            rank,
            previous: None,
            operations: Vec::new(),
        }
    }

    /// Gather a statically proved column-parallel output.
    ///
    /// `Out` is the global output-feature extent and must divide by two.
    /// `outer_elements` is the product of every non-output axis.
    pub fn push_column_static<K, Axis, Out>(
        &mut self,
        id: TensorParallelId,
        outer_elements: usize,
        stream: StreamId,
    ) -> Result<SequenceToken, TensorParallelError>
    where
        K: ConstDType + BuiltinDType + TensorParallelDType,
        Axis: PlacementAxis,
        Out: TwoWayShard,
    {
        let local_elements = checked_product(
            TensorParallelDimension::OutputFeatures,
            outer_elements,
            <Out as ShardDivisible<U2>>::LOCAL,
        )?;
        let global_elements = checked_double(local_elements)?;
        let rank = self.rank;
        let collective = TensorParallelCollective::ColumnOutputGather {
            tensor_axis: Axis::INDEX,
        };
        self.push_common(
            id,
            collective,
            local_elements,
            global_elements,
            K::DTYPE,
            |inner, tag, dependency| {
                inner.push_static_tagged::<
                    K,
                    Sharded<TwoRankTensorParallel, Axis>,
                    Replicated<TwoRankTensorParallel>,
                >(
                    tag,
                    MeshAxis::Tensor,
                    rank,
                    local_elements,
                    stream,
                    dependency,
                )
            },
        )
    }

    /// Runtime-selected counterpart of [`push_column_static`](Self::push_column_static).
    pub fn push_column_dyn(
        &mut self,
        id: TensorParallelId,
        global_shape: &[usize],
        output_axis: usize,
        dtype: DTypeId,
        stream: StreamId,
    ) -> Result<SequenceToken, TensorParallelError> {
        validate_tensor_parallel_dtype(dtype)?;
        let (local_elements, global_elements) = dynamic_sharded_elements(
            TensorParallelDimension::OutputFeatures,
            global_shape,
            output_axis,
        )?;
        let rank = self.rank;
        let collective = TensorParallelCollective::ColumnOutputGather {
            tensor_axis: output_axis,
        };
        self.push_common(
            id,
            collective,
            local_elements,
            global_elements,
            dtype,
            |inner, tag, dependency| {
                inner.push_dyn_tagged(
                    tag,
                    MeshAxis::Tensor,
                    rank,
                    local_elements,
                    dtype,
                    PlacementKind::Sharded { axis: output_axis },
                    PlacementKind::Replicated,
                    stream,
                    dependency,
                )
            },
        )
    }

    /// Sum statically proved row-parallel local products.
    ///
    /// `In` is the global contraction extent and must divide by two.
    pub fn push_row_static<K, In>(
        &mut self,
        id: TensorParallelId,
        output_elements: usize,
        stream: StreamId,
    ) -> Result<SequenceToken, TensorParallelError>
    where
        K: ConstDType + BuiltinDType + TensorParallelDType,
        In: TwoWayShard,
    {
        let _ = <In as ShardDivisible<U2>>::LOCAL;
        let rank = self.rank;
        self.push_common(
            id,
            TensorParallelCollective::RowOutputSum,
            output_elements,
            output_elements,
            K::DTYPE,
            |inner, tag, dependency| {
                inner.push_static_tagged::<
                    K,
                    Partial<TwoRankTensorParallel, Sum>,
                    Replicated<TwoRankTensorParallel>,
                >(
                    tag,
                    MeshAxis::Tensor,
                    rank,
                    output_elements,
                    stream,
                    dependency,
                )
            },
        )
    }

    /// Runtime-selected counterpart of [`push_row_static`](Self::push_row_static).
    pub fn push_row_dyn(
        &mut self,
        id: TensorParallelId,
        global_input_features: usize,
        output_elements: usize,
        dtype: DTypeId,
        stream: StreamId,
    ) -> Result<SequenceToken, TensorParallelError> {
        validate_tensor_parallel_dtype(dtype)?;
        validate_two_way_extent(
            TensorParallelDimension::InputFeatures,
            global_input_features,
        )?;
        let rank = self.rank;
        self.push_common(
            id,
            TensorParallelCollective::RowOutputSum,
            output_elements,
            output_elements,
            dtype,
            |inner, tag, dependency| {
                inner.push_dyn_tagged(
                    tag,
                    MeshAxis::Tensor,
                    rank,
                    output_elements,
                    dtype,
                    PlacementKind::Partial {
                        reduction: crate::exec::ReduceOp::Sum,
                    },
                    PlacementKind::Replicated,
                    stream,
                    dependency,
                )
            },
        )
    }

    /// Gather statically proved attention-head shards.
    ///
    /// `Heads` is the global head count. `elements_per_head` includes batch,
    /// sequence, and head-width factors.
    pub fn push_attention_static<K, Axis, Heads>(
        &mut self,
        id: TensorParallelId,
        elements_per_head: usize,
        stream: StreamId,
    ) -> Result<SequenceToken, TensorParallelError>
    where
        K: ConstDType + BuiltinDType + TensorParallelDType,
        Axis: PlacementAxis,
        Heads: TwoWayShard,
    {
        let local_elements = checked_product(
            TensorParallelDimension::AttentionHeads,
            elements_per_head,
            <Heads as ShardDivisible<U2>>::LOCAL,
        )?;
        let global_elements = checked_double(local_elements)?;
        let rank = self.rank;
        let collective = TensorParallelCollective::AttentionHeadGather {
            tensor_axis: Axis::INDEX,
        };
        self.push_common(
            id,
            collective,
            local_elements,
            global_elements,
            K::DTYPE,
            |inner, tag, dependency| {
                inner.push_static_tagged::<
                    K,
                    Sharded<TwoRankTensorParallel, Axis>,
                    Replicated<TwoRankTensorParallel>,
                >(
                    tag,
                    MeshAxis::Tensor,
                    rank,
                    local_elements,
                    stream,
                    dependency,
                )
            },
        )
    }

    /// Runtime-selected counterpart of
    /// [`push_attention_static`](Self::push_attention_static).
    pub fn push_attention_dyn(
        &mut self,
        id: TensorParallelId,
        global_shape: &[usize],
        head_axis: usize,
        dtype: DTypeId,
        stream: StreamId,
    ) -> Result<SequenceToken, TensorParallelError> {
        validate_tensor_parallel_dtype(dtype)?;
        let (local_elements, global_elements) = dynamic_sharded_elements(
            TensorParallelDimension::AttentionHeads,
            global_shape,
            head_axis,
        )?;
        let rank = self.rank;
        let collective = TensorParallelCollective::AttentionHeadGather {
            tensor_axis: head_axis,
        };
        self.push_common(
            id,
            collective,
            local_elements,
            global_elements,
            dtype,
            |inner, tag, dependency| {
                inner.push_dyn_tagged(
                    tag,
                    MeshAxis::Tensor,
                    rank,
                    local_elements,
                    dtype,
                    PlacementKind::Sharded { axis: head_axis },
                    PlacementKind::Replicated,
                    stream,
                    dependency,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn push_common(
        &mut self,
        id: TensorParallelId,
        collective: TensorParallelCollective,
        local_elements: usize,
        global_elements: usize,
        dtype: DTypeId,
        append: impl FnOnce(
            &mut CollectivePlanBuilder<'a, TwoRankTensorParallel>,
            CollectiveTag,
            Option<SequenceToken>,
        ) -> Result<SequenceToken, PlanError>,
    ) -> Result<SequenceToken, TensorParallelError> {
        if self.operations.iter().any(|operation| operation.id == id) {
            return Err(TensorParallelError::DuplicateOperation { id });
        }
        let tag = collective.plan_tag(id);
        let sequence = append(&mut self.inner, tag, self.previous)?;
        self.operations.push(TensorParallelDescriptor {
            id,
            collective,
            local_elements,
            global_elements,
            dtype,
            sequence,
        });
        self.previous = Some(sequence);
        Ok(sequence)
    }

    /// Freeze the plan. An empty TP plan is rejected rather than reported as a
    /// successful distributed execution.
    pub fn finish(self) -> Result<TensorParallelPlan, TensorParallelError> {
        if self.operations.is_empty() {
            return Err(TensorParallelError::NoOperations);
        }
        Ok(TensorParallelPlan {
            collective: self.inner.finish(),
            operations: self.operations,
        })
    }
}

fn dynamic_sharded_elements(
    dimension: TensorParallelDimension,
    global_shape: &[usize],
    axis: usize,
) -> Result<(usize, usize), TensorParallelError> {
    let Some(&extent) = global_shape.get(axis) else {
        return Err(TensorParallelError::AxisOutOfBounds {
            axis,
            rank: global_shape.len(),
        });
    };
    validate_two_way_extent(dimension, extent)?;
    let global_elements = checked_numel(global_shape)?;
    Ok((global_elements / 2, global_elements))
}

fn checked_numel(shape: &[usize]) -> Result<usize, TensorParallelError> {
    shape.iter().copied().try_fold(1usize, |elements, extent| {
        elements
            .checked_mul(extent)
            .ok_or(TensorParallelError::ElementCountOverflow)
    })
}

fn checked_product(
    _dimension: TensorParallelDimension,
    lhs: usize,
    rhs: usize,
) -> Result<usize, TensorParallelError> {
    lhs.checked_mul(rhs)
        .ok_or(TensorParallelError::ElementCountOverflow)
}

fn checked_double(elements: usize) -> Result<usize, TensorParallelError> {
    elements
        .checked_mul(2)
        .ok_or(TensorParallelError::ElementCountOverflow)
}

/// Failure while building a TP=2 linear/attention plan.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum TensorParallelError {
    /// Zero is reserved for generic unlabelled collectives.
    #[error("tensor-parallel operation identity zero is reserved")]
    ReservedOperationId,
    /// Operation identities must leave room for the semantic tag bits.
    #[error("tensor-parallel operation identity {found} exceeds maximum {maximum}")]
    OperationIdTooLarge {
        /// Largest losslessly encodable identity.
        maximum: u64,
        /// Rejected identity.
        found: u64,
    },
    /// The same semantic operation appeared twice in one plan.
    #[error("tensor-parallel operation identity {id:?} appears more than once")]
    DuplicateOperation {
        /// Repeated operation identity.
        id: TensorParallelId,
    },
    /// A tensor-parallel plan described no communication.
    #[error("a tensor-parallel plan must contain at least one operation")]
    NoOperations,
    /// Runtime `Dyn` selected a non-floating linear/attention dtype.
    #[error("dtype {dtype:?} cannot represent tensor-parallel linear or attention values")]
    UnsupportedTensorDType {
        /// Rejected runtime dtype.
        dtype: DTypeId,
    },
    /// A required TP=2 dimension did not divide evenly.
    #[error("{dimension:?} extent {extent} is not divisible by {ranks} tensor-parallel ranks")]
    NonDivisible {
        /// Semantic dimension that failed.
        dimension: TensorParallelDimension,
        /// Runtime extent.
        extent: usize,
        /// Required rank count.
        ranks: usize,
    },
    /// Runtime tensor axis was outside the supplied shape.
    #[error("tensor axis {axis} is outside rank {rank}")]
    AxisOutOfBounds {
        /// Rejected axis.
        axis: usize,
        /// Runtime tensor rank.
        rank: usize,
    },
    /// Shape cardinality could not be represented by `usize`.
    #[error("tensor-parallel element count overflows usize")]
    ElementCountOverflow,
    /// The underlying collective plan was invalid.
    #[error(transparent)]
    Plan(#[from] PlanError),
    /// Shared collective validation failed.
    #[error(transparent)]
    Collective(#[from] CollectiveError),
}
