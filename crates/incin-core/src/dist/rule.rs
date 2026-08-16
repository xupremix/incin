//! Logical placement rules and their proof-carrying result.
//!
//! This module proves only distribution facts: shard arithmetic, legal
//! placement transitions, and agreement between global and local shapes.
//! Physical device existence remains [`DeviceMesh::bind`](crate::dist::mesh::DeviceMesh::bind),
//! while collective ordering belongs to the distributed planner.

use crate::dist::mesh::ValidMesh;
use crate::dist::placement::{
    ConstPlacement, Local, Partial, PipelineStage, Placement, PlacementBuf, PlacementKind,
    Replicated, Sharded,
};
use crate::exec::ExecutionDescriptor;
use crate::shapes::ShapeBuf;
use crate::shapes::shape::Shape;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{Div, Rem};
use typenum::operator_aliases::{Mod, Quot};
use typenum::{NonZero, Same, U0, Unsigned};

/// Behavior requested when a dimension is not evenly divisible.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardRemainderPolicy {
    /// Refuse a non-integral shard.
    #[default]
    Reject,
    /// Pad to an equal partition and mask the padding.
    ///
    /// A rule must also define a neutral value before this can be used.
    PadAndMask,
    /// Use variable-length shards and variable-count collectives.
    ///
    /// This is reserved for a later transport capability.
    Ragged,
}

/// A compile-time proof that `Self` divides exactly by `Degree`.
#[diagnostic::on_unimplemented(
    message = "dimension `{Self}` cannot be evenly sharded by `{Degree}`",
    label = "non-integral local dimension",
    note = "static sharding requires a zero typenum remainder"
)]
pub trait ShardDivisible<Degree>: Unsigned {
    /// Type-level local extent, `Self / Degree`.
    type Local: Unsigned;

    /// Runtime projection of [`Local`](Self::Local).
    const LOCAL: usize = <Self::Local as Unsigned>::USIZE;
}

impl<Extent, Degree> ShardDivisible<Degree> for Extent
where
    Extent: Unsigned + Div<Degree> + Rem<Degree>,
    Degree: Unsigned + NonZero,
    Mod<Extent, Degree>: Same<U0, Output = U0>,
    Quot<Extent, Degree>: Unsigned,
{
    type Local = Quot<Extent, Degree>;
}

/// Communication or data movement justified by a placement transition.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlacementTransition {
    /// The placement is unchanged.
    Identity,
    /// Every rank selects its local shard from a replicated value.
    LocalShard,
    /// Shards are gathered into a complete value on every rank.
    AllGather,
    /// Partial values are reduced into a complete value on every rank.
    AllReduce,
    /// Partial values are reduced while the result remains sharded.
    ReduceScatter,
}

/// Compile-time proof that `Self` may become `To` directly.
#[diagnostic::on_unimplemented(
    message = "placement transition from `{Self}` to `{To}` is not legal",
    label = "no direct placement transition",
    note = "use an explicit replicated intermediate when no direct collective exists"
)]
pub trait LegalTransition<To: Placement>: Placement {
    /// Runtime projection of the proved transition.
    const TRANSITION: PlacementTransition;
}

impl LegalTransition<Local> for Local {
    const TRANSITION: PlacementTransition = PlacementTransition::Identity;
}

impl<Mesh> LegalTransition<Replicated<Mesh>> for Replicated<Mesh>
where
    Mesh: ValidMesh,
{
    const TRANSITION: PlacementTransition = PlacementTransition::Identity;
}

impl<Mesh, Axis> LegalTransition<Sharded<Mesh, Axis>> for Sharded<Mesh, Axis>
where
    Mesh: ValidMesh,
    Axis: crate::dist::placement::PlacementAxis,
{
    const TRANSITION: PlacementTransition = PlacementTransition::Identity;
}

impl<Mesh, Reduction> LegalTransition<Partial<Mesh, Reduction>> for Partial<Mesh, Reduction>
where
    Mesh: ValidMesh,
    Reduction: crate::dist::placement::PartialReduction,
{
    const TRANSITION: PlacementTransition = PlacementTransition::Identity;
}

impl<Mesh, const INDEX: usize> LegalTransition<PipelineStage<Mesh, INDEX>>
    for PipelineStage<Mesh, INDEX>
where
    Mesh: ValidMesh,
{
    const TRANSITION: PlacementTransition = PlacementTransition::Identity;
}

impl<Mesh, Axis> LegalTransition<Sharded<Mesh, Axis>> for Replicated<Mesh>
where
    Mesh: ValidMesh,
    Axis: crate::dist::placement::PlacementAxis,
{
    const TRANSITION: PlacementTransition = PlacementTransition::LocalShard;
}

impl<Mesh, Axis> LegalTransition<Replicated<Mesh>> for Sharded<Mesh, Axis>
where
    Mesh: ValidMesh,
    Axis: crate::dist::placement::PlacementAxis,
{
    const TRANSITION: PlacementTransition = PlacementTransition::AllGather;
}

impl<Mesh, Reduction> LegalTransition<Replicated<Mesh>> for Partial<Mesh, Reduction>
where
    Mesh: ValidMesh,
    Reduction: crate::dist::placement::PartialReduction,
{
    const TRANSITION: PlacementTransition = PlacementTransition::AllReduce;
}

impl<Mesh, Reduction, Axis> LegalTransition<Sharded<Mesh, Axis>> for Partial<Mesh, Reduction>
where
    Mesh: ValidMesh,
    Reduction: crate::dist::placement::PartialReduction,
    Axis: crate::dist::placement::PlacementAxis,
{
    const TRANSITION: PlacementTransition = PlacementTransition::ReduceScatter;
}

/// Placements that represent complete values.
///
/// [`Partial`] is intentionally absent. An operation requiring a complete
/// tensor expresses that fact as `P: CompletePlacement`, so a partial value is
/// rejected before lowering rather than checked by every consumer.
pub trait CompletePlacement: Placement {}

impl CompletePlacement for Local {}
impl<Mesh: ValidMesh> CompletePlacement for Replicated<Mesh> {}
impl<Mesh, Axis> CompletePlacement for Sharded<Mesh, Axis>
where
    Mesh: ValidMesh,
    Axis: crate::dist::placement::PlacementAxis,
{
}
impl<Mesh: ValidMesh, const INDEX: usize> CompletePlacement for PipelineStage<Mesh, INDEX> {}

/// Placement compatibility for local elementwise execution.
pub trait ElementwisePlacement<Rhs: Placement>: CompletePlacement {
    /// Placement of the elementwise result.
    type Output: CompletePlacement;
}

impl ElementwisePlacement<Local> for Local {
    type Output = Local;
}

impl<Mesh: ValidMesh> ElementwisePlacement<Replicated<Mesh>> for Replicated<Mesh> {
    type Output = Replicated<Mesh>;
}

impl<Mesh, Axis> ElementwisePlacement<Sharded<Mesh, Axis>> for Sharded<Mesh, Axis>
where
    Mesh: ValidMesh,
    Axis: crate::dist::placement::PlacementAxis,
{
    type Output = Sharded<Mesh, Axis>;
}

impl<Mesh, Axis> ElementwisePlacement<Replicated<Mesh>> for Sharded<Mesh, Axis>
where
    Mesh: ValidMesh,
    Axis: crate::dist::placement::PlacementAxis,
{
    type Output = Sharded<Mesh, Axis>;
}

impl<Mesh, Axis> ElementwisePlacement<Sharded<Mesh, Axis>> for Replicated<Mesh>
where
    Mesh: ValidMesh,
    Axis: crate::dist::placement::PlacementAxis,
{
    type Output = Sharded<Mesh, Axis>;
}

impl<Mesh: ValidMesh, const INDEX: usize> ElementwisePlacement<PipelineStage<Mesh, INDEX>>
    for PipelineStage<Mesh, INDEX>
{
    type Output = PipelineStage<Mesh, INDEX>;
}

/// Placement produced by reducing the axis a tensor is sharded over.
///
/// The result is intentionally [`Partial`]. Completing it requires one of the
/// legal all-reduce or reduce-scatter transitions above.
pub trait ReduceShardedAxis<Reduction>: CompletePlacement
where
    Reduction: crate::dist::placement::PartialReduction,
{
    /// Incomplete local reduction result.
    type Output: Placement;
}

impl<Mesh, Axis, Reduction> ReduceShardedAxis<Reduction> for Sharded<Mesh, Axis>
where
    Mesh: ValidMesh,
    Axis: crate::dist::placement::PlacementAxis,
    Reduction: crate::dist::placement::PartialReduction,
{
    type Output = Partial<Mesh, Reduction>;
}

/// Failures found while lowering logical distributed metadata.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributedError {
    /// A sharding request named no shards.
    #[error("cannot shard an extent into zero parts")]
    ZeroShards,
    /// A tensor axis does not exist.
    #[error("axis {axis} is outside rank {rank}")]
    AxisOutOfBounds {
        /// Requested axis.
        axis: usize,
        /// Tensor rank.
        rank: usize,
    },
    /// Equal shards cannot cover the global extent.
    #[error("axis {axis} extent {extent} is not divisible by {shards} shards")]
    NonDivisible {
        /// Tensor axis being sharded.
        axis: usize,
        /// Global axis extent.
        extent: usize,
        /// Requested shard count.
        shards: usize,
    },
    /// The selected remainder policy needs semantics or transport not present.
    #[error("shard remainder policy {policy:?} is not available")]
    UnsupportedRemainderPolicy {
        /// Requested policy.
        policy: ShardRemainderPolicy,
    },
    /// A stage index does not fit the pipeline degree.
    #[error("pipeline stage {index} is outside a pipeline with {stages} stages")]
    PipelineStageOutOfRange {
        /// Requested stage.
        index: usize,
        /// Pipeline degree.
        stages: usize,
    },
    /// The descriptor and typed global output disagree.
    #[error("operation output does not match the typed global output")]
    GlobalShapeMismatch,
    /// A distributed result had no rank-local shape.
    #[error("distributed output must contain at least one local shape")]
    NoLocalShapes,
    /// A local shape had a different rank from the global result.
    #[error("local shape {local} has rank {found}, expected {expected}")]
    LocalRankMismatch {
        /// Index into the local-shape list.
        local: usize,
        /// Global rank.
        expected: usize,
        /// Local rank.
        found: usize,
    },
    /// A placement produced the wrong number of rank-local shapes.
    #[error("placement {placement:?} requires {expected} local shapes, found {found}")]
    LocalShapeCount {
        /// Placement whose mesh fixed the count.
        placement: PlacementKind,
        /// Count derived from the logical mesh.
        expected: usize,
        /// Count supplied by the lowering input.
        found: usize,
    },
    /// A local extent disagreed with the placement rule.
    #[error("local shape {local} axis {axis} has extent {found}, expected {expected}")]
    LocalExtentMismatch {
        /// Index into the local-shape list.
        local: usize,
        /// Tensor axis.
        axis: usize,
        /// Expected local extent.
        expected: usize,
        /// Actual local extent.
        found: usize,
    },
    /// A unary transition rule received no input placement.
    #[error("a placement transition requires at least one input placement")]
    NoInputPlacements,
    /// An input did not have the typestate named by its rule.
    #[error("input {input} has placement {found:?}, expected {expected:?}")]
    UnexpectedInputPlacement {
        /// Input index.
        input: usize,
        /// Placement required by the rule.
        expected: PlacementKind,
        /// Placement supplied by the descriptor input.
        found: PlacementKind,
    },
    /// A runtime-selected placement transition has no legal static analogue.
    #[error("placement transition from {from:?} to {to:?} is not legal")]
    IllegalTransition {
        /// Runtime source placement.
        from: PlacementKind,
        /// Runtime destination placement.
        to: PlacementKind,
    },
}

/// Runtime counterpart of [`LegalTransition`].
///
/// Static placement pairs fail through trait resolution. A tensor whose
/// placement parameter is [`Dyn`](crate::shapes::Dyn) reaches this checked
/// path and receives the same transition vocabulary as a structured error.
pub fn validate_transition(
    from: PlacementKind,
    to: PlacementKind,
) -> Result<PlacementTransition, DistributedError> {
    let transition = match (from, to) {
        (PlacementKind::Local, PlacementKind::Local)
        | (PlacementKind::Replicated, PlacementKind::Replicated) => PlacementTransition::Identity,
        (PlacementKind::Sharded { axis: from_axis }, PlacementKind::Sharded { axis: to_axis })
            if from_axis == to_axis =>
        {
            PlacementTransition::Identity
        }
        (
            PlacementKind::Partial {
                reduction: from_reduction,
            },
            PlacementKind::Partial {
                reduction: to_reduction,
            },
        ) if from_reduction == to_reduction => PlacementTransition::Identity,
        (
            PlacementKind::PipelineStage { index: from_index },
            PlacementKind::PipelineStage { index: to_index },
        ) if from_index == to_index => PlacementTransition::Identity,
        (PlacementKind::Replicated, PlacementKind::Sharded { .. }) => {
            PlacementTransition::LocalShard
        }
        (PlacementKind::Sharded { .. }, PlacementKind::Replicated) => {
            PlacementTransition::AllGather
        }
        (PlacementKind::Partial { .. }, PlacementKind::Replicated) => {
            PlacementTransition::AllReduce
        }
        (PlacementKind::Partial { .. }, PlacementKind::Sharded { .. }) => {
            PlacementTransition::ReduceScatter
        }
        _ => return Err(DistributedError::IllegalTransition { from, to }),
    };
    Ok(transition)
}

/// Validate one equal local shard of a runtime-resolved global shape.
pub fn validate_shard(
    global: &ShapeBuf,
    axis: usize,
    shards: usize,
    policy: ShardRemainderPolicy,
) -> Result<ShapeBuf, DistributedError> {
    if shards == 0 {
        return Err(DistributedError::ZeroShards);
    }
    let Some(&extent) = global.dims().get(axis) else {
        return Err(DistributedError::AxisOutOfBounds {
            axis,
            rank: global.rank(),
        });
    };
    if extent % shards != 0 {
        return match policy {
            ShardRemainderPolicy::Reject => Err(DistributedError::NonDivisible {
                axis,
                extent,
                shards,
            }),
            ShardRemainderPolicy::PadAndMask | ShardRemainderPolicy::Ragged => {
                Err(DistributedError::UnsupportedRemainderPolicy { policy })
            }
        };
    }

    let mut local = global.dims().to_vec();
    local[axis] = extent / shards;
    Ok(ShapeBuf::from_slice(&local))
}

/// Validate a runtime pipeline index.
pub const fn validate_pipeline_stage(index: usize, stages: usize) -> Result<(), DistributedError> {
    if index < stages {
        Ok(())
    } else {
        Err(DistributedError::PipelineStageOutOfRange { index, stages })
    }
}

/// Untrusted inputs to a distributed lowering rule.
///
/// The global shape retains its frontend type `S`; local shapes and placements
/// are runtime metadata that the rule must check against it.
pub struct DistributedInputs<O, S>
where
    S: Shape,
{
    operation: O,
    marker: PhantomData<fn() -> S>,
    global_shape: ShapeBuf,
    local_shapes: Vec<ShapeBuf>,
    input_placements: PlacementBuf,
}

impl<O, S> DistributedInputs<O, S>
where
    S: Shape,
{
    /// Assemble metadata for validation.
    #[must_use]
    pub fn new(
        operation: O,
        global_shape: ShapeBuf,
        local_shapes: Vec<ShapeBuf>,
        input_placements: PlacementBuf,
    ) -> Self {
        Self {
            operation,
            marker: PhantomData,
            global_shape,
            local_shapes,
            input_placements,
        }
    }
}

/// A logical distributed operation together with the rule that proved it.
///
/// The fields are private and its constructor is crate-private. External
/// executors can inspect the result but cannot fabricate one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidatedDistributed<O> {
    operation: O,
    global_shape: ShapeBuf,
    local_shapes: Vec<ShapeBuf>,
    input_placements: PlacementBuf,
    output_placement: PlacementKind,
    transition: PlacementTransition,
}

impl<O> ValidatedDistributed<O> {
    pub(crate) fn new(
        operation: O,
        global_shape: ShapeBuf,
        local_shapes: Vec<ShapeBuf>,
        input_placements: PlacementBuf,
        output_placement: PlacementKind,
        transition: PlacementTransition,
    ) -> Self {
        Self {
            operation,
            global_shape,
            local_shapes,
            input_placements,
            output_placement,
            transition,
        }
    }

    /// Resolved operation descriptor.
    #[must_use]
    pub const fn operation(&self) -> &O {
        &self.operation
    }

    /// Global result shape before placement.
    #[must_use]
    pub const fn global_shape(&self) -> &ShapeBuf {
        &self.global_shape
    }

    /// One validated shape per rank-local result.
    #[must_use]
    pub fn local_shapes(&self) -> &[ShapeBuf] {
        &self.local_shapes
    }

    /// Runtime projections of the input typestates.
    #[must_use]
    pub const fn input_placements(&self) -> &PlacementBuf {
        &self.input_placements
    }

    /// Runtime projection of the output typestate.
    #[must_use]
    pub const fn output_placement(&self) -> PlacementKind {
        self.output_placement
    }

    /// Data movement proved necessary by the placement change.
    #[must_use]
    pub const fn transition(&self) -> PlacementTransition {
        self.transition
    }
}

impl<O: fmt::Debug> fmt::Display for ValidatedDistributed<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} [{:?} -> {:?}]",
            self.operation, self.input_placements, self.output_placement
        )
    }
}

/// Lowering contract for a logical distributed operation.
pub trait DistributedRule<Inputs> {
    /// Typed global result.
    type GlobalOutput: Shape;
    /// Proved output placement.
    type OutputPlacement: Placement;
    /// Backend-neutral operation descriptor.
    type Descriptor: ExecutionDescriptor;

    /// Validate runtime metadata against the typed placement and shape rules.
    fn lower_distributed(
        inputs: &Inputs,
    ) -> Result<ValidatedDistributed<Self::Descriptor>, DistributedError>;
}

/// Unary placement transition checked against global and local shapes.
pub struct PlacementTransitionRule<From, To>(PhantomData<fn() -> (From, To)>);

impl<From, To> PlacementTransitionRule<From, To> {
    /// Apply this transition's distributed lowering rule.
    pub fn lower<O, S>(
        inputs: &DistributedInputs<O, S>,
    ) -> Result<ValidatedDistributed<O>, DistributedError>
    where
        O: ExecutionDescriptor,
        S: Shape,
        From: LegalTransition<To> + ConstPlacement,
        To: ConstPlacement,
    {
        <Self as DistributedRule<DistributedInputs<O, S>>>::lower_distributed(inputs)
    }
}

impl<O, S, From, To> DistributedRule<DistributedInputs<O, S>> for PlacementTransitionRule<From, To>
where
    O: ExecutionDescriptor,
    S: Shape,
    From: LegalTransition<To> + ConstPlacement,
    To: ConstPlacement,
{
    type GlobalOutput = S;
    type OutputPlacement = To;
    type Descriptor = O;

    fn lower_distributed(
        inputs: &DistributedInputs<O, S>,
    ) -> Result<ValidatedDistributed<O>, DistributedError> {
        let global = inputs.global_shape.clone();
        if inputs.operation.output_shape() != Some(&global) {
            return Err(DistributedError::GlobalShapeMismatch);
        }
        if inputs.input_placements.is_empty() {
            return Err(DistributedError::NoInputPlacements);
        }
        let expected_input = From::PLACEMENT;
        for (input, &found) in inputs.input_placements.as_slice().iter().enumerate() {
            if found != expected_input {
                return Err(DistributedError::UnexpectedInputPlacement {
                    input,
                    expected: expected_input,
                    found,
                });
            }
        }

        let output = To::PLACEMENT;
        validate_placement::<From>(expected_input)?;
        validate_placement::<To>(output)?;
        validate_local_shapes::<To>(&global, &inputs.local_shapes, output)?;

        Ok(ValidatedDistributed::new(
            inputs.operation.clone(),
            global,
            inputs.local_shapes.clone(),
            inputs.input_placements.clone(),
            output,
            From::TRANSITION,
        ))
    }
}

fn validate_placement<P: Placement>(placement: PlacementKind) -> Result<(), DistributedError> {
    if let PlacementKind::PipelineStage { index } = placement {
        validate_pipeline_stage(index, P::PIPELINE_DEGREE)?;
    }
    Ok(())
}

fn validate_local_shapes<P: Placement>(
    global: &ShapeBuf,
    locals: &[ShapeBuf],
    placement: PlacementKind,
) -> Result<(), DistributedError> {
    if locals.is_empty() {
        return Err(DistributedError::NoLocalShapes);
    }
    let expected_count = match placement {
        PlacementKind::Local => 1,
        PlacementKind::PipelineStage { .. } => P::RANKS / P::PIPELINE_DEGREE,
        PlacementKind::Replicated
        | PlacementKind::Sharded { .. }
        | PlacementKind::Partial { .. } => P::RANKS,
    };
    if locals.len() != expected_count {
        return Err(DistributedError::LocalShapeCount {
            placement,
            expected: expected_count,
            found: locals.len(),
        });
    }
    for (local_index, local) in locals.iter().enumerate() {
        if local.rank() != global.rank() {
            return Err(DistributedError::LocalRankMismatch {
                local: local_index,
                expected: global.rank(),
                found: local.rank(),
            });
        }
    }

    let sharded_axis = match placement {
        PlacementKind::Sharded { axis } => Some(axis),
        PlacementKind::Local
        | PlacementKind::Replicated
        | PlacementKind::Partial { .. }
        | PlacementKind::PipelineStage { .. } => None,
    };

    if let Some(axis) = sharded_axis {
        let expected_local =
            validate_shard(global, axis, P::SHARD_DEGREE, ShardRemainderPolicy::Reject)?;
        for (local_index, local) in locals.iter().enumerate() {
            for (axis_index, (&found, &expected)) in local
                .dims()
                .iter()
                .zip(expected_local.dims().iter())
                .enumerate()
            {
                if found != expected {
                    return Err(DistributedError::LocalExtentMismatch {
                        local: local_index,
                        axis: axis_index,
                        expected,
                        found,
                    });
                }
            }
        }
    } else {
        for (local_index, local) in locals.iter().enumerate() {
            for (axis, (&found, &expected)) in
                local.dims().iter().zip(global.dims().iter()).enumerate()
            {
                if found != expected {
                    return Err(DistributedError::LocalExtentMismatch {
                        local: local_index,
                        axis,
                        expected,
                        found,
                    });
                }
            }
        }
    }

    Ok(())
}
