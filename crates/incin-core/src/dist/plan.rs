//! Checked collective plans and cross-rank preflight agreement.
//!
//! A placement transition says *which* movement is legal. This module turns
//! that proof into an ordered descriptor containing every value a transport
//! must not infer independently: group, sequence, element and byte counts,
//! dtype, reduction, placements, stream, and dependency.

use alloc::{borrow::ToOwned, string::String, vec, vec::Vec};

use half::{bf16, f16};
use typenum::{B1, IsLessOrEqual, NonZero, U2, U4294967295, Unsigned};

use crate::dist::collective::{
    CollectiveDType, CollectiveError, CollectiveKind, CollectiveReductionDType, GroupId, StreamId,
    validate_collective_dtype, validate_collective_reduction,
};
use crate::dist::mesh::{
    DeviceMesh, LinkClass, MeshAxis, MeshId, ProcessLayout, TopologyFingerprint, ValidMesh,
};
use crate::dist::pipeline::{
    PipelineDType, PipelineSchedule, StaticPipelineSchedule, validate_microbatches,
};
use crate::dist::placement::{
    ConstPlacement, Partial, PartialReduction, Placement, PlacementAxis, PlacementKind,
    PlacementOn, Replicated, Sharded,
};
use crate::dist::rule::{
    DistributedError, LegalTransition, PlacementTransition, ShardDivisible, ShardRemainderPolicy,
};
use crate::exec::ReduceOp;
use crate::shapes::Dyn;
use crate::shapes::error::OperationKind;
use crate::shapes::error::ShapeError;
use crate::tensor::dtype::{BuiltinDType, ConstDType, DTypeId};

/// Monotonic position of a collective in one plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SequenceToken(u64);

impl SequenceToken {
    /// Zero-based sequence number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Stable semantic identity attached to one collective.
///
/// Counts, dtypes, and placements are not enough to distinguish two
/// same-shaped parameters. A higher-level planner such as data parallelism
/// assigns a tag so swapping those parameters changes the preflight hash
/// before either rank launches.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CollectiveTag(u64);

impl CollectiveTag {
    /// An unlabelled collective, used by the generic planner methods.
    pub const ANONYMOUS: Self = Self(0);

    /// Build a stable caller-defined identity.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Numeric identity included in the plan hash.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One fully checked transport launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveDescriptor {
    tag: CollectiveTag,
    group: GroupId,
    kind: CollectiveKind,
    input_elements: usize,
    output_elements: usize,
    input_bytes: usize,
    output_bytes: usize,
    dtype: DTypeId,
    source: PlacementKind,
    destination: PlacementKind,
    sequence: SequenceToken,
    stream: StreamId,
    depends_on: Option<SequenceToken>,
}

impl CollectiveDescriptor {
    /// Higher-level semantic identity included in preflight agreement.
    #[must_use]
    pub const fn tag(&self) -> CollectiveTag {
        self.tag
    }

    /// Ordered communicator identity and cardinality.
    #[must_use]
    pub const fn group(&self) -> GroupId {
        self.group
    }

    /// Collective operation and reduction semantics.
    #[must_use]
    pub const fn kind(&self) -> CollectiveKind {
        self.kind
    }

    /// Number of logical input elements at each rank.
    #[must_use]
    pub const fn input_elements(&self) -> usize {
        self.input_elements
    }

    /// Number of logical output elements at each rank.
    #[must_use]
    pub const fn output_elements(&self) -> usize {
        self.output_elements
    }

    /// Checked input byte count at each rank.
    #[must_use]
    pub const fn input_bytes(&self) -> usize {
        self.input_bytes
    }

    /// Checked output byte count at each rank.
    #[must_use]
    pub const fn output_bytes(&self) -> usize {
        self.output_bytes
    }

    /// Runtime projection of the static or `Dyn` dtype.
    #[must_use]
    pub const fn dtype(&self) -> DTypeId {
        self.dtype
    }

    /// Placement consumed by this movement.
    #[must_use]
    pub const fn source(&self) -> PlacementKind {
        self.source
    }

    /// Placement produced by this movement.
    #[must_use]
    pub const fn destination(&self) -> PlacementKind {
        self.destination
    }

    /// Monotonic launch position.
    #[must_use]
    pub const fn sequence(&self) -> SequenceToken {
        self.sequence
    }

    /// Logical communication stream.
    #[must_use]
    pub const fn stream(&self) -> StreamId {
        self.stream
    }

    /// Earlier collective that must complete before this launch.
    pub const fn depends_on(&self) -> Option<SequenceToken> {
        self.depends_on
    }
}

/// Immutable ordered plan tied to one physical mesh identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectivePlan {
    mesh: MeshId,
    descriptors: Vec<CollectiveDescriptor>,
    hash: u64,
}

impl CollectivePlan {
    /// Physical/logical mesh this plan was built against.
    #[must_use]
    pub const fn mesh_id(&self) -> MeshId {
        self.mesh
    }

    /// Collectives in launch order.
    #[must_use]
    pub fn descriptors(&self) -> &[CollectiveDescriptor] {
        &self.descriptors
    }

    /// Stable hash compared before any rank launches.
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Compact value exchanged during preflight.
    #[must_use]
    pub fn summary(&self) -> PlanSummary {
        PlanSummary {
            mesh: self.mesh,
            hash: self.hash,
            collectives: self.descriptors.len(),
        }
    }
}

/// Builder that assigns sequence tokens and derives all message counts.
#[derive(Debug)]
pub struct CollectivePlanBuilder<'a, M: ValidMesh> {
    mesh: &'a DeviceMesh<M>,
    descriptors: Vec<CollectiveDescriptor>,
}

/// Static placement/dtype pair that denotes an actual collective.
///
/// Local-only identity and replicated-to-sharded selection intentionally have
/// no implementation. A static caller cannot submit them to a collective
/// planner and wait for a runtime "no collective required" error.
#[diagnostic::on_unimplemented(
    message = "`{Self}` to `{To}` is not a collective supported for dtype `{K}`",
    label = "no statically valid collective transition",
    note = "integer mean reductions require a floating dtype; local-only placement changes do not belong in a collective plan"
)]
pub trait PlannedCollectiveTransition<K: CollectiveDType, To: Placement>:
    LegalTransition<To>
{
}

impl<K, M, Axis> PlannedCollectiveTransition<K, Replicated<M>> for Sharded<M, Axis>
where
    K: CollectiveDType,
    M: ValidMesh,
    Axis: PlacementAxis,
{
}

impl<K, M, R> PlannedCollectiveTransition<K, Replicated<M>> for Partial<M, R>
where
    K: CollectiveReductionDType<R>,
    M: ValidMesh,
    R: PartialReduction,
{
}

impl<K, M, R, Axis> PlannedCollectiveTransition<K, Sharded<M, Axis>> for Partial<M, R>
where
    K: CollectiveReductionDType<R>,
    M: ValidMesh,
    R: PartialReduction,
    Axis: PlacementAxis,
{
}

impl<'a, M: ValidMesh> CollectivePlanBuilder<'a, M> {
    /// Begin a plan for a bound mesh.
    #[must_use]
    pub fn new(mesh: &'a DeviceMesh<M>) -> Self {
        Self {
            mesh,
            descriptors: Vec::new(),
        }
    }

    /// Append a transition whose dtype and placements are compile-time known.
    ///
    /// Unsupported static dtypes and illegal transitions have no satisfying
    /// trait implementation and therefore fail to compile.
    pub fn push_static<K, From, To>(
        &mut self,
        axis: MeshAxis,
        rank: usize,
        input_elements: usize,
        stream: StreamId,
        depends_on: Option<SequenceToken>,
    ) -> Result<SequenceToken, PlanError>
    where
        K: ConstDType + BuiltinDType + CollectiveDType,
        From: ConstPlacement + PlannedCollectiveTransition<K, To> + PlacementOn<M>,
        To: ConstPlacement + PlacementOn<M>,
    {
        self.push_static_tagged::<K, From, To>(
            CollectiveTag::ANONYMOUS,
            axis,
            rank,
            input_elements,
            stream,
            depends_on,
        )
    }

    /// Append a statically checked transition with a semantic identity.
    ///
    /// The tag has no transport meaning; it exists so higher-level planners
    /// can make same-shaped operation reordering visible to preflight.
    #[allow(clippy::too_many_arguments)]
    pub fn push_static_tagged<K, From, To>(
        &mut self,
        tag: CollectiveTag,
        axis: MeshAxis,
        rank: usize,
        input_elements: usize,
        stream: StreamId,
        depends_on: Option<SequenceToken>,
    ) -> Result<SequenceToken, PlanError>
    where
        K: ConstDType + BuiltinDType + CollectiveDType,
        From: ConstPlacement + PlannedCollectiveTransition<K, To> + PlacementOn<M>,
        To: ConstPlacement + PlacementOn<M>,
    {
        self.push_checked(
            tag,
            axis,
            rank,
            input_elements,
            K::DTYPE,
            From::PLACEMENT,
            To::PLACEMENT,
            From::TRANSITION,
            stream,
            depends_on,
        )
    }

    /// Append the runtime-selected counterpart of [`push_static`](Self::push_static).
    #[allow(clippy::too_many_arguments)]
    pub fn push_dyn(
        &mut self,
        axis: MeshAxis,
        rank: usize,
        input_elements: usize,
        dtype: DTypeId,
        source: PlacementKind,
        destination: PlacementKind,
        stream: StreamId,
        depends_on: Option<SequenceToken>,
    ) -> Result<SequenceToken, PlanError> {
        self.push_dyn_tagged(
            CollectiveTag::ANONYMOUS,
            axis,
            rank,
            input_elements,
            dtype,
            source,
            destination,
            stream,
            depends_on,
        )
    }

    /// Append a runtime-checked transition with a semantic identity.
    #[allow(clippy::too_many_arguments)]
    pub fn push_dyn_tagged(
        &mut self,
        tag: CollectiveTag,
        axis: MeshAxis,
        rank: usize,
        input_elements: usize,
        dtype: DTypeId,
        source: PlacementKind,
        destination: PlacementKind,
        stream: StreamId,
        depends_on: Option<SequenceToken>,
    ) -> Result<SequenceToken, PlanError> {
        validate_collective_dtype(dtype)?;
        let transition = crate::dist::validate_transition(source, destination)?;
        self.push_checked(
            tag,
            axis,
            rank,
            input_elements,
            dtype,
            source,
            destination,
            transition,
            stream,
            depends_on,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn push_checked(
        &mut self,
        tag: CollectiveTag,
        axis: MeshAxis,
        rank: usize,
        input_elements: usize,
        dtype: DTypeId,
        source: PlacementKind,
        destination: PlacementKind,
        transition: PlacementTransition,
        stream: StreamId,
        depends_on: Option<SequenceToken>,
    ) -> Result<SequenceToken, PlanError> {
        validate_collective_dtype(dtype)?;
        let runtime_transition = crate::dist::validate_transition(source, destination)?;
        if runtime_transition != transition {
            return Err(PlanError::TransitionMismatch {
                typed: transition,
                runtime: runtime_transition,
            });
        }
        if let Some(dependency) = depends_on
            && dependency.0 >= self.descriptors.len() as u64
        {
            return Err(PlanError::UnknownDependency {
                dependency,
                next: self.descriptors.len(),
            });
        }

        let members = self
            .mesh
            .groups()
            .group(axis, rank)
            .ok_or(PlanError::RankOutOfRange {
                rank,
                world: M::WORLD,
            })?;
        let group = GroupId::new(group_token(self.mesh.id(), axis, &members), members.len())?;
        let kind = kind_for_transition(transition, source)?;
        if let CollectiveKind::AllReduce(op) | CollectiveKind::ReduceScatter(op) = kind {
            validate_collective_reduction(dtype, op)?;
        }
        if matches!(
            kind,
            CollectiveKind::AllGather | CollectiveKind::ReduceScatter(_)
        ) && axis != MeshAxis::Tensor
        {
            return Err(PlanError::WrongAxis {
                kind,
                expected: MeshAxis::Tensor,
                found: axis,
            });
        }
        let output_elements = output_elements(kind, input_elements, group.ranks())?;
        let input_bytes = dtype.size_bytes(input_elements, OperationKind::Storage)?;
        let output_bytes = dtype.size_bytes(output_elements, OperationKind::Storage)?;
        let sequence = SequenceToken(
            u64::try_from(self.descriptors.len()).map_err(|_| PlanError::SequenceOverflow)?,
        );

        self.descriptors.push(CollectiveDescriptor {
            tag,
            group,
            kind,
            input_elements,
            output_elements,
            input_bytes,
            output_bytes,
            dtype,
            source,
            destination,
            sequence,
            stream,
            depends_on,
        });
        Ok(sequence)
    }

    /// Append one globally described point-to-point transfer.
    ///
    /// Higher-level pipeline planning owns the static source/destination-stage
    /// proof. This crate-visible entry point owns the shared runtime
    /// invariants and descriptor hashing so static and dynamic pipeline paths
    /// cannot accidentally mint different wire contracts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_send_recv_tagged(
        &mut self,
        tag: CollectiveTag,
        rank: usize,
        input_elements: usize,
        dtype: DTypeId,
        source_rank: usize,
        destination_rank: usize,
        source_stage: usize,
        destination_stage: usize,
        stream: StreamId,
        depends_on: Option<SequenceToken>,
    ) -> Result<SequenceToken, PlanError> {
        validate_collective_dtype(dtype)?;
        if let Some(dependency) = depends_on
            && dependency.0 >= self.descriptors.len() as u64
        {
            return Err(PlanError::UnknownDependency {
                dependency,
                next: self.descriptors.len(),
            });
        }

        let members = self.mesh.groups().group(MeshAxis::Pipeline, rank).ok_or(
            PlanError::RankOutOfRange {
                rank,
                world: M::WORLD,
            },
        )?;
        let group = GroupId::new(
            group_token(self.mesh.id(), MeshAxis::Pipeline, &members),
            members.len(),
        )?;
        validate_peer("source", source_rank, group.ranks())?;
        validate_peer("destination", destination_rank, group.ranks())?;
        if source_rank == destination_rank {
            return Err(CollectiveError::SamePeer { rank: source_rank }.into());
        }

        let input_bytes = dtype.size_bytes(input_elements, OperationKind::Storage)?;
        let sequence = SequenceToken(
            u64::try_from(self.descriptors.len()).map_err(|_| PlanError::SequenceOverflow)?,
        );
        self.descriptors.push(CollectiveDescriptor {
            tag,
            group,
            kind: CollectiveKind::SendRecv {
                source: source_rank,
                destination: destination_rank,
            },
            input_elements,
            output_elements: input_elements,
            input_bytes,
            output_bytes: input_bytes,
            dtype,
            source: PlacementKind::PipelineStage {
                index: source_stage,
            },
            destination: PlacementKind::PipelineStage {
                index: destination_stage,
            },
            sequence,
            stream,
            depends_on,
        });
        Ok(sequence)
    }

    /// Freeze ordering and compute the stable preflight hash.
    #[must_use]
    pub fn finish(self) -> CollectivePlan {
        let hash = plan_hash(self.mesh.id(), &self.descriptors);
        CollectivePlan {
            mesh: self.mesh.id(),
            descriptors: self.descriptors,
            hash,
        }
    }
}

/// Compact plan identity exchanged by every rank before launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanSummary {
    mesh: MeshId,
    hash: u64,
    collectives: usize,
}

impl PlanSummary {
    /// Rebuild a summary received from another process.
    ///
    /// A reconstructed summary is only data. [`preflight`] is what compares
    /// all ranks and mints the [`AgreedPlan`] proof used by a transport.
    #[must_use]
    pub const fn from_parts(mesh: MeshId, hash: u64, collectives: usize) -> Self {
        Self {
            mesh,
            hash,
            collectives,
        }
    }

    /// Mesh identity.
    #[must_use]
    pub const fn mesh_id(self) -> MeshId {
        self.mesh
    }

    /// Stable descriptor hash.
    #[must_use]
    pub const fn hash(self) -> u64 {
        self.hash
    }

    /// Number of collective launches.
    #[must_use]
    pub const fn collective_count(self) -> usize {
        self.collectives
    }
}

/// Sealed result of all-rank plan agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgreedPlan {
    summary: PlanSummary,
    ranks: usize,
}

impl AgreedPlan {
    /// Agreed plan summary.
    #[must_use]
    pub const fn summary(self) -> PlanSummary {
        self.summary
    }

    /// Number of ranks participating in preflight.
    #[must_use]
    pub const fn ranks(self) -> usize {
        self.ranks
    }
}

/// Compare mesh, count, and hash before any collective launch.
pub fn preflight(
    expected_ranks: usize,
    summaries: &[PlanSummary],
) -> Result<AgreedPlan, PlanError> {
    if expected_ranks == 0 {
        return Err(PlanError::EmptyPreflight);
    }
    if summaries.len() != expected_ranks {
        return Err(PlanError::PreflightRankCount {
            expected: expected_ranks,
            found: summaries.len(),
        });
    }
    let expected = summaries[0];
    for (rank, &found) in summaries.iter().enumerate().skip(1) {
        if found.mesh != expected.mesh {
            return Err(PlanError::MeshMismatch {
                rank,
                expected: expected.mesh,
                found: found.mesh,
            });
        }
        if found.collectives != expected.collectives {
            return Err(PlanError::CollectiveCountMismatch {
                rank,
                expected: expected.collectives,
                found: found.collectives,
            });
        }
        if found.hash != expected.hash {
            return Err(PlanError::PlanHashMismatch {
                rank,
                expected: expected.hash,
                found: found.hash,
            });
        }
    }
    Ok(AgreedPlan {
        summary: expected,
        ranks: expected_ranks,
    })
}

/// Failures found while constructing or agreeing on collective plans.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// Shared collective contract rejected the descriptor.
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    /// Dynamic placement transition is not legal.
    #[error(transparent)]
    Distributed(#[from] DistributedError),
    /// Element or byte count overflowed.
    #[error(transparent)]
    Shape(#[from] ShapeError),
    /// Selected rank does not exist in the mesh.
    #[error("rank {rank} is outside a mesh with {world} ranks")]
    RankOutOfRange {
        /// Requested rank.
        rank: usize,
        /// Bound mesh cardinality.
        world: usize,
    },
    /// Identity/local-shard transitions require no transport launch.
    #[error("placement transition {transition:?} does not require a collective")]
    NoCollectiveRequired {
        /// Transition that should remain local.
        transition: PlacementTransition,
    },
    /// Reduction transition did not originate from `Partial`.
    #[error("placement transition {transition:?} requires a partial source, found {placement:?}")]
    MissingReduction {
        /// Transition requiring reduction semantics.
        transition: PlacementTransition,
        /// Actual source placement.
        placement: PlacementKind,
    },
    /// Static transition projection disagrees with the runtime rule table.
    #[error("typed transition {typed:?} disagrees with runtime transition {runtime:?}")]
    TransitionMismatch {
        /// Transition selected by the trait implementation.
        typed: PlacementTransition,
        /// Transition derived from runtime placement projections.
        runtime: PlacementTransition,
    },
    /// Placement movement was assigned to the wrong mesh communicator.
    #[error("collective {kind:?} requires the {expected:?} axis, found {found:?}")]
    WrongAxis {
        /// Planned operation.
        kind: CollectiveKind,
        /// Axis implied by the placement transition.
        expected: MeshAxis,
        /// Axis supplied by the caller.
        found: MeshAxis,
    },
    /// Dependency is not an earlier token in this plan.
    #[error("dependency token {dependency:?} is not earlier than next sequence {next}")]
    UnknownDependency {
        /// Rejected dependency.
        dependency: SequenceToken,
        /// Next zero-based sequence index.
        next: usize,
    },
    /// Descriptor count no longer fits a sequence token.
    #[error("collective sequence exceeds u64")]
    SequenceOverflow,
    /// No ranks participated in preflight.
    #[error("plan preflight requires at least one rank")]
    EmptyPreflight,
    /// Submitted summaries do not cover the expected world.
    #[error("plan preflight expected {expected} ranks, found {found}")]
    PreflightRankCount {
        /// Expected world size.
        expected: usize,
        /// Submitted summary count.
        found: usize,
    },
    /// A rank built its plan for a different physical/logical mesh.
    #[error("rank {rank} has mesh {found:?}, expected {expected:?}")]
    MeshMismatch {
        /// First disagreeing rank.
        rank: usize,
        /// Rank-zero mesh.
        expected: MeshId,
        /// Disagreeing mesh.
        found: MeshId,
    },
    /// A rank plans a different number of launches.
    #[error("rank {rank} plans {found} collectives, expected {expected}")]
    CollectiveCountMismatch {
        /// First disagreeing rank.
        rank: usize,
        /// Rank-zero count.
        expected: usize,
        /// Disagreeing count.
        found: usize,
    },
    /// Descriptor contents or ordering diverge.
    #[error("rank {rank} has plan hash {found:#x}, expected {expected:#x}")]
    PlanHashMismatch {
        /// First disagreeing rank.
        rank: usize,
        /// Rank-zero hash.
        expected: u64,
        /// Disagreeing hash.
        found: u64,
    },
}

fn kind_for_transition(
    transition: PlacementTransition,
    source: PlacementKind,
) -> Result<CollectiveKind, PlanError> {
    match transition {
        PlacementTransition::AllGather => Ok(CollectiveKind::AllGather),
        PlacementTransition::AllReduce | PlacementTransition::ReduceScatter => {
            let PlacementKind::Partial { reduction } = source else {
                return Err(PlanError::MissingReduction {
                    transition,
                    placement: source,
                });
            };
            if transition == PlacementTransition::AllReduce {
                Ok(CollectiveKind::AllReduce(reduction))
            } else {
                Ok(CollectiveKind::ReduceScatter(reduction))
            }
        }
        PlacementTransition::Identity | PlacementTransition::LocalShard => {
            Err(PlanError::NoCollectiveRequired { transition })
        }
    }
}

fn output_elements(kind: CollectiveKind, input: usize, ranks: usize) -> Result<usize, PlanError> {
    match kind {
        CollectiveKind::AllReduce(_)
        | CollectiveKind::AllToAll
        | CollectiveKind::SendRecv { .. } => {
            if matches!(kind, CollectiveKind::AllToAll) && !input.is_multiple_of(ranks) {
                return Err(CollectiveError::NonDivisible {
                    elements: input,
                    ranks,
                }
                .into());
            }
            Ok(input)
        }
        CollectiveKind::AllGather => input
            .checked_mul(ranks)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Storage,
                expression: "rank-local elements * collective ranks",
            })
            .map_err(Into::into),
        CollectiveKind::ReduceScatter(_) => {
            if !input.is_multiple_of(ranks) {
                return Err(CollectiveError::NonDivisible {
                    elements: input,
                    ranks,
                }
                .into());
            }
            Ok(input / ranks)
        }
    }
}

fn validate_peer(endpoint: &'static str, rank: usize, ranks: usize) -> Result<(), CollectiveError> {
    if rank >= ranks {
        Err(CollectiveError::PeerOutOfRange {
            endpoint,
            rank,
            ranks,
        })
    } else {
        Ok(())
    }
}

fn group_token(mesh: MeshId, axis: MeshAxis, members: &[usize]) -> u64 {
    let mut digest = StableDigest::new()
        .bytes(b"incin.collective.group.v1")
        .number(mesh.digest())
        .bytes(axis.name().as_bytes())
        .number(members.len() as u64);
    for &rank in members {
        digest = digest.number(rank as u64);
    }
    digest.finish()
}

fn plan_hash(mesh: MeshId, descriptors: &[CollectiveDescriptor]) -> u64 {
    let mut digest = StableDigest::new()
        .bytes(b"incin.collective.plan.v2")
        .number(mesh.digest())
        .number(descriptors.len() as u64);
    for descriptor in descriptors {
        digest = digest
            .number(descriptor.tag.get())
            .number(descriptor.group.token())
            .number(descriptor.group.ranks() as u64)
            .collective(descriptor.kind)
            .number(descriptor.input_elements as u64)
            .number(descriptor.output_elements as u64)
            .number(descriptor.input_bytes as u64)
            .number(descriptor.output_bytes as u64)
            .dtype(descriptor.dtype)
            .placement(descriptor.source)
            .placement(descriptor.destination)
            .number(descriptor.sequence.0)
            .number(u64::from(descriptor.stream.get()))
            .number(descriptor.depends_on.map_or(u64::MAX, |token| token.0));
    }
    digest.finish()
}

#[derive(Debug, Clone, Copy)]
struct StableDigest(u64);

impl StableDigest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        self = self.number(bytes.len() as u64);
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    fn number(self, value: u64) -> Self {
        self.bytes_raw(&value.to_le_bytes())
    }

    fn bytes_raw(mut self, bytes: &[u8]) -> Self {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    fn dtype(self, dtype: DTypeId) -> Self {
        self.bytes(dtype.name().as_bytes())
    }

    fn collective(self, kind: CollectiveKind) -> Self {
        match kind {
            CollectiveKind::AllReduce(op) => self.bytes(b"all-reduce").reduce(op),
            CollectiveKind::AllGather => self.bytes(b"all-gather"),
            CollectiveKind::ReduceScatter(op) => self.bytes(b"reduce-scatter").reduce(op),
            CollectiveKind::AllToAll => self.bytes(b"all-to-all"),
            CollectiveKind::SendRecv {
                source,
                destination,
            } => self
                .bytes(b"send-recv")
                .number(source as u64)
                .number(destination as u64),
        }
    }

    fn reduce(self, op: ReduceOp) -> Self {
        self.bytes(match op {
            ReduceOp::Sum => b"sum",
            ReduceOp::Mean => b"mean",
            ReduceOp::Max => b"max",
            ReduceOp::Min => b"min",
            ReduceOp::Prod => b"prod",
        })
    }

    fn placement(self, placement: PlacementKind) -> Self {
        match placement {
            PlacementKind::Local => self.bytes(b"local"),
            PlacementKind::Replicated => self.bytes(b"replicated"),
            PlacementKind::Sharded { axis } => self.bytes(b"sharded").number(axis as u64),
            PlacementKind::Partial { reduction } => self.bytes(b"partial").reduce(reduction),
            PlacementKind::PipelineStage { index } => {
                self.bytes(b"pipeline-stage").number(index as u64)
            }
        }
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

// ===========================================================================
// Two-rank hybrid strategy planning.
// ===========================================================================

/// A physical two-rank topology shared by DP=2, TP=2, and PP=2 candidates.
///
/// The topology intentionally contains no logical mesh degrees. The hybrid
/// planner compares several logical interpretations of the same two devices;
/// carrying the `MeshId` of whichever interpretation happened to be bound
/// first would bias that comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwoRankPlanningTopology {
    fingerprint: u64,
    link: LinkClass,
    transport: String,
    process_layout: ProcessLayout,
}

impl TwoRankPlanningTopology {
    /// Project a statically two-rank bound mesh into the planner's physical
    /// topology.
    ///
    /// A mesh whose type-level world is not exactly two does not satisfy this
    /// function's bound and therefore fails to compile.
    pub fn from_static_mesh<M>(mesh: &DeviceMesh<M>) -> Result<Self, HybridPlanError>
    where
        M: ValidMesh<World = U2>,
    {
        Self::from_fingerprint(mesh.fingerprint())
    }

    /// Validate a runtime-discovered topology for the `Dyn` planning path.
    pub fn from_fingerprint(fingerprint: &TopologyFingerprint) -> Result<Self, HybridPlanError> {
        if fingerprint.devices().len() != 2 {
            return Err(HybridPlanError::TopologyWorld {
                expected: 2,
                found: fingerprint.devices().len(),
            });
        }

        if let ProcessLayout::ProcessPerRank { world, .. } = fingerprint.layout()
            && *world != 2
        {
            return Err(HybridPlanError::ProcessWorld {
                expected: 2,
                found: *world,
            });
        }

        let forward = fingerprint
            .links()
            .iter()
            .find_map(|&(from, to, class)| (from == 0 && to == 1).then_some(class))
            .ok_or(HybridPlanError::MissingLink {
                from_rank: 0,
                to_rank: 1,
            })?;
        let backward = fingerprint
            .links()
            .iter()
            .find_map(|&(from, to, class)| (from == 1 && to == 0).then_some(class))
            .ok_or(HybridPlanError::MissingLink {
                from_rank: 1,
                to_rank: 0,
            })?;
        if !forward.reaches() {
            return Err(HybridPlanError::UnreachableLink {
                from_rank: 0,
                to_rank: 1,
            });
        }
        if !backward.reaches() {
            return Err(HybridPlanError::UnreachableLink {
                from_rank: 1,
                to_rank: 0,
            });
        }

        Ok(Self {
            fingerprint: fingerprint.digest(),
            link: core::cmp::max(forward, backward),
            transport: fingerprint.transport().library().to_owned(),
            process_layout: fingerprint.layout().clone(),
        })
    }

    /// Stable physical-topology identity used by the report.
    #[must_use]
    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Least direct of the two ordered links between ranks.
    #[must_use]
    pub const fn link(&self) -> LinkClass {
        self.link
    }

    /// Communication library reported by topology discovery.
    #[must_use]
    pub fn transport(&self) -> &str {
        &self.transport
    }

    /// Process layout included in the topology assumption.
    #[must_use]
    pub const fn process_layout(&self) -> &ProcessLayout {
        &self.process_layout
    }
}

/// Logical two-rank strategy considered by the hybrid planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParallelStrategyKind {
    /// Two replicas, each processing half of the batch.
    Data,
    /// One model whose selected dimensions are split across two ranks.
    Tensor,
    /// Two sequential model stages.
    Pipeline,
}

/// Set of strategies the automatic planner may consider.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StrategySet(u8);

impl StrategySet {
    /// No strategy.
    pub const NONE: Self = Self(0);
    /// Data parallelism.
    pub const DATA: Self = Self(1 << 0);
    /// Tensor parallelism.
    pub const TENSOR: Self = Self(1 << 1);
    /// Pipeline parallelism.
    pub const PIPELINE: Self = Self(1 << 2);
    /// Every strategy implemented by the two-rank planner.
    pub const ALL: Self = Self(Self::DATA.0 | Self::TENSOR.0 | Self::PIPELINE.0);

    /// Whether `strategy` belongs to this set.
    #[must_use]
    pub const fn contains(self, strategy: ParallelStrategyKind) -> bool {
        let bit = match strategy {
            ParallelStrategyKind::Data => Self::DATA.0,
            ParallelStrategyKind::Tensor => Self::TENSOR.0,
            ParallelStrategyKind::Pipeline => Self::PIPELINE.0,
        };
        self.0 & bit != 0
    }

    /// Union two sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether the set contains no strategies.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for StrategySet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl core::ops::BitOrAssign for StrategySet {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

/// Manual or automatic strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParallelStrategy {
    /// Require data parallelism.
    Data,
    /// Require tensor parallelism.
    Tensor,
    /// Require pipeline parallelism.
    Pipeline,
    /// Compare every allowed feasible strategy.
    Auto {
        /// Candidate set considered before feasibility filtering.
        allowed: StrategySet,
    },
}

/// Per-rank memory ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryLimit {
    /// Same absolute byte ceiling for both ranks.
    PerRankBytes(usize),
    /// Fraction of each device's discovered capacity available to the plan.
    ///
    /// The accepted range is `(0, 1]`. Capacity remains a runtime physical
    /// fact even when all tensor dimensions are static.
    PerDeviceFraction(f64),
}

/// Objective used to order feasible candidates.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanObjective {
    /// Analytical compute, communication, and pipeline-bubble estimate.
    #[default]
    MinimizeStepTime,
    /// Lowest maximum per-rank peak memory.
    MinimizeMemory,
    /// Lowest aggregate communication volume.
    MinimizeCommunication,
}

/// Runtime strategy and policy inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParallelOptions {
    /// Manual selection or allowed automatic search space.
    pub strategy: ParallelStrategy,
    /// Hard memory ceiling.
    pub memory_limit: MemoryLimit,
    /// Sharding behavior; the initial planner supports exact rejection only.
    pub remainder: ShardRemainderPolicy,
    /// Pipeline schedule used by the PP=2 candidate.
    pub schedule: PipelineSchedule,
    /// Candidate ordering objective.
    pub objective: PlanObjective,
}

/// Policy shared by the compile-time strategy entry points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticParallelOptions {
    /// Hard memory ceiling.
    pub memory_limit: MemoryLimit,
    /// Exact sharding policy.
    pub remainder: ShardRemainderPolicy,
    /// Candidate ordering objective, recorded in the report.
    pub objective: PlanObjective,
}

/// Runtime workload values needed by the analytical planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridWorkload {
    batch_size: usize,
    tensor_shard_extent: usize,
    parameter_elements: usize,
    activation_elements_per_microbatch: usize,
    microbatches: usize,
    optimizer_state_copies: usize,
    device_capacity_bytes: [usize; 2],
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

/// One exact logical shard recorded in a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardEvidence {
    field: WorkloadField,
    global: usize,
    per_rank: [usize; 2],
}

impl ShardEvidence {
    /// Sharded logical quantity.
    #[must_use]
    pub const fn field(self) -> WorkloadField {
        self.field
    }

    /// Global value before partitioning.
    #[must_use]
    pub const fn global(self) -> usize {
        self.global
    }

    /// Exact value assigned to each rank.
    #[must_use]
    pub const fn per_rank(self) -> [usize; 2] {
        self.per_rank
    }
}

/// Communication primitive modeled by a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanningCollectiveKind {
    /// Gradient all-reduce.
    AllReduce,
    /// Activation all-gather.
    AllGather,
    /// Gradient reduce-scatter paired with an all-gather.
    ReduceScatter,
    /// Pipeline point-to-point activation or gradient.
    SendRecv,
}

/// Exact aggregate communication contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommunicationEvidence {
    kind: PlanningCollectiveKind,
    launches: usize,
    bytes: usize,
}

impl CommunicationEvidence {
    /// Modeled primitive.
    #[must_use]
    pub const fn kind(self) -> PlanningCollectiveKind {
        self.kind
    }

    /// Number of logical launches per step.
    #[must_use]
    pub const fn launches(self) -> usize {
        self.launches
    }

    /// Aggregate payload bytes across all launches.
    #[must_use]
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

/// Why a strategy was absent from the feasible candidate set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrategyRejection {
    /// Automatic search set excluded the strategy.
    NotAllowed,
    /// A manual strategy was selected instead.
    NotSelected,
    /// Exact two-way sharding was impossible.
    NonDivisible {
        /// Quantity that did not divide.
        field: WorkloadField,
        /// Global value.
        value: usize,
        /// Required degree.
        degree: usize,
    },
    /// Peak memory crossed a hard rank-local limit.
    MemoryExceeded {
        /// First rank over its limit.
        rank: usize,
        /// Modeled peak.
        required: usize,
        /// Effective limit.
        limit: usize,
    },
}

/// One rejected strategy and the precise feasibility reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedStrategy {
    strategy: ParallelStrategyKind,
    reason: StrategyRejection,
}

impl RejectedStrategy {
    /// Rejected logical layout.
    #[must_use]
    pub const fn strategy(&self) -> ParallelStrategyKind {
        self.strategy
    }

    /// Rejection evidence.
    #[must_use]
    pub const fn reason(&self) -> &StrategyRejection {
        &self.reason
    }
}

/// Feasible strategy with inspectable analytical evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyCandidate {
    strategy: ParallelStrategyKind,
    dtype: DTypeId,
    shards: Vec<ShardEvidence>,
    collectives: Vec<CommunicationEvidence>,
    per_rank_peak_memory: [usize; 2],
    memory_limits: [usize; 2],
    communication_bytes: usize,
    estimated_step_cost: u128,
    topology_fingerprint: u64,
    link: LinkClass,
    transport: String,
    schedule: Option<PipelineSchedule>,
}

impl StrategyCandidate {
    /// Candidate logical layout.
    #[must_use]
    pub const fn strategy(&self) -> ParallelStrategyKind {
        self.strategy
    }

    /// Static projection or runtime-selected dtype.
    #[must_use]
    pub const fn dtype(&self) -> DTypeId {
        self.dtype
    }

    /// Exact logical partitions.
    #[must_use]
    pub fn shards(&self) -> &[ShardEvidence] {
        &self.shards
    }

    /// Modeled communication primitives.
    #[must_use]
    pub fn collectives(&self) -> &[CommunicationEvidence] {
        &self.collectives
    }

    /// Analytical peak bytes for ranks zero and one.
    #[must_use]
    pub const fn per_rank_peak_memory(&self) -> [usize; 2] {
        self.per_rank_peak_memory
    }

    /// Effective hard memory limit on both ranks.
    #[must_use]
    pub const fn memory_limits(&self) -> [usize; 2] {
        self.memory_limits
    }

    /// Aggregate logical payload bytes for one step.
    #[must_use]
    pub const fn communication_bytes(&self) -> usize {
        self.communication_bytes
    }

    /// Deterministic analytical score, not a measured duration.
    #[must_use]
    pub const fn estimated_step_cost(&self) -> u128 {
        self.estimated_step_cost
    }

    /// Stable physical topology assumed by this estimate.
    #[must_use]
    pub const fn topology_fingerprint(&self) -> u64 {
        self.topology_fingerprint
    }

    /// Least direct rank-to-rank path used by the estimate.
    #[must_use]
    pub const fn link(&self) -> LinkClass {
        self.link
    }

    /// Communication library assumed by the estimate.
    #[must_use]
    pub fn transport(&self) -> &str {
        &self.transport
    }

    /// Pipeline schedule for PP=2, or `None` for DP/TP.
    pub const fn schedule(&self) -> Option<PipelineSchedule> {
        self.schedule
    }
}

/// Inspectable result of hybrid feasibility filtering and objective ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridPlanReport {
    objective: PlanObjective,
    chosen: StrategyCandidate,
    feasible: Vec<StrategyCandidate>,
    pareto_frontier: Vec<ParallelStrategyKind>,
    rejected: Vec<RejectedStrategy>,
}

impl HybridPlanReport {
    /// Objective used for selection.
    #[must_use]
    pub const fn objective(&self) -> PlanObjective {
        self.objective
    }

    /// Selected feasible strategy.
    #[must_use]
    pub const fn chosen(&self) -> &StrategyCandidate {
        &self.chosen
    }

    /// Every feasible candidate in stable strategy order.
    #[must_use]
    pub fn feasible_candidates(&self) -> &[StrategyCandidate] {
        &self.feasible
    }

    /// Non-dominated strategies over memory, communication, and step score.
    #[must_use]
    pub fn pareto_frontier(&self) -> &[ParallelStrategyKind] {
        &self.pareto_frontier
    }

    /// Excluded or infeasible alternatives.
    #[must_use]
    pub fn rejected(&self) -> &[RejectedStrategy] {
        &self.rejected
    }
}

/// Stateless two-rank hybrid planner.
#[derive(Debug, Default, Clone, Copy)]
pub struct HybridPlanner;

impl HybridPlanner {
    /// Plan from runtime-resolved (`Dyn`) workload and dtype values.
    pub fn plan_dyn(
        topology: &TwoRankPlanningTopology,
        dtype: DTypeId,
        workload: HybridWorkload,
        options: ParallelOptions,
    ) -> Result<HybridPlanReport, HybridPlanError> {
        validate_hybrid_plan_dtype(dtype)?;
        validate_microbatches(workload.microbatches).map_err(|_| {
            HybridPlanError::ZeroWorkloadField {
                field: WorkloadField::Microbatches,
            }
        })?;
        if workload.microbatches > u32::MAX as usize {
            return Err(HybridPlanError::MicrobatchLimit {
                found: workload.microbatches,
                maximum: u32::MAX as usize,
            });
        }
        if options.remainder != ShardRemainderPolicy::Reject {
            return Err(HybridPlanError::UnsupportedRemainderPolicy {
                found: options.remainder,
            });
        }
        let limits = resolve_memory_limits(options.memory_limit, workload.device_capacity_bytes)?;
        let requested = requested_strategies(options.strategy)?;
        let mut feasible = Vec::new();
        let mut rejected = Vec::new();

        for strategy in [
            ParallelStrategyKind::Data,
            ParallelStrategyKind::Tensor,
            ParallelStrategyKind::Pipeline,
        ] {
            if !requested.contains(strategy) {
                rejected.push(RejectedStrategy {
                    strategy,
                    reason: match options.strategy {
                        ParallelStrategy::Auto { .. } => StrategyRejection::NotAllowed,
                        _ => StrategyRejection::NotSelected,
                    },
                });
                continue;
            }

            match build_candidate(
                strategy,
                topology,
                dtype,
                workload,
                options.schedule,
                limits,
            )? {
                Ok(candidate) => feasible.push(candidate),
                Err(reason) => rejected.push(RejectedStrategy { strategy, reason }),
            }
        }

        if feasible.is_empty() {
            return Err(HybridPlanError::NoFeasibleStrategy { rejected });
        }

        let pareto_frontier = pareto_frontier(&feasible);
        let chosen_index = choose_candidate(&feasible, options.objective);
        let chosen = feasible[chosen_index].clone();
        Ok(HybridPlanReport {
            objective: options.objective,
            chosen,
            feasible,
            pareto_frontier,
            rejected,
        })
    }

    /// Plan an automatic search whose logical values and dtype are all static.
    ///
    /// The bounds prove exact DP batch sharding, exact TP dimension and
    /// parameter sharding, nonzero bounded PP microbatches, and a floating
    /// dtype before runtime. Physical capacities and topology remain runtime
    /// observations and are still validated.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_auto_static<
        K,
        Batch,
        TensorExtent,
        Parameters,
        Activations,
        Microbatches,
        Schedule,
    >(
        topology: &TwoRankPlanningTopology,
        optimizer_state_copies: usize,
        device_capacity_bytes: [usize; 2],
        allowed: StrategySet,
        policy: StaticParallelOptions,
    ) -> Result<HybridPlanReport, HybridPlanError>
    where
        K: ConstDType + BuiltinDType + HybridPlanDType,
        Batch: Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        TensorExtent:
            Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Parameters:
            Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Activations: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Microbatches: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Schedule: StaticPipelineSchedule,
    {
        let workload = HybridWorkload::new(
            Batch::USIZE,
            TensorExtent::USIZE,
            Parameters::USIZE,
            Activations::USIZE,
            Microbatches::USIZE,
            optimizer_state_copies,
            device_capacity_bytes,
        )?;
        Self::plan_dyn(
            topology,
            K::DTYPE,
            workload,
            ParallelOptions {
                strategy: ParallelStrategy::Auto { allowed },
                memory_limit: policy.memory_limit,
                remainder: policy.remainder,
                schedule: Schedule::SCHEDULE,
                objective: policy.objective,
            },
        )
    }

    /// Require a statically valid DP=2 plan.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_data_static<K, Batch, Parameters, Activations, Microbatches>(
        topology: &TwoRankPlanningTopology,
        tensor_shard_extent: usize,
        optimizer_state_copies: usize,
        device_capacity_bytes: [usize; 2],
        policy: StaticParallelOptions,
    ) -> Result<HybridPlanReport, HybridPlanError>
    where
        K: ConstDType + BuiltinDType + HybridPlanDType,
        Batch: Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Parameters: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Activations: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Microbatches: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
    {
        let workload = HybridWorkload::new(
            Batch::USIZE,
            tensor_shard_extent,
            Parameters::USIZE,
            Activations::USIZE,
            Microbatches::USIZE,
            optimizer_state_copies,
            device_capacity_bytes,
        )?;
        plan_static_selected(
            topology,
            K::DTYPE,
            workload,
            ParallelStrategy::Data,
            PipelineSchedule::GPipe,
            policy,
        )
    }

    /// Require a statically valid TP=2 plan.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_tensor_static<K, TensorExtent, Parameters, Activations, Microbatches>(
        topology: &TwoRankPlanningTopology,
        batch_size: usize,
        optimizer_state_copies: usize,
        device_capacity_bytes: [usize; 2],
        policy: StaticParallelOptions,
    ) -> Result<HybridPlanReport, HybridPlanError>
    where
        K: ConstDType + BuiltinDType + HybridPlanDType,
        TensorExtent:
            Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Parameters:
            Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Activations: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Microbatches: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
    {
        let workload = HybridWorkload::new(
            batch_size,
            TensorExtent::USIZE,
            Parameters::USIZE,
            Activations::USIZE,
            Microbatches::USIZE,
            optimizer_state_copies,
            device_capacity_bytes,
        )?;
        plan_static_selected(
            topology,
            K::DTYPE,
            workload,
            ParallelStrategy::Tensor,
            PipelineSchedule::GPipe,
            policy,
        )
    }

    /// Require a statically valid PP=2 plan and schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_pipeline_static<K, Parameters, Activations, Microbatches, Schedule>(
        topology: &TwoRankPlanningTopology,
        batch_size: usize,
        tensor_shard_extent: usize,
        optimizer_state_copies: usize,
        device_capacity_bytes: [usize; 2],
        policy: StaticParallelOptions,
    ) -> Result<HybridPlanReport, HybridPlanError>
    where
        K: ConstDType + BuiltinDType + HybridPlanDType,
        Parameters:
            Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Activations: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Microbatches: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Schedule: StaticPipelineSchedule,
    {
        let workload = HybridWorkload::new(
            batch_size,
            tensor_shard_extent,
            Parameters::USIZE,
            Activations::USIZE,
            Microbatches::USIZE,
            optimizer_state_copies,
            device_capacity_bytes,
        )?;
        plan_static_selected(
            topology,
            K::DTYPE,
            workload,
            ParallelStrategy::Pipeline,
            Schedule::SCHEDULE,
            policy,
        )
    }
}

fn plan_static_selected(
    topology: &TwoRankPlanningTopology,
    dtype: DTypeId,
    workload: HybridWorkload,
    strategy: ParallelStrategy,
    schedule: PipelineSchedule,
    policy: StaticParallelOptions,
) -> Result<HybridPlanReport, HybridPlanError> {
    HybridPlanner::plan_dyn(
        topology,
        dtype,
        workload,
        ParallelOptions {
            strategy,
            memory_limit: policy.memory_limit,
            remainder: policy.remainder,
            schedule,
            objective: policy.objective,
        },
    )
}

fn requested_strategies(strategy: ParallelStrategy) -> Result<StrategySet, HybridPlanError> {
    let requested = match strategy {
        ParallelStrategy::Data => StrategySet::DATA,
        ParallelStrategy::Tensor => StrategySet::TENSOR,
        ParallelStrategy::Pipeline => StrategySet::PIPELINE,
        ParallelStrategy::Auto { allowed } => allowed,
    };
    if requested.is_empty() {
        Err(HybridPlanError::EmptyStrategySet)
    } else {
        Ok(requested)
    }
}

fn resolve_memory_limits(
    limit: MemoryLimit,
    capacities: [usize; 2],
) -> Result<[usize; 2], HybridPlanError> {
    match limit {
        MemoryLimit::PerRankBytes(bytes) => {
            if bytes == 0 {
                Err(HybridPlanError::ZeroMemoryLimit)
            } else {
                Ok([bytes; 2])
            }
        }
        MemoryLimit::PerDeviceFraction(fraction) => {
            if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
                return Err(HybridPlanError::InvalidMemoryFraction);
            }
            let first = (capacities[0] as f64 * fraction).floor() as usize;
            let second = (capacities[1] as f64 * fraction).floor() as usize;
            if first == 0 || second == 0 {
                Err(HybridPlanError::ZeroMemoryLimit)
            } else {
                Ok([first, second])
            }
        }
    }
}

fn build_candidate(
    strategy: ParallelStrategyKind,
    topology: &TwoRankPlanningTopology,
    dtype: DTypeId,
    workload: HybridWorkload,
    schedule: PipelineSchedule,
    limits: [usize; 2],
) -> Result<Result<StrategyCandidate, StrategyRejection>, HybridPlanError> {
    let parameter_bytes = dtype
        .size_bytes(workload.parameter_elements, OperationKind::Storage)
        .map_err(HybridPlanError::Shape)?;
    let activation_bytes = dtype
        .size_bytes(
            workload.activation_elements_per_microbatch,
            OperationKind::Storage,
        )
        .map_err(HybridPlanError::Shape)?;
    let optimizer_bytes = checked_mul(
        parameter_bytes,
        workload.optimizer_state_copies,
        "parameter bytes * optimizer state copies",
    )?;
    let full_model_memory = checked_sum(
        &[parameter_bytes, parameter_bytes, optimizer_bytes],
        "parameters + gradients + optimizer states",
    )?;
    let compute_work = checked_sum(
        &[
            parameter_bytes,
            checked_mul(
                activation_bytes,
                workload.microbatches,
                "activation bytes * microbatches",
            )?,
        ],
        "parameters + step activations",
    )?;

    let (shards, collectives, memory, communication, pipeline_schedule, bubble_cost) =
        match strategy {
            ParallelStrategyKind::Data => {
                if !workload.batch_size.is_multiple_of(2) {
                    return Ok(Err(StrategyRejection::NonDivisible {
                        field: WorkloadField::BatchSize,
                        value: workload.batch_size,
                        degree: 2,
                    }));
                }
                let memory = checked_add_array(
                    [full_model_memory; 2],
                    [activation_bytes; 2],
                    "DP persistent + live activation",
                )?;
                let communication =
                    checked_mul(parameter_bytes, 2, "two-rank gradient all-reduce payload")?;
                (
                    vec![
                        ShardEvidence {
                            field: WorkloadField::BatchSize,
                            global: workload.batch_size,
                            per_rank: [workload.batch_size / 2; 2],
                        },
                        ShardEvidence {
                            field: WorkloadField::ParameterElements,
                            global: workload.parameter_elements,
                            per_rank: [workload.parameter_elements; 2],
                        },
                    ],
                    vec![CommunicationEvidence {
                        kind: PlanningCollectiveKind::AllReduce,
                        launches: 1,
                        bytes: communication,
                    }],
                    memory,
                    communication,
                    None,
                    0,
                )
            }
            ParallelStrategyKind::Tensor => {
                if !workload.tensor_shard_extent.is_multiple_of(2) {
                    return Ok(Err(StrategyRejection::NonDivisible {
                        field: WorkloadField::TensorShardExtent,
                        value: workload.tensor_shard_extent,
                        degree: 2,
                    }));
                }
                if !workload.parameter_elements.is_multiple_of(2) {
                    return Ok(Err(StrategyRejection::NonDivisible {
                        field: WorkloadField::ParameterElements,
                        value: workload.parameter_elements,
                        degree: 2,
                    }));
                }
                let sharded_model = full_model_memory / 2;
                let memory = checked_add_array(
                    [sharded_model; 2],
                    [activation_bytes; 2],
                    "TP sharded model + gathered activation",
                )?;
                let one_collective = activation_bytes;
                let communication =
                    checked_mul(one_collective, 2, "TP all-gather + reduce-scatter payload")?;
                (
                    vec![
                        ShardEvidence {
                            field: WorkloadField::TensorShardExtent,
                            global: workload.tensor_shard_extent,
                            per_rank: [workload.tensor_shard_extent / 2; 2],
                        },
                        ShardEvidence {
                            field: WorkloadField::ParameterElements,
                            global: workload.parameter_elements,
                            per_rank: [workload.parameter_elements / 2; 2],
                        },
                    ],
                    vec![
                        CommunicationEvidence {
                            kind: PlanningCollectiveKind::AllGather,
                            launches: 1,
                            bytes: one_collective,
                        },
                        CommunicationEvidence {
                            kind: PlanningCollectiveKind::ReduceScatter,
                            launches: 1,
                            bytes: one_collective,
                        },
                    ],
                    memory,
                    communication,
                    None,
                    0,
                )
            }
            ParallelStrategyKind::Pipeline => {
                if !workload.parameter_elements.is_multiple_of(2) {
                    return Ok(Err(StrategyRejection::NonDivisible {
                        field: WorkloadField::ParameterElements,
                        value: workload.parameter_elements,
                        degree: 2,
                    }));
                }
                let stage_model = full_model_memory / 2;
                let live = match schedule {
                    PipelineSchedule::GPipe => [workload.microbatches; 2],
                    PipelineSchedule::OneForwardOneBackward => {
                        [core::cmp::min(workload.microbatches, 2), 1]
                    }
                };
                let live_bytes = [
                    checked_mul(activation_bytes, live[0], "stage-zero activation residency")?,
                    checked_mul(activation_bytes, live[1], "stage-one activation residency")?,
                ];
                let memory = checked_add_array(
                    [stage_model; 2],
                    live_bytes,
                    "PP stage model + live activations",
                )?;
                let launches = checked_mul(
                    workload.microbatches,
                    2,
                    "pipeline forward + backward launches",
                )?;
                let communication = checked_mul(
                    activation_bytes,
                    launches,
                    "pipeline activation payload * launches",
                )?;
                let useful_slots = checked_mul(
                    workload.microbatches,
                    4,
                    "two-stage forward/backward useful slots",
                )?;
                let bubble_cost = (compute_work as u128).saturating_mul(4) / useful_slots as u128;
                (
                    vec![
                        ShardEvidence {
                            field: WorkloadField::ParameterElements,
                            global: workload.parameter_elements,
                            per_rank: [workload.parameter_elements / 2; 2],
                        },
                        ShardEvidence {
                            field: WorkloadField::Microbatches,
                            global: workload.microbatches,
                            per_rank: [workload.microbatches; 2],
                        },
                    ],
                    vec![CommunicationEvidence {
                        kind: PlanningCollectiveKind::SendRecv,
                        launches,
                        bytes: communication,
                    }],
                    memory,
                    communication,
                    Some(schedule),
                    bubble_cost,
                )
            }
        };

    for rank in 0..2 {
        if memory[rank] > limits[rank] {
            return Ok(Err(StrategyRejection::MemoryExceeded {
                rank,
                required: memory[rank],
                limit: limits[rank],
            }));
        }
    }

    let link_weight = match topology.link {
        LinkClass::SameDevice => 1_u128,
        LinkClass::HighBandwidth => 2,
        LinkClass::PeerCapable => 3,
        LinkClass::HostBounce => 6,
        LinkClass::Network => 8,
        LinkClass::Unreachable => {
            return Err(HybridPlanError::UnreachableLink {
                from_rank: 0,
                to_rank: 1,
            });
        }
    };
    let estimated_step_cost = (compute_work as u128 / 2)
        .saturating_add((communication as u128).saturating_mul(link_weight))
        .saturating_add(bubble_cost);

    Ok(Ok(StrategyCandidate {
        strategy,
        dtype,
        shards,
        collectives,
        per_rank_peak_memory: memory,
        memory_limits: limits,
        communication_bytes: communication,
        estimated_step_cost,
        topology_fingerprint: topology.fingerprint,
        link: topology.link,
        transport: topology.transport.clone(),
        schedule: pipeline_schedule,
    }))
}

fn choose_candidate(candidates: &[StrategyCandidate], objective: PlanObjective) -> usize {
    let mut chosen = 0;
    for index in 1..candidates.len() {
        let candidate = &candidates[index];
        let current = &candidates[chosen];
        let candidate_key = objective_key(candidate, objective);
        let current_key = objective_key(current, objective);
        if candidate_key < current_key
            || (candidate_key == current_key && candidate.strategy < current.strategy)
        {
            chosen = index;
        }
    }
    chosen
}

fn objective_key(candidate: &StrategyCandidate, objective: PlanObjective) -> u128 {
    match objective {
        PlanObjective::MinimizeStepTime => candidate.estimated_step_cost,
        PlanObjective::MinimizeMemory => core::cmp::max(
            candidate.per_rank_peak_memory[0],
            candidate.per_rank_peak_memory[1],
        ) as u128,
        PlanObjective::MinimizeCommunication => candidate.communication_bytes as u128,
    }
}

fn pareto_frontier(candidates: &[StrategyCandidate]) -> Vec<ParallelStrategyKind> {
    let mut frontier = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let candidate_memory = core::cmp::max(
            candidate.per_rank_peak_memory[0],
            candidate.per_rank_peak_memory[1],
        );
        let dominated = candidates.iter().enumerate().any(|(other_index, other)| {
            if index == other_index {
                return false;
            }
            let other_memory =
                core::cmp::max(other.per_rank_peak_memory[0], other.per_rank_peak_memory[1]);
            let no_worse = other_memory <= candidate_memory
                && other.communication_bytes <= candidate.communication_bytes
                && other.estimated_step_cost <= candidate.estimated_step_cost;
            let strictly_better = other_memory < candidate_memory
                || other.communication_bytes < candidate.communication_bytes
                || other.estimated_step_cost < candidate.estimated_step_cost;
            no_worse && strictly_better
        });
        if !dominated {
            frontier.push(candidate.strategy);
        }
    }
    frontier
}

fn checked_mul(lhs: usize, rhs: usize, expression: &'static str) -> Result<usize, HybridPlanError> {
    lhs.checked_mul(rhs)
        .ok_or(HybridPlanError::ArithmeticOverflow { expression })
}

fn checked_sum(values: &[usize], expression: &'static str) -> Result<usize, HybridPlanError> {
    values.iter().try_fold(0_usize, |sum, &value| {
        sum.checked_add(value)
            .ok_or(HybridPlanError::ArithmeticOverflow { expression })
    })
}

fn checked_add_array(
    lhs: [usize; 2],
    rhs: [usize; 2],
    expression: &'static str,
) -> Result<[usize; 2], HybridPlanError> {
    Ok([
        lhs[0]
            .checked_add(rhs[0])
            .ok_or(HybridPlanError::ArithmeticOverflow { expression })?,
        lhs[1]
            .checked_add(rhs[1])
            .ok_or(HybridPlanError::ArithmeticOverflow { expression })?,
    ])
}

/// Hybrid topology, workload, policy, or feasibility failure.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum HybridPlanError {
    /// Runtime topology is not exactly two ranks.
    #[error("hybrid planner requires exactly {expected} devices, found {found}")]
    TopologyWorld {
        /// Required world.
        expected: usize,
        /// Discovered world.
        found: usize,
    },
    /// Process layout disagrees with the physical device count.
    #[error("hybrid planner requires process world {expected}, found {found}")]
    ProcessWorld {
        /// Required process world.
        expected: usize,
        /// Discovered process world.
        found: usize,
    },
    /// Topology fingerprint omitted an ordered rank link.
    #[error("topology has no link from rank {from_rank} to rank {to_rank}")]
    MissingLink {
        /// Sending rank.
        from_rank: usize,
        /// Receiving rank.
        to_rank: usize,
    },
    /// Topology has an explicitly unreachable ordered link.
    #[error("topology link from rank {from_rank} to rank {to_rank} is unreachable")]
    UnreachableLink {
        /// Sending rank.
        from_rank: usize,
        /// Receiving rank.
        to_rank: usize,
    },
    /// Static dtypes lack an implementation; `Dyn` reaches this variant.
    #[error("hybrid planning requires a floating dtype, found {dtype:?}")]
    UnsupportedDType {
        /// Runtime dtype.
        dtype: DTypeId,
    },
    /// Required logical workload field was zero.
    #[error("hybrid workload field {field:?} must be nonzero")]
    ZeroWorkloadField {
        /// Rejected field.
        field: WorkloadField,
    },
    /// Physical memory capacity was zero.
    #[error("rank {rank} reports zero device memory capacity")]
    ZeroDeviceCapacity {
        /// Rejected rank.
        rank: usize,
    },
    /// Absolute or resolved fractional memory limit was zero.
    #[error("memory limit must resolve to at least one byte per rank")]
    ZeroMemoryLimit,
    /// Fraction was non-finite, non-positive, or greater than one.
    #[error("per-device memory fraction must be finite and in (0, 1]")]
    InvalidMemoryFraction,
    /// Automatic selection was given no candidates.
    #[error("automatic planning requires at least one allowed strategy")]
    EmptyStrategySet,
    /// Padding and ragged sharding are not yet implemented.
    #[error("hybrid planner supports only ShardRemainderPolicy::Reject, found {found:?}")]
    UnsupportedRemainderPolicy {
        /// Requested policy.
        found: ShardRemainderPolicy,
    },
    /// Runtime microbatch count exceeds the tag/schedule representation.
    #[error("microbatch count {found} exceeds supported maximum {maximum}")]
    MicrobatchLimit {
        /// Rejected value.
        found: usize,
        /// Maximum accepted value.
        maximum: usize,
    },
    /// Checked storage sizing failed.
    #[error(transparent)]
    Shape(ShapeError),
    /// Planner-specific checked arithmetic failed.
    #[error("hybrid planning arithmetic overflow in {expression}")]
    ArithmeticOverflow {
        /// Expression that did not fit.
        expression: &'static str,
    },
    /// Every requested strategy failed feasibility.
    #[error("no feasible two-rank strategy")]
    NoFeasibleStrategy {
        /// Complete set of feasibility failures.
        rejected: Vec<RejectedStrategy>,
    },
}
