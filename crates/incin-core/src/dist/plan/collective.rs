//! Checked collective descriptors, plan construction, and the plan builder.

use super::*;

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
