//! Compile-time tensor placements and their runtime projection.
//!
//! A placement is represented twice for the same reason a logical mesh has
//! both marker types and runtime degrees: the marker participates in trait
//! bounds, while [`PlacementKind`] is inspectable metadata carried by a
//! descriptor. Neither representation claims that a physical mesh exists.

#[cfg(feature = "distributed")]
use crate::exec::ReduceOp;
#[cfg(feature = "distributed")]
use alloc::vec::Vec;
#[cfg(feature = "distributed")]
use core::fmt;
use core::marker::PhantomData;
#[cfg(feature = "distributed")]
use typenum::Unsigned;

use crate::tensor::base::Dyn;

/// Runtime projection of a compile-time placement.
///
/// Mesh identity is deliberately absent. A placement typestate names a logical
/// mesh type, but `MeshId` is produced only after
/// that mesh is bound to physical devices. Putting a `MeshId` here would force
/// [`Placement::to_incin`] either to fabricate one or to read runtime state
/// from a static marker. Distributed descriptors pair this logical projection
/// with their separately validated bound mesh.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlacementKind {
    /// One tensor on one backend device.
    #[default]
    Local,
    /// The complete tensor is present at every rank in a mesh group.
    #[cfg(feature = "distributed")]
    Replicated,
    /// One tensor axis is partitioned into equal, contiguous shards.
    #[cfg(feature = "distributed")]
    Sharded {
        /// Zero-based tensor-axis index.
        axis: usize,
    },
    /// Each rank holds an incomplete reduction result.
    #[cfg(feature = "distributed")]
    Partial {
        /// The reduction needed to make the value complete.
        reduction: ReduceOp,
    },
    /// The tensor belongs to one stage of a pipeline mesh.
    #[cfg(feature = "distributed")]
    PipelineStage {
        /// Zero-based stage index.
        index: usize,
    },
}

impl PlacementKind {
    /// Whether this placement can be consumed as a complete tensor.
    #[must_use]
    pub const fn is_complete(self) -> bool {
        match self {
            #[cfg(feature = "distributed")]
            Self::Partial { .. } => false,
            _ => true,
        }
    }

    /// Whether this placement requires a logical mesh.
    #[must_use]
    pub const fn is_distributed(self) -> bool {
        match self {
            Self::Local => false,
            #[cfg(feature = "distributed")]
            _ => true,
        }
    }
}

/// Compile-time placement carried by storage and lowering rules.
pub trait Placement: 'static + Clone + core::fmt::Debug + Send + Sync {
    /// Runtime representation stored by a placement-bearing tensor.
    ///
    /// [`Local`] uses a zero-sized `PhantomData`; static distributed
    /// placements store only the runtime rank, and [`Dyn`] stores both rank
    /// and [`PlacementKind`]. This follows the same static/runtime split as
    /// tensor shape, dtype, and device without enlarging ordinary tensors.
    type Field: Clone + core::fmt::Debug + Default + Send + Sync;

    /// Build a checked stored field from runtime placement metadata.
    #[doc(hidden)]
    fn try_from_incin(kind: PlacementKind, rank: usize) -> Option<Self::Field>;

    /// Resolve the placement represented by a tensor field.
    fn to_incin(field: &Self::Field) -> PlacementKind;

    /// Resolve the rank represented by a tensor field.
    fn rank(field: &Self::Field) -> usize;

    /// Number of rank-local results this placement's mesh contains.
    #[doc(hidden)]
    const RANKS: usize = 1;

    /// Equal partitions of a sharded tensor axis.
    #[doc(hidden)]
    const SHARD_DEGREE: usize = 1;

    /// Stages in a pipeline placement's mesh.
    #[doc(hidden)]
    const PIPELINE_DEGREE: usize = 1;
}

/// A placement whose complete logical identity is known at compile time.
///
/// Runtime-selected [`Dyn`] placement deliberately does not implement this
/// trait. Rules needing a static proof require `ConstPlacement`; APIs that
/// accept `Dyn` validate its [`PlacementKind`] at their checked boundary.
pub trait ConstPlacement: Placement {
    /// Compile-time projection used by distributed lowering rules.
    const PLACEMENT: PlacementKind;
}

/// A distributed placement attached to one specific logical mesh.
///
/// Collective planning uses this bound to prevent a placement proved for one
/// mesh type from being inserted into a plan bound to another mesh.
#[cfg(feature = "distributed")]
pub trait PlacementOn<Mesh: crate::dist::mesh::ValidMesh>: Placement {}

/// Rank metadata for a compile-time-known distributed placement.
///
/// `Local` does not use this field and therefore pays no rank-storage cost.
#[doc(hidden)]
pub struct RankedPlacement<P> {
    rank: usize,
    marker: PhantomData<fn() -> P>,
}

impl<P> RankedPlacement<P> {
    fn new(rank: usize) -> Self {
        Self {
            rank,
            marker: PhantomData,
        }
    }
}

impl<P> Clone for RankedPlacement<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> Copy for RankedPlacement<P> {}

impl<P> Default for RankedPlacement<P> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<P> core::fmt::Debug for RankedPlacement<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RankedPlacement")
            .field("rank", &self.rank)
            .finish()
    }
}

/// Placement and rank metadata for `Tensor<..., P = Dyn>`.
#[doc(hidden)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynamicPlacement {
    kind: PlacementKind,
    rank: usize,
}

/// A tensor held by one backend device.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Local;

impl Placement for Local {
    type Field = PhantomData<Self>;

    fn try_from_incin(kind: PlacementKind, rank: usize) -> Option<Self::Field> {
        (kind == Self::PLACEMENT && rank == 0).then_some(PhantomData)
    }

    fn to_incin(_: &Self::Field) -> PlacementKind {
        Self::PLACEMENT
    }

    fn rank(_: &Self::Field) -> usize {
        0
    }
}

impl ConstPlacement for Local {
    const PLACEMENT: PlacementKind = PlacementKind::Local;
}

impl Placement for Dyn {
    type Field = DynamicPlacement;

    fn try_from_incin(kind: PlacementKind, rank: usize) -> Option<Self::Field> {
        Some(DynamicPlacement { kind, rank })
    }

    fn to_incin(field: &Self::Field) -> PlacementKind {
        field.kind
    }

    fn rank(field: &Self::Field) -> usize {
        field.rank
    }
}

/// A complete tensor copied across a logical mesh.
#[cfg(feature = "distributed")]
pub struct Replicated<Mesh>(PhantomData<fn() -> Mesh>);

/// A tensor partitioned along tensor axis `Axis` over a logical mesh.
#[cfg(feature = "distributed")]
pub struct Sharded<Mesh, Axis>(PhantomData<fn() -> (Mesh, Axis)>);

/// Local reduction results that still need a collective.
#[cfg(feature = "distributed")]
pub struct Partial<Mesh, Reduction>(PhantomData<fn() -> (Mesh, Reduction)>);

/// A tensor assigned to pipeline stage `INDEX`.
#[cfg(feature = "distributed")]
pub struct PipelineStage<Mesh, const INDEX: usize>(PhantomData<fn() -> Mesh>);

#[cfg(feature = "distributed")]
macro_rules! marker_impls {
    ($name:ident<$($parameter:ident),+>, $rendered:literal) => {
        impl<$($parameter),+> Copy for $name<$($parameter),+> {}

        impl<$($parameter),+> Clone for $name<$($parameter),+> {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<$($parameter),+> Default for $name<$($parameter),+> {
            fn default() -> Self {
                Self(PhantomData)
            }
        }

        impl<$($parameter),+> fmt::Debug for $name<$($parameter),+> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str($rendered)
            }
        }
    };
}

#[cfg(feature = "distributed")]
marker_impls!(Replicated<Mesh>, "Replicated");
#[cfg(feature = "distributed")]
marker_impls!(Sharded<Mesh, Axis>, "Sharded");
#[cfg(feature = "distributed")]
marker_impls!(Partial<Mesh, Reduction>, "Partial");

#[cfg(feature = "distributed")]
impl<Mesh, const INDEX: usize> Copy for PipelineStage<Mesh, INDEX> {}

#[cfg(feature = "distributed")]
impl<Mesh, const INDEX: usize> Clone for PipelineStage<Mesh, INDEX> {
    fn clone(&self) -> Self {
        *self
    }
}

#[cfg(feature = "distributed")]
impl<Mesh, const INDEX: usize> Default for PipelineStage<Mesh, INDEX> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[cfg(feature = "distributed")]
impl<Mesh, const INDEX: usize> fmt::Debug for PipelineStage<Mesh, INDEX> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PipelineStage<{INDEX}>")
    }
}

/// A type-level tensor-axis index usable by [`Sharded`].
#[cfg(feature = "distributed")]
pub trait PlacementAxis: 'static {
    /// Zero-based runtime projection.
    const INDEX: usize;
}

#[cfg(feature = "distributed")]
impl<Axis> PlacementAxis for Axis
where
    Axis: Unsigned + 'static,
{
    const INDEX: usize = Axis::USIZE;
}

/// A type-level reduction usable by [`Partial`].
#[cfg(feature = "distributed")]
pub trait PartialReduction: 'static {
    /// Runtime reduction descriptor.
    const OP: ReduceOp;
}

#[cfg(feature = "distributed")]
macro_rules! reduction_markers {
    ($(($name:ident, $op:ident)),+ $(,)?) => {
        $(
            #[doc = concat!("Type-level `", stringify!($op), "` reduction.")]
            #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
            pub struct $name;

            impl PartialReduction for $name {
                const OP: ReduceOp = ReduceOp::$op;
            }
        )+
    };
}

#[cfg(feature = "distributed")]
reduction_markers!(
    (Sum, Sum),
    (Mean, Mean),
    (Max, Max),
    (Min, Min),
    (Prod, Prod),
);

#[cfg(feature = "distributed")]
impl<Mesh> Placement for Replicated<Mesh>
where
    Mesh: crate::dist::mesh::ValidMesh,
{
    type Field = RankedPlacement<Self>;
    const RANKS: usize = Mesh::WORLD;

    fn try_from_incin(kind: PlacementKind, rank: usize) -> Option<Self::Field> {
        (kind == Self::PLACEMENT).then(|| RankedPlacement::new(rank))
    }

    fn to_incin(_: &Self::Field) -> PlacementKind {
        Self::PLACEMENT
    }

    fn rank(field: &Self::Field) -> usize {
        field.rank
    }
}

#[cfg(feature = "distributed")]
impl<Mesh> ConstPlacement for Replicated<Mesh>
where
    Mesh: crate::dist::mesh::ValidMesh,
{
    const PLACEMENT: PlacementKind = PlacementKind::Replicated;
}

#[cfg(feature = "distributed")]
impl<Mesh> PlacementOn<Mesh> for Replicated<Mesh> where Mesh: crate::dist::mesh::ValidMesh {}

#[cfg(feature = "distributed")]
impl<Mesh, Axis> Placement for Sharded<Mesh, Axis>
where
    Mesh: crate::dist::mesh::ValidMesh,
    Axis: PlacementAxis,
{
    type Field = RankedPlacement<Self>;
    const RANKS: usize = Mesh::WORLD;
    const SHARD_DEGREE: usize = Mesh::TENSOR;

    fn try_from_incin(kind: PlacementKind, rank: usize) -> Option<Self::Field> {
        (kind == Self::PLACEMENT).then(|| RankedPlacement::new(rank))
    }

    fn to_incin(_: &Self::Field) -> PlacementKind {
        Self::PLACEMENT
    }

    fn rank(field: &Self::Field) -> usize {
        field.rank
    }
}

#[cfg(feature = "distributed")]
impl<Mesh, Axis> ConstPlacement for Sharded<Mesh, Axis>
where
    Mesh: crate::dist::mesh::ValidMesh,
    Axis: PlacementAxis,
{
    const PLACEMENT: PlacementKind = PlacementKind::Sharded { axis: Axis::INDEX };
}

#[cfg(feature = "distributed")]
impl<Mesh, Axis> PlacementOn<Mesh> for Sharded<Mesh, Axis>
where
    Mesh: crate::dist::mesh::ValidMesh,
    Axis: PlacementAxis,
{
}

#[cfg(feature = "distributed")]
impl<Mesh, Reduction> Placement for Partial<Mesh, Reduction>
where
    Mesh: crate::dist::mesh::ValidMesh,
    Reduction: PartialReduction,
{
    type Field = RankedPlacement<Self>;
    const RANKS: usize = Mesh::WORLD;

    fn try_from_incin(kind: PlacementKind, rank: usize) -> Option<Self::Field> {
        (kind == Self::PLACEMENT).then(|| RankedPlacement::new(rank))
    }

    fn to_incin(_: &Self::Field) -> PlacementKind {
        Self::PLACEMENT
    }

    fn rank(field: &Self::Field) -> usize {
        field.rank
    }
}

#[cfg(feature = "distributed")]
impl<Mesh, Reduction> ConstPlacement for Partial<Mesh, Reduction>
where
    Mesh: crate::dist::mesh::ValidMesh,
    Reduction: PartialReduction,
{
    const PLACEMENT: PlacementKind = PlacementKind::Partial {
        reduction: Reduction::OP,
    };
}

#[cfg(feature = "distributed")]
impl<Mesh, Reduction> PlacementOn<Mesh> for Partial<Mesh, Reduction>
where
    Mesh: crate::dist::mesh::ValidMesh,
    Reduction: PartialReduction,
{
}

#[cfg(feature = "distributed")]
impl<Mesh, const INDEX: usize> Placement for PipelineStage<Mesh, INDEX>
where
    Mesh: crate::dist::mesh::ValidMesh,
{
    type Field = RankedPlacement<Self>;
    const RANKS: usize = Mesh::WORLD;
    const PIPELINE_DEGREE: usize = Mesh::PIPELINE;

    fn try_from_incin(kind: PlacementKind, rank: usize) -> Option<Self::Field> {
        (kind == Self::PLACEMENT).then(|| RankedPlacement::new(rank))
    }

    fn to_incin(_: &Self::Field) -> PlacementKind {
        Self::PLACEMENT
    }

    fn rank(field: &Self::Field) -> usize {
        field.rank
    }
}

#[cfg(feature = "distributed")]
impl<Mesh, const INDEX: usize> ConstPlacement for PipelineStage<Mesh, INDEX>
where
    Mesh: crate::dist::mesh::ValidMesh,
{
    const PLACEMENT: PlacementKind = PlacementKind::PipelineStage { index: INDEX };
}

#[cfg(feature = "distributed")]
impl<Mesh, const INDEX: usize> PlacementOn<Mesh> for PipelineStage<Mesh, INDEX> where
    Mesh: crate::dist::mesh::ValidMesh
{
}

/// Runtime placements of an operation's inputs.
///
/// This is an owned slice rather than a fixed array because operation arity is
/// descriptor-specific. Constructing one makes no proof claim; it becomes
/// trusted only after a distributed lowering rule checks it.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
#[cfg(feature = "distributed")]
pub struct PlacementBuf {
    placements: Vec<PlacementKind>,
}

#[cfg(feature = "distributed")]
impl PlacementBuf {
    /// Copy placement metadata from a slice.
    #[must_use]
    pub fn from_slice(placements: &[PlacementKind]) -> Self {
        Self {
            placements: placements.to_vec(),
        }
    }

    /// Number of input placements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.placements.len()
    }

    /// Whether no input placements were recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.placements.is_empty()
    }

    /// Borrow the placements in input order.
    #[must_use]
    pub fn as_slice(&self) -> &[PlacementKind] {
        &self.placements
    }
}

#[cfg(feature = "distributed")]
impl<const N: usize> From<[PlacementKind; N]> for PlacementBuf {
    fn from(placements: [PlacementKind; N]) -> Self {
        Self {
            placements: placements.into(),
        }
    }
}
