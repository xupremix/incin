//! Typed logical device meshes.
//!
//! PROPOSALS.md §3.8 splits distributed correctness into two proof domains. The
//! *logical* one — shard counts, placements, collective semantics, and
//! global/local shapes — is provable from the types alone. The *physical* one —
//! installed devices, rank mapping, memory, peer access, link topology,
//! transport versions — is not provable at all until a process looks at a
//! machine.
//!
//! This module is the logical half and only the logical half. A [`MeshSpec`]
//! says how many ranks the topology has along each axis and nothing about which
//! devices they are; `DeviceMesh::bind` and the topology fingerprint are
//! `DST-002`. §3.8 is explicit that the compile-time claim is *logical device
//! selection and validation*, never hardware existence, and a mesh type that
//! held a `DeviceId` would be making the second claim while checking the first.
//!
//! ```rust
//! use incin_core::dist::mesh::{Data, MeshSpec, Pipeline, TensorParallel, ValidMesh};
//! use incin_core::typenum::{U1, U3};
//!
//! /// Three-way tensor parallelism over three ranks.
//! type ThreeWayTensorMesh = MeshSpec<Data<U1>, TensorParallel<U3>, Pipeline<U1>>;
//!
//! assert_eq!(ThreeWayTensorMesh::WORLD, 3);
//! assert_eq!(ThreeWayTensorMesh::TENSOR, 3);
//! ```

use core::marker::PhantomData;
use core::ops::Mul;

use typenum::{NonZero, Prod, U1, Unsigned};

/// The batch axis: how many replicas of the whole model there are.
///
/// A degree of one means "not data parallel", which is why it is the default
/// for the other two axes rather than an absence.
pub struct Data<N>(PhantomData<N>);

/// The tensor-parallel axis: how many ranks one layer's weights are split over.
pub struct TensorParallel<N>(PhantomData<N>);

/// The pipeline axis: how many sequential stages the model is cut into.
pub struct Pipeline<N>(PhantomData<N>);

/// A logical parallel topology: data × tensor × pipeline.
///
/// The axes are positional and each position accepts only its own marker, so
/// `MeshSpec<Data<U1>, Pipeline<U3>, TensorParallel<U1>>` does not implement
/// [`ValidMesh`]. That matters because the swap is silent otherwise: it has the
/// same world size and a completely different meaning, and "three pipeline
/// stages" is not a typo away from "three-way tensor parallelism" in any
/// diagnostic that would print later.
///
/// Omitted axes default to one, matching the `mesh![dp = 3]` form `UX-002`
/// will expand to:
///
/// ```rust
/// use incin_core::dist::mesh::{Data, MeshSpec, ValidMesh};
/// use incin_core::typenum::U3;
///
/// assert_eq!(MeshSpec::<Data<U3>>::WORLD, 3);
/// assert_eq!(MeshSpec::<Data<U3>>::PIPELINE, 1);
/// ```
pub struct MeshSpec<DP, TP = TensorParallel<U1>, PP = Pipeline<U1>>(PhantomData<(DP, TP, PP)>);

/// A topology whose degrees are all nonzero and whose product is countable.
///
/// This is the whole compile-time contract of a mesh. It is implemented for
/// exactly one shape of type — three correctly ordered axis markers over
/// nonzero `typenum` degrees — so every way of being an invalid topology is the
/// absence of this implementation rather than a runtime check that something
/// has to remember to call.
///
/// [`World`] is an associated *type* rather than only a constant so that a
/// caller can bound on it. §3.8's example is three GPUs, for which `DP=3`,
/// `TP=3`, and `PP=3` are all valid and a rectangular `2 × 2` is not; a
/// `M: ValidMesh<World = U3>` bound is how that sentence becomes a compile
/// error instead of a comment.
///
/// ```rust
/// use incin_core::dist::mesh::{Data, MeshSpec, Pipeline, TensorParallel, ValidMesh};
/// use incin_core::typenum::{U1, U3};
///
/// fn on_three_ranks<M: ValidMesh<World = U3>>() -> usize {
///     M::WORLD
/// }
///
/// assert_eq!(on_three_ranks::<MeshSpec<Data<U3>>>(), 3);
/// assert_eq!(on_three_ranks::<MeshSpec<Data<U1>, TensorParallel<U3>>>(), 3);
/// assert_eq!(
///     on_three_ranks::<MeshSpec<Data<U1>, TensorParallel<U1>, Pipeline<U3>>>(),
///     3
/// );
/// ```
///
/// [`World`]: ValidMesh::World
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid logical mesh",
    label = "Invalid mesh topology",
    note = "A mesh is `MeshSpec<Data<DP>, TensorParallel<TP>, Pipeline<PP>>` with the axes in \
            that order and every degree nonzero"
)]
pub trait ValidMesh: 'static {
    /// The type-level world size, `DATA × TENSOR × PIPELINE`.
    ///
    /// Computed with the same `typenum` `Mul` the shape rules use, so a mesh
    /// and a shape agree on what multiplication means by construction rather
    /// than by review.
    type World: Unsigned;

    /// Replicas of the whole model.
    const DATA: usize;
    /// Ranks one layer's weights are split over.
    const TENSOR: usize;
    /// Sequential stages the model is cut into.
    const PIPELINE: usize;

    /// The number of ranks this topology describes.
    ///
    /// This is the count `DeviceMesh::bind` will hold a device list against in
    /// `DST-002`. It is a projection of [`World`] and never computed a second
    /// way, because a world size that two pieces of code derive separately is a
    /// world size they can disagree about.
    ///
    /// [`World`]: ValidMesh::World
    const WORLD: usize = <Self::World as Unsigned>::USIZE;
}

/// The one shape of type that is a mesh.
///
/// `NonZero` is `typenum`'s own marker and is not implemented for `UTerm`, so a
/// zero degree fails to satisfy the bound and the mesh has no implementation at
/// all. §3.8 requires "nonzero axes and checked `DP × TP × PP` multiplication
/// on stable Rust"; both are bounds here rather than assertions, so neither can
/// be reached by a program that compiled.
impl<DP, TP, PP> ValidMesh for MeshSpec<Data<DP>, TensorParallel<TP>, Pipeline<PP>>
where
    DP: Unsigned + NonZero + Mul<TP> + 'static,
    TP: Unsigned + NonZero + 'static,
    PP: Unsigned + NonZero + 'static,
    Prod<DP, TP>: Mul<PP>,
    Prod<Prod<DP, TP>, PP>: Unsigned,
{
    type World = Prod<Prod<DP, TP>, PP>;

    const DATA: usize = DP::USIZE;
    const TENSOR: usize = TP::USIZE;
    const PIPELINE: usize = PP::USIZE;
}
