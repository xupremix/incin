//! Typed device meshes: the logical topology and its physical binding.
//!
//! PROPOSALS.md §3.8 splits distributed correctness into two proof domains. The
//! *logical* one - shard counts, placements, collective semantics, and
//! global/local shapes - is provable from the types alone. The *physical* one -
//! installed devices, rank mapping, memory, peer access, link topology,
//! transport versions - is not provable at all until a process looks at a
//! machine.
//!
//! Both halves are in this file and the boundary between them is the point. A
//! [`MeshSpec`] says how many ranks the topology has along each axis and
//! nothing about which devices they are; it is checked by the compiler and
//! holds no device. A [`DeviceMesh`] is what a [`MeshSpec`] becomes once
//! [`DeviceMesh::bind`] has held real devices against it, and it exists only at
//! runtime. §3.8 is explicit that the compile-time claim is *logical device
//! selection and validation*, never hardware existence, so a single type that
//! made both claims would be making the second one while checking the first.
//! Keeping the two next to each other is how that stays visible.
//!
//! Nothing here reads hardware. Every question binding asks a machine goes
//! through [`TopologyProbe`], whose implementors live with the backends
//! (`DST-005`, `DST-006`) because `incin-core` is `no_std` and talks to no
//! device. Everything this module decides is a pure function of that trait's
//! answers, which is what lets the evidence suite bind a three-GPU mesh on a
//! CI runner with no GPUs.
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

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::ops::Mul;

use typenum::{NonZero, Prod, U1, Unsigned};

use crate::tensor::device::{DeviceId, DeviceKind};

pub use incin_macros::mesh;

// ===========================================================================
// The logical half: proved by the compiler, holds no device.
// ===========================================================================

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
/// exactly one shape of type - three correctly ordered axis markers over
/// nonzero `typenum` degrees - so every way of being an invalid topology is the
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
    /// This is the count [`DeviceMesh::bind`] holds a device list against. It
    /// is a projection of [`World`] and never computed a second
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

// ===========================================================================
// The physical half: proved by looking at a machine, holds no compile-time
// claim. Everything below is a pure function of `TopologyProbe`'s answers.
// ===========================================================================

/// Every question binding asks a machine, and the only impure surface in this
/// module.
///
/// §2.11's physical-proof list - installed devices, rank mapping, peer access,
/// link topology, transport versions, process layout - is exactly the set of
/// things a type cannot know. This trait is that set, and `DeviceMesh::bind`
/// consults nothing else, so every rule in this file can be exercised against
/// a machine that does not exist. That is `UX-014`'s
/// `Host` seam (in the `incin` facade's `doctor` module) applied to the same
/// problem for the same reason: the evidence for a distributed rule has to run
/// on a CI runner with no GPUs, or it is not evidence, it is a description.
///
/// Implementors ship with the backends, not here. Answering "what is the link
/// class between CUDA 0 and CUDA 3" means calling CUDA, and `incin-core` is
/// `no_std` and links no driver. `DST-005` implements it for the deterministic
/// CPU reference transport and `DST-006` for NCCL.
pub trait TopologyProbe {
    /// The device's persistent identity, or `None` if the probe cannot see it.
    ///
    /// `None` is the honest answer for an ordinal that names no installed
    /// device, and it is a different failure from a device that exists and is
    /// unsuitable: the first is [`BindError::UnknownDevice`], the second is
    /// one of the agreement guards.
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity>;

    /// How two devices reach each other.
    ///
    /// Asked in both directions is not the same as asked once - a link can be
    /// asymmetric - so `bind` asks for the pair it is about to require and
    /// does not assume the reverse.
    fn link(&self, from: DeviceId, to: DeviceId) -> LinkClass;

    /// The communication library this process would use.
    fn transport(&self) -> TransportVersion;

    /// How ranks are distributed over processes.
    fn layout(&self) -> ProcessLayout;
}

/// A device's identity, which is deliberately more than its ordinal.
///
/// §2.11: "Device ordinal alone is not a valid persistent identity." It is not
/// stable across reboots, it is renumbered by `CUDA_VISIBLE_DEVICES`, and two
/// processes with different visibility masks will disagree about which
/// physical card `1` is while both believing they agree. So the ordinal is
/// kept - it is what the caller asked for and what a diagnostic has to print -
/// and the vendor-stable id is what identity actually *means* here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceIdentity {
    /// The ordinal the caller named this device by.
    device: DeviceId,
    /// The vendor-stable identifier: a CUDA UUID, a PCI address, or whatever
    /// the backend can promise survives a process restart.
    persistent: String,
    /// The compute architecture, e.g. `sm_90`. Distinct from the backend
    /// family: two CUDA devices can be different architectures, and a mesh
    /// spanning both is a mesh whose ranks do not run the same kernels.
    architecture: String,
}

impl DeviceIdentity {
    /// Builds an identity from a probe's answer.
    #[must_use]
    pub fn new(device: DeviceId, persistent: String, architecture: String) -> Self {
        Self {
            device,
            persistent,
            architecture,
        }
    }

    /// The ordinal the caller named this device by.
    #[must_use]
    pub const fn device(&self) -> DeviceId {
        self.device
    }

    /// The vendor-stable identifier.
    #[must_use]
    pub fn persistent(&self) -> &str {
        &self.persistent
    }

    /// The compute architecture.
    #[must_use]
    pub fn architecture(&self) -> &str {
        &self.architecture
    }
}

/// How directly two devices can reach each other.
///
/// The variants are ordered by directness so that a requirement can be stated
/// as a minimum rather than as a set, and [`LinkClass::reaches`] is the
/// predicate `bind` uses. §2.11 lists peer access and link topology as
/// separate physical facts; they are one classification here because every
/// decision made from them is "is this good enough for the collective I am
/// about to place on it".
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LinkClass {
    /// The two ordinals are the same device.
    SameDevice,
    /// A vendor high-bandwidth interconnect, e.g. NVLink.
    HighBandwidth,
    /// PCIe with peer access enabled.
    PeerCapable,
    /// PCIe without peer access: transfers stage through host memory.
    HostBounce,
    /// A network fabric between separate hosts.
    Network,
    /// No path at all.
    Unreachable,
}

impl LinkClass {
    /// Whether any transfer is possible over this link.
    ///
    /// Only [`Unreachable`](LinkClass::Unreachable) is false. A slow link is a
    /// performance problem and this is a correctness predicate - `bind`
    /// refuses meshes that cannot communicate, not meshes that communicate
    /// badly, because the second is a judgement no library should silently
    /// make on a caller's behalf.
    #[must_use]
    pub const fn reaches(self) -> bool {
        !matches!(self, Self::Unreachable)
    }

    /// The stable name used in diagnostics and in the fingerprint digest.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SameDevice => "same-device",
            Self::HighBandwidth => "high-bandwidth",
            Self::PeerCapable => "peer-capable",
            Self::HostBounce => "host-bounce",
            Self::Network => "network",
            Self::Unreachable => "unreachable",
        }
    }
}

/// The communication library and version this process would use.
///
/// §2.11 puts transport/library versions in the fingerprint because two ranks
/// on mismatched versions can complete a handshake and then disagree about
/// wire format. Recording it does not prevent that; it makes the resulting
/// [`MeshId`] differ, which is what turns a silent corruption into a mismatch
/// two processes can notice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportVersion {
    library: String,
    major: u32,
    minor: u32,
    patch: u32,
}

impl TransportVersion {
    /// Records a library and its version triple.
    #[must_use]
    pub fn new(library: String, major: u32, minor: u32, patch: u32) -> Self {
        Self {
            library,
            major,
            minor,
            patch,
        }
    }

    /// The library name, e.g. `nccl`.
    #[must_use]
    pub fn library(&self) -> &str {
        &self.library
    }

    /// The version triple.
    #[must_use]
    pub const fn version(&self) -> (u32, u32, u32) {
        (self.major, self.minor, self.patch)
    }
}

/// How the mesh's ranks are distributed over operating-system processes.
///
/// §2.11 requires "agreement on rank/process/communicator identity". This is
/// the process half: a launcher that starts eight processes and tells each one
/// it is rank 0 of 8 has produced a layout that no amount of correct collective
/// code recovers from, and it is detectable before any communicator exists.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessLayout {
    /// One process drives every rank. The single-node, single-process case,
    /// and the one the evidence suite uses.
    SingleProcess,
    /// One process per rank, which is what every real launcher produces.
    ProcessPerRank {
        /// This process's rank.
        rank: usize,
        /// The world size this process was told it is part of.
        world: usize,
    },
}

impl ProcessLayout {
    /// The stable tag used in diagnostics and in the fingerprint digest.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::SingleProcess => "single-process",
            Self::ProcessPerRank { .. } => "process-per-rank",
        }
    }
}

/// The number of axes a mesh has.
///
/// Named rather than written as `3` in every signature below because §2.11's
/// strategy table lists expert parallelism as a fourth axis, and the arithmetic
/// in [`CollectiveGroups`] is written over the axis array rather than over
/// three hardcoded degrees. Adding an axis should be adding an entry, not
/// re-deriving a rank convention.
pub const AXIS_COUNT: usize = 3;

/// One axis of a mesh, as a runtime value.
///
/// The typestate counterpart is the marker types [`Data`], [`TensorParallel`],
/// and [`Pipeline`]. This is the same typestate/projection pairing `Placement`
/// and [`PlacementKind`](crate::dist::PlacementKind) use: the markers prove
/// things, the enum is what a group query and a diagnostic can name.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MeshAxis {
    /// Replicas of the whole model.
    Data,
    /// Sequential stages.
    Pipeline,
    /// Ranks one layer's weights are split over.
    Tensor,
}

impl MeshAxis {
    /// The stable name used in diagnostics and in the fingerprint digest.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Data => "data",
            Self::Pipeline => "pipeline",
            Self::Tensor => "tensor",
        }
    }
}

/// The rank layout of a mesh and the collective groups it induces.
///
/// A rank is a single integer and a mesh is `AXIS_COUNT`-dimensional, so
/// something has to fix how one becomes the other. The convention here is
/// **data outermost, then pipeline, then tensor innermost**, i.e.
///
/// ```text
/// rank = d × (PP × TP) + p × TP + t
/// ```
///
/// which makes tensor-parallel peers a *contiguous run* of ranks. That is not
/// arbitrary: tensor parallelism exchanges activations on every layer and is
/// the most bandwidth-hungry axis, launchers assign consecutive ranks to the
/// same host, and so the innermost axis is the one that lands on the fastest
/// link. Data parallelism is outermost because it communicates least - once
/// per step, on gradients.
///
/// Groups are computed from the degrees on demand rather than stored, so there
/// is no second copy of the topology to disagree with the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveGroups {
    /// Axes and their degrees, most significant first. The array order *is*
    /// the layout convention; nothing else encodes it.
    axes: [(MeshAxis, usize); AXIS_COUNT],
}

impl CollectiveGroups {
    /// The groups induced by a logical mesh's degrees.
    #[must_use]
    pub fn of<M: ValidMesh>() -> Self {
        Self {
            axes: [
                (MeshAxis::Data, M::DATA),
                (MeshAxis::Pipeline, M::PIPELINE),
                (MeshAxis::Tensor, M::TENSOR),
            ],
        }
    }

    /// The number of ranks, as the product of the degrees.
    #[must_use]
    pub fn world(&self) -> usize {
        self.axes.iter().map(|&(_, degree)| degree).product()
    }

    /// The degree of one axis.
    #[must_use]
    pub fn degree(&self, axis: MeshAxis) -> usize {
        self.position(axis).map_or(1, |i| self.axes[i].1)
    }

    /// The position of an axis in the layout, most significant first.
    fn position(&self, axis: MeshAxis) -> Option<usize> {
        self.axes.iter().position(|&(a, _)| a == axis)
    }

    /// The stride of the axis at `position`: the rank distance between
    /// neighbours along it, i.e. the product of every less significant degree.
    fn stride(&self, position: usize) -> usize {
        self.axes[position + 1..]
            .iter()
            .map(|&(_, degree)| degree)
            .product()
    }

    /// A rank's coordinate along every axis, most significant first, or `None`
    /// if the rank is not in this mesh.
    pub fn coordinates(&self, rank: usize) -> Option<[usize; AXIS_COUNT]> {
        if rank >= self.world() {
            return None;
        }
        let mut out = [0; AXIS_COUNT];
        for (position, slot) in out.iter_mut().enumerate() {
            *slot = (rank / self.stride(position)) % self.axes[position].1;
        }
        Some(out)
    }

    /// A rank's coordinate along one axis, or `None` if the rank is not in
    /// this mesh.
    pub fn coordinate(&self, rank: usize, axis: MeshAxis) -> Option<usize> {
        let position = self.position(axis)?;
        Some(self.coordinates(rank)?[position])
    }

    /// The rank at a set of coordinates, or `None` if any is out of range.
    ///
    /// The inverse of [`coordinates`](Self::coordinates), and the round trip
    /// between them is what the evidence suite pins: a layout convention that
    /// is not its own inverse is one that silently permutes a mesh.
    pub fn rank_of(&self, coordinates: [usize; AXIS_COUNT]) -> Option<usize> {
        let mut rank = 0;
        for (position, &coordinate) in coordinates.iter().enumerate() {
            if coordinate >= self.axes[position].1 {
                return None;
            }
            rank += coordinate * self.stride(position);
        }
        Some(rank)
    }

    /// Every rank that shares a collective along `axis` with `rank`, itself
    /// included, in ascending order.
    ///
    /// This is the member list of the communicator that axis's collectives run
    /// on: an all-reduce along [`MeshAxis::Data`] is over the data group, an
    /// all-gather along [`MeshAxis::Tensor`] over the tensor group.
    pub fn group(&self, axis: MeshAxis, rank: usize) -> Option<Vec<usize>> {
        let position = self.position(axis)?;
        let coordinate = self.coordinates(rank)?[position];
        let stride = self.stride(position);
        let base = rank - coordinate * stride;
        Some(
            (0..self.axes[position].1)
                .map(|k| base + k * stride)
                .collect(),
        )
    }
}

/// A streaming FNV-1a accumulator.
///
/// Hand-rolled and 25 lines because the digest has to be *identical in two
/// processes that never speak to each other*, which rules out `ahash` (seeded
/// per process) and `DefaultHasher` (explicitly unstable across releases). FNV
/// is not a cryptographic hash and is not used as one: it summarizes a
/// fingerprint two ranks either agree on or do not, and an attacker who can
/// choose your GPU UUIDs has already won.
#[derive(Debug, Clone, Copy)]
struct Digest(u64);

impl Digest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    /// Absorbs a length before the payload, so that `"ab" + "c"` and
    /// `"a" + "bc"` are different fingerprints rather than the same one.
    fn field(self, bytes: &[u8]) -> Self {
        self.number(bytes.len() as u64).bytes(bytes)
    }

    fn text(self, text: &str) -> Self {
        self.field(text.as_bytes())
    }

    fn number(self, value: u64) -> Self {
        self.bytes(&value.to_le_bytes())
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

/// Everything about a machine that two ranks have to agree on.
///
/// §2.11: "A topology fingerprint includes stable device identity,
/// architecture, relevant link classes, transport/library versions, and
/// process layout." Those are exactly the fields, and *relevant* is doing real
/// work in that sentence - the recorded links are the ones the mesh's own
/// collective groups need, not every pair, because a mesh does not care
/// whether two ranks that never communicate can reach each other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyFingerprint {
    /// One identity per rank, indexed by rank.
    devices: Vec<DeviceIdentity>,
    /// `(from, to, class)` for every ordered pair inside a collective group,
    /// in a deterministic order so that the digest is reproducible.
    links: Vec<(usize, usize, LinkClass)>,
    /// The communication library and version.
    transport: TransportVersion,
    /// How ranks map onto processes.
    layout: ProcessLayout,
}

impl TopologyFingerprint {
    /// The identity bound to each rank, indexed by rank.
    #[must_use]
    pub fn devices(&self) -> &[DeviceIdentity] {
        &self.devices
    }

    /// The link classes between ranks that share a collective group.
    #[must_use]
    pub fn links(&self) -> &[(usize, usize, LinkClass)] {
        &self.links
    }

    /// The communication library and version.
    #[must_use]
    pub const fn transport(&self) -> &TransportVersion {
        &self.transport
    }

    /// How ranks map onto processes.
    #[must_use]
    pub const fn layout(&self) -> &ProcessLayout {
        &self.layout
    }

    /// A stable summary of every field.
    ///
    /// Stable means: the same machine bound the same way produces the same
    /// number in a different process, on a different day, in a different build
    /// of this crate. Nothing here is a pointer, an address, an iteration
    /// order over a hash map, or a value that a process seeds at startup.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut digest = Digest::new().field(b"incin.topology.v1");

        digest = digest.number(self.devices.len() as u64);
        for identity in &self.devices {
            digest = digest
                .text(identity.device.kind().name())
                .number(identity.device.ordinal() as u64)
                .text(&identity.persistent)
                .text(&identity.architecture);
        }

        digest = digest.number(self.links.len() as u64);
        for &(from, to, class) in &self.links {
            digest = digest
                .number(from as u64)
                .number(to as u64)
                .text(class.name());
        }

        digest = digest
            .text(&self.transport.library)
            .number(u64::from(self.transport.major))
            .number(u64::from(self.transport.minor))
            .number(u64::from(self.transport.patch))
            .text(self.layout.name());

        digest = match self.layout {
            ProcessLayout::SingleProcess => digest,
            // `rank` is this process's coordinate, not a property of the
            // mesh. Including it makes rank 0 and rank 1 derive different
            // MeshIds for the same multi-process job, defeating the plan-hash
            // agreement the identifier exists to support. The layout kind and
            // world size are shared physical facts; the local rank remains
            // available through `TopologyFingerprint::layout`.
            ProcessLayout::ProcessPerRank { world, .. } => digest.number(world as u64),
        };

        digest.finish()
    }
}

/// The identity of a bound mesh: its physical fingerprint and its logical
/// degrees, together.
///
/// Both halves are needed and neither is sufficient. Two processes on the same
/// eight GPUs that disagree about whether that is `DP=8` or `DP=2, TP=4` are
/// running incompatible programs, and their fingerprints are identical. Two
/// processes that agree on `DP=8` while looking at different hardware are also
/// incompatible, and their degrees are identical.
///
/// This is the handle `DST-003`'s placements carry, and it is computed rather
/// than assigned so that ranks agree on it without a round trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshId(u64);

impl MeshId {
    /// Rebuild an identity received from another process.
    ///
    /// This is only data until collective-plan preflight compares it with the
    /// local identity.
    #[must_use]
    pub const fn from_digest(digest: u64) -> Self {
        Self(digest)
    }

    /// The underlying digest.
    #[must_use]
    pub const fn digest(self) -> u64 {
        self.0
    }
}

/// A logical mesh that has been held against real devices and accepted.
///
/// The type parameter is the [`MeshSpec`] this was bound for, so a
/// `DeviceMesh<MeshSpec<Data<U3>>>` and a
/// `DeviceMesh<MeshSpec<Data<U1>, TensorParallel<U3>>>` are different types
/// even though both hold three of the same devices. That is the whole reason
/// the logical half exists: the two are not interchangeable, and nothing at
/// runtime would tell them apart.
///
/// The only way to build one is [`bind`](Self::bind), so a `DeviceMesh` in
/// hand is a claim every guard in this module has already passed.
pub struct DeviceMesh<M> {
    fingerprint: TopologyFingerprint,
    id: MeshId,
    spec: PhantomData<M>,
}

// These four are written out rather than derived because `derive` would bound
// them on `M: Debug + Clone + PartialEq + Eq`, and `M` is a `MeshSpec` - a
// marker that is never constructed and implements none of them. The bound
// would make a `DeviceMesh` undebuggable for exactly the type parameters it is
// meant to be used with.

impl<M> fmt::Debug for DeviceMesh<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceMesh")
            .field("id", &self.id)
            .field("fingerprint", &self.fingerprint)
            .finish()
    }
}

impl<M> Clone for DeviceMesh<M> {
    fn clone(&self) -> Self {
        Self {
            fingerprint: self.fingerprint.clone(),
            id: self.id,
            spec: PhantomData,
        }
    }
}

impl<M> PartialEq for DeviceMesh<M> {
    /// Two meshes of the same type are equal when they bound the same machine.
    /// The identity already folds in the degrees, and the degrees are `M`.
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.fingerprint == other.fingerprint
    }
}

impl<M> Eq for DeviceMesh<M> {}

/// Why a set of devices is not this mesh.
///
/// One variant per guard, each naming the rank and the values involved,
/// because every one of these is something a user has to go fix in a launcher
/// script and "binding failed" does not tell them which. This enum is here
/// rather than alongside the crate's other error enums because it exists only
/// under the
/// `distributed` feature, and a core error type whose variant set depends on a
/// feature is one that callers cannot match on portably.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum BindError {
    #[error("mesh needs {expected} ranks but {found} devices were given")]
    /// The device list is not the size the type says it is.
    ///
    /// The first guard because every later one indexes by rank, and a rank
    /// that has no device is not a diagnosis, it is a panic.
    RankCount {
        /// `M::WORLD`.
        expected: usize,
        /// How many devices the caller passed.
        found: usize,
    },

    #[error("device {device:?} was given twice, as rank {first} and rank {second}")]
    /// The same ordinal appears at two ranks.
    ///
    /// Two ranks sharing one device is the classic launcher misconfiguration:
    /// it runs, it is half as fast as it should be, and it silently
    /// double-counts a gradient.
    RepeatedDevice {
        /// The ordinal given twice.
        device: DeviceId,
        /// The lower rank it was given for.
        first: usize,
        /// The higher rank it was given for.
        second: usize,
    },

    #[error("rank {rank} names device {device:?}, which the probe cannot see")]
    /// An ordinal that names no installed device.
    UnknownDevice {
        /// The rank that named it.
        rank: usize,
        /// The ordinal that resolved to nothing.
        device: DeviceId,
    },

    #[error(
        "rank {first} and rank {second} name different ordinals but the same physical device \
         ({persistent})"
    )]
    /// Two distinct ordinals resolved to one physical device.
    ///
    /// This is why §2.11 says the ordinal is not an identity. `RepeatedDevice`
    /// catches the same number twice; this catches the case a visibility mask
    /// creates, where two different numbers are the same card and nothing
    /// short of the persistent id can tell.
    AliasedDevice {
        /// The lower rank.
        first: usize,
        /// The higher rank.
        second: usize,
        /// The persistent id both resolved to.
        persistent: String,
    },

    #[error(
        "rank 0 is a {} device but rank {rank} is a {} device",
        expected.name(),
        found.name()
    )]
    /// The mesh spans more than one backend family.
    MixedBackendFamily {
        /// The family rank 0 established.
        expected: DeviceKind,
        /// The family this rank has.
        found: DeviceKind,
        /// The first rank that disagreed.
        rank: usize,
    },

    #[error("rank 0 is architecture {expected} but rank {rank} is {found}")]
    /// The mesh spans more than one compute architecture.
    ///
    /// Same family, different capabilities: the ranks do not agree on which
    /// kernels exist, and a collective is a place where they all have to.
    MixedArchitecture {
        /// The architecture rank 0 established.
        expected: String,
        /// The architecture this rank has.
        found: String,
        /// The first rank that disagreed.
        rank: usize,
    },

    #[error("process layout {layout} does not describe a {world}-rank mesh")]
    /// The launcher's idea of the world disagrees with the type's.
    UnsupportedProcessLayout {
        /// The layout the probe reported, rendered.
        layout: String,
        /// `M::WORLD`.
        world: usize,
    },

    #[error("rank {from} cannot reach rank {to}, which shares its {} group", axis.name())]
    /// Two ranks that have to run a collective together have no path.
    UnreachableGroup {
        /// The axis whose group they share.
        axis: MeshAxis,
        /// The rank the link was probed from.
        from: usize,
        /// The rank it could not reach.
        to: usize,
    },

    #[error("collective group along {} for rank {rank} has no members", axis.name())]
    /// The mesh's own group table did not cover a rank the count guard checked.
    ///
    /// `axis` comes from the table itself and `rank` is below `M::WORLD`,
    /// so `None` here means the type-level degrees disagree with the world
    /// size - a broken mesh definition, not a launcher mistake. Returned
    /// rather than panicked because `bind` already reports every other
    /// binding failure as data.
    InconsistentTopology {
        /// The axis whose group was missing.
        axis: MeshAxis,
        /// The rank the group was requested for.
        rank: usize,
    },
}

impl<M: ValidMesh> DeviceMesh<M> {
    /// Holds a device list against this mesh's logical topology.
    ///
    /// Rank `i` is `devices[i]`, under the layout convention
    /// [`CollectiveGroups`] documents. The guards run in the order their
    /// variants are declared, which is from cheapest and most likely to
    /// most specific, so the first thing a user sees is the thing most likely
    /// to be wrong.
    ///
    /// §2.11's binding list also includes backend/dtype capabilities and
    /// estimated peak memory. Neither is here: the first needs a
    /// [`CapabilityRegistry`](crate::exec::capability::CapabilityRegistry) per
    /// rank, which arrives with the backends in `DST-005`/`DST-006`, and the
    /// second needs a plan to estimate, which is `DST-007`.
    ///
    /// # Errors
    ///
    /// Returns the first [`BindError`] any guard produces.
    pub fn bind<P: TopologyProbe>(devices: &[DeviceId], probe: &P) -> Result<Self, BindError> {
        if devices.len() != M::WORLD {
            return Err(BindError::RankCount {
                expected: M::WORLD,
                found: devices.len(),
            });
        }

        for (second, &device) in devices.iter().enumerate() {
            if let Some(first) = devices[..second].iter().position(|&seen| seen == device) {
                return Err(BindError::RepeatedDevice {
                    device,
                    first,
                    second,
                });
            }
        }

        let mut identities: Vec<DeviceIdentity> = Vec::with_capacity(devices.len());
        for (rank, &device) in devices.iter().enumerate() {
            let identity = probe
                .identify(device)
                .ok_or(BindError::UnknownDevice { rank, device })?;

            if let Some(first) = identities
                .iter()
                .position(|seen| seen.persistent == identity.persistent)
            {
                return Err(BindError::AliasedDevice {
                    first,
                    second: rank,
                    persistent: identity.persistent,
                });
            }

            if let Some(head) = identities.first() {
                if head.device.kind() != identity.device.kind() {
                    return Err(BindError::MixedBackendFamily {
                        expected: head.device.kind(),
                        found: identity.device.kind(),
                        rank,
                    });
                }
                if head.architecture != identity.architecture {
                    return Err(BindError::MixedArchitecture {
                        expected: head.architecture.clone(),
                        found: identity.architecture,
                        rank,
                    });
                }
            }

            identities.push(identity);
        }

        let layout = probe.layout();
        if let ProcessLayout::ProcessPerRank { rank, world } = layout
            && (world != M::WORLD || rank >= world)
        {
            return Err(BindError::UnsupportedProcessLayout {
                layout: alloc::format!("{layout:?}"),
                world: M::WORLD,
            });
        }

        let groups = CollectiveGroups::of::<M>();
        let mut links = Vec::new();
        for &(axis, _) in &groups.axes {
            for from in 0..M::WORLD {
                let members = groups
                    .group(axis, from)
                    .ok_or(BindError::InconsistentTopology { axis, rank: from })?;
                for to in members {
                    if to == from {
                        continue;
                    }
                    let class = probe.link(devices[from], devices[to]);
                    if !class.reaches() {
                        return Err(BindError::UnreachableGroup { axis, from, to });
                    }
                    links.push((from, to, class));
                }
            }
        }

        let fingerprint = TopologyFingerprint {
            devices: identities,
            links,
            transport: probe.transport(),
            layout,
        };
        let id = MeshId(
            Digest::new()
                .number(fingerprint.digest())
                .number(M::DATA as u64)
                .number(M::PIPELINE as u64)
                .number(M::TENSOR as u64)
                .finish(),
        );

        Ok(Self {
            fingerprint,
            id,
            spec: PhantomData,
        })
    }

    /// This mesh's identity: its fingerprint and its degrees.
    #[must_use]
    pub const fn id(&self) -> MeshId {
        self.id
    }

    /// What the probe said about the machine this was bound to.
    #[must_use]
    pub const fn fingerprint(&self) -> &TopologyFingerprint {
        &self.fingerprint
    }

    /// The rank layout and the collective groups it induces.
    ///
    /// Recomputed from the type's degrees rather than stored, so a
    /// `DeviceMesh` cannot come to hold a topology that disagrees with the
    /// `MeshSpec` it is parameterized by.
    #[must_use]
    pub fn groups(&self) -> CollectiveGroups {
        CollectiveGroups::of::<M>()
    }

    /// The identity bound to `rank`, or `None` if the mesh has no such rank.
    pub fn device(&self, rank: usize) -> Option<&DeviceIdentity> {
        self.fingerprint.devices.get(rank)
    }
}
