use super::*;

/// Collective algorithm family offered to a transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CollectiveAlgorithm {
    /// Bandwidth-oriented ring.
    Ring,
    /// Latency-oriented tree.
    Tree,
}

/// Collective wire protocol family offered to a transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CollectiveProtocol {
    /// General protocol.
    Simple,
    /// Low-latency protocol.
    LowLatency,
    /// Wider low-latency protocol.
    LowLatency128,
}

/// Type-level collective algorithm for static candidate construction.
pub trait StaticCollectiveAlgorithm: 'static {
    /// Runtime projection recorded in the candidate.
    const ALGORITHM: CollectiveAlgorithm;
}

/// Static ring marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ring;

impl StaticCollectiveAlgorithm for Ring {
    const ALGORITHM: CollectiveAlgorithm = CollectiveAlgorithm::Ring;
}

/// Static tree marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tree;

impl StaticCollectiveAlgorithm for Tree {
    const ALGORITHM: CollectiveAlgorithm = CollectiveAlgorithm::Tree;
}

/// Type-level collective protocol for static candidate construction.
pub trait StaticCollectiveProtocol: 'static {
    /// Runtime projection recorded in the candidate.
    const PROTOCOL: CollectiveProtocol;
}

/// Static general-protocol marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Simple;

impl StaticCollectiveProtocol for Simple {
    const PROTOCOL: CollectiveProtocol = CollectiveProtocol::Simple;
}

/// Static low-latency protocol marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LowLatency;

impl StaticCollectiveProtocol for LowLatency {
    const PROTOCOL: CollectiveProtocol = CollectiveProtocol::LowLatency;
}

/// Static 128-bit low-latency protocol marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LowLatency128;

impl StaticCollectiveProtocol for LowLatency128 {
    const PROTOCOL: CollectiveProtocol = CollectiveProtocol::LowLatency128;
}

/// Compile-time collective operation selected for one tuning problem.
pub trait StaticCollectiveTuning<K: CollectiveDType, Elements: Unsigned>: 'static {
    /// Runtime collective descriptor.
    const KIND: CollectiveKind;
}

/// Static all-gather marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneAllGather;

impl<K, Elements> StaticCollectiveTuning<K, Elements> for TuneAllGather
where
    K: CollectiveDType,
    Elements: Unsigned,
{
    const KIND: CollectiveKind = CollectiveKind::AllGather;
}

/// Static all-to-all marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneAllToAll;

impl<K, Elements> StaticCollectiveTuning<K, Elements> for TuneAllToAll
where
    K: CollectiveDType,
    Elements: Unsigned + ShardDivisible<U2>,
{
    const KIND: CollectiveKind = CollectiveKind::AllToAll;
}

/// Static all-reduce marker parameterized by a reduction proof marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneAllReduce<R>(PhantomData<R>);

impl<K, Elements, R> StaticCollectiveTuning<K, Elements> for TuneAllReduce<R>
where
    K: CollectiveReductionDType<R>,
    Elements: Unsigned,
    R: PartialReduction,
{
    const KIND: CollectiveKind = CollectiveKind::AllReduce(R::OP);
}

/// Static reduce-scatter marker parameterized by a reduction proof marker.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneReduceScatter<R>(PhantomData<R>);

impl<K, Elements, R> StaticCollectiveTuning<K, Elements> for TuneReduceScatter<R>
where
    K: CollectiveReductionDType<R>,
    Elements: Unsigned + ShardDivisible<U2>,
    R: PartialReduction,
{
    const KIND: CollectiveKind = CollectiveKind::ReduceScatter(R::OP);
}

/// Static point-to-point marker from rank zero to rank one.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneSendZeroToOne;

impl<K, Elements> StaticCollectiveTuning<K, Elements> for TuneSendZeroToOne
where
    K: CollectiveDType,
    Elements: Unsigned,
{
    const KIND: CollectiveKind = CollectiveKind::SendRecv {
        source: 0,
        destination: 1,
    };
}

/// Static point-to-point marker from rank one to rank zero.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TuneSendOneToZero;

impl<K, Elements> StaticCollectiveTuning<K, Elements> for TuneSendOneToZero
where
    K: CollectiveDType,
    Elements: Unsigned,
{
    const KIND: CollectiveKind = CollectiveKind::SendRecv {
        source: 1,
        destination: 0,
    };
}
