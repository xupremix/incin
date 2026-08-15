//! Backend-neutral collective interface.

use incin_core::exec::ReduceOp;
use incin_core::tensor::dtype::DType;

pub use incin_core::dist::{CollectiveDType, CollectiveError, CollectiveKind, GroupId, StreamId};

/// Completed collective values together with its backend event.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectiveOutput<B, E> {
    buffers: alloc::vec::Vec<B>,
    event: E,
}

impl<B, E> CollectiveOutput<B, E> {
    #[cfg(feature = "distributed-reference")]
    pub(crate) fn new(buffers: alloc::vec::Vec<B>, event: E) -> Self {
        Self { buffers, event }
    }

    /// One output buffer per rank, in group order.
    #[must_use]
    pub fn buffers(&self) -> &[B] {
        &self.buffers
    }

    /// Consume the result into its rank buffers and event.
    #[must_use]
    pub fn into_parts(self) -> (alloc::vec::Vec<B>, E) {
        (self.buffers, self.event)
    }

    /// Completion/dependency event.
    #[must_use]
    pub const fn event(&self) -> &E {
        &self.event
    }
}

/// Collective operations over dtype-indexed storage.
pub trait CollectiveBackend {
    /// Transport buffer selected by tensor dtype.
    type Buffer<K: DType>: Clone;
    /// Completion/dependency token.
    type Event: Clone;

    /// Reduce rank buffers and return the same complete result to every rank.
    fn all_reduce<K: CollectiveDType>(
        &self,
        group: GroupId,
        inputs: &[Self::Buffer<K>],
        op: ReduceOp,
        stream: StreamId,
    ) -> Result<CollectiveOutput<Self::Buffer<K>, Self::Event>, CollectiveError>;

    /// Concatenate shards in rank order and return the result to every rank.
    fn all_gather<K: CollectiveDType>(
        &self,
        group: GroupId,
        inputs: &[Self::Buffer<K>],
        stream: StreamId,
    ) -> Result<CollectiveOutput<Self::Buffer<K>, Self::Event>, CollectiveError>;

    /// Reduce complete values and return one equal contiguous shard per rank.
    fn reduce_scatter<K: CollectiveDType>(
        &self,
        group: GroupId,
        inputs: &[Self::Buffer<K>],
        op: ReduceOp,
        stream: StreamId,
    ) -> Result<CollectiveOutput<Self::Buffer<K>, Self::Event>, CollectiveError>;

    /// Exchange equal contiguous chunks among every rank.
    fn all_to_all<K: CollectiveDType>(
        &self,
        group: GroupId,
        inputs: &[Self::Buffer<K>],
        stream: StreamId,
    ) -> Result<CollectiveOutput<Self::Buffer<K>, Self::Event>, CollectiveError>;

    /// Transfer one source buffer to one destination in an ordered group.
    ///
    /// The operation is global: all ranks observe the same source and
    /// destination metadata. This keeps point-to-point pipeline plans
    /// preflightable without hashing different local `send` and `recv` calls.
    fn send_recv<K: CollectiveDType>(
        &self,
        group: GroupId,
        inputs: &[Self::Buffer<K>],
        source: usize,
        destination: usize,
        stream: StreamId,
    ) -> Result<CollectiveOutput<Self::Buffer<K>, Self::Event>, CollectiveError>;
}
