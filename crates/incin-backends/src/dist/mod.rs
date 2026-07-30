//! Distributed collective transport contracts.
//!
//! The deterministic reference implementation is intentionally independent of
//! CUDA and networking. It establishes collective values, counts, dtype
//! behavior, and adjoints before a native transport is allowed to optimize
//! them.

pub mod collective;
#[cfg(feature = "distributed-nccl")]
pub mod nccl;
#[cfg(feature = "distributed-reference")]
pub mod reference;
pub mod tuning;

pub use collective::{
    CollectiveBackend, CollectiveDType, CollectiveError, CollectiveKind, CollectiveOutput, GroupId,
    StreamId,
};
#[cfg(feature = "distributed-nccl")]
pub use nccl::{
    BootstrapRole, NcclBuffer, NcclEvent, NcclTopology, NcclTransport, NcclTransportError,
    TwoRankBootstrapConfig,
};
#[cfg(feature = "distributed-reference")]
pub use reference::{
    ReferenceBuffer, ReferenceEvent, ReferenceTopology, ReferenceTransport, ReferenceValues,
};
pub use tuning::{
    CandidateRound, CollectiveAlgorithm, CollectiveProtocol, CollectiveTuningBudget,
    CollectiveTuningCandidate, CollectiveTuningError, CollectiveTuningKey, CollectiveTuningProblem,
    CommitVote, CommittedCollectiveTuning, LowLatency, LowLatency128, ProvisionalCollectiveTuning,
    RankSampleReport, Ring, Simple, StaticCollectiveAlgorithm, StaticCollectiveProtocol,
    StaticCollectiveTuning, Tree, TuneAllGather, TuneAllReduce, TuneAllToAll, TuneReduceScatter,
    TuneSendOneToZero, TuneSendZeroToOne, commit_collective_tuning, select_collective_candidate,
};
