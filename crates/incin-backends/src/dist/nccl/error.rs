//! Structured bootstrap, plan, metadata, CUDA, and NCCL failures.

use std::io;
use std::time::Duration;

use incin_core::dist::{
    CollectiveError, CollectiveKind, ContextError, DataParallelError, GradientId, PipelineError,
    PipelineTransfer, PlanError, TensorParallelCollective, TensorParallelError,
};
use incin_core::tensor::dtype::DTypeId;

/// Structured bootstrap, plan, metadata, CUDA, and NCCL failures.
#[non_exhaustive]
#[derive(thiserror::Error, Debug)]
pub enum NcclTransportError {
    /// The shared process context is shut down or failed.
    #[error(transparent)]
    Context(#[from] ContextError),
    /// Shared collective validation failed.
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    /// Cross-rank plan agreement failed.
    #[error(transparent)]
    Plan(#[from] PlanError),
    /// Data-parallel dtype or gradient-plan validation failed.
    #[error(transparent)]
    DataParallel(#[from] DataParallelError),
    /// Tensor-parallel shape, dtype, or plan validation failed.
    #[error(transparent)]
    TensorParallel(#[from] TensorParallelError),
    /// Pipeline shape, dtype, schedule, or plan validation failed.
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
    /// The requested parameter has no gradient in the completed backward pass.
    #[error("parameter gradient {id:?} was not produced by backward")]
    MissingGradient { id: GradientId },
    /// Caller and plan disagree on which parameter occupies this sequence.
    #[error("gradient identity is {found}, expected {expected}")]
    GradientIdentity { expected: u64, found: u64 },
    /// The next descriptor is not a DP mean all-reduce.
    #[error(
        "gradient descriptor must be Partial<Mean> -> Replicated all-reduce, got {kind:?} from {from_placement:?} to {to_placement:?}"
    )]
    NotDataParallelGradient {
        kind: CollectiveKind,
        from_placement: incin_core::dist::PlacementKind,
        to_placement: incin_core::dist::PlacementKind,
    },
    /// A gradient allocation belongs to another local CUDA ordinal.
    #[error("gradient is on CUDA device {found}, transport owns CUDA device {expected}")]
    GradientDevice { expected: usize, found: usize },
    /// NCCL's scalar buffer path requires a contiguous offset-zero gradient.
    #[error(
        "gradient layout is {layout:?} at element offset {offset}; contiguous offset-zero required"
    )]
    GradientLayout {
        offset: usize,
        layout: incin_core::exec::LayoutClass,
    },
    /// Caller and plan disagree on the tensor-parallel semantic operation.
    #[error("tensor-parallel operation tag is {found}, expected {expected}")]
    TensorParallelIdentity { expected: u64, found: u64 },
    /// The next descriptor does not implement the requested TP operation.
    #[error(
        "tensor-parallel descriptor for {expected:?} got {kind:?} from {from_placement:?} to {to_placement:?}"
    )]
    NotTensorParallelOperation {
        expected: TensorParallelCollective,
        kind: CollectiveKind,
        from_placement: incin_core::dist::PlacementKind,
        to_placement: incin_core::dist::PlacementKind,
    },
    /// A TP tensor allocation belongs to another local CUDA ordinal.
    #[error(
        "tensor-parallel input is on CUDA device {found}, transport owns CUDA device {expected}"
    )]
    TensorParallelDevice { expected: usize, found: usize },
    /// Direct NCCL tensor execution requires contiguous offset-zero storage.
    #[error(
        "tensor-parallel input layout is {layout:?} at element offset {offset}; contiguous offset-zero required"
    )]
    TensorParallelLayout {
        offset: usize,
        layout: incin_core::exec::LayoutClass,
    },
    /// Rank-local tensor shape disagrees with the requested logical shard.
    #[error("tensor-parallel input shape is {found:?}, expected {expected:?}")]
    TensorParallelShape {
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    /// Requested global shape disagrees with the immutable plan.
    #[error("tensor-parallel output has {found} elements, expected {expected}")]
    TensorParallelOutputElements { expected: usize, found: usize },
    /// Caller and plan disagree on pipeline boundary, direction, or microbatch.
    #[error("pipeline transfer tag is {found}, expected {expected}")]
    PipelineIdentity { expected: u64, found: u64 },
    /// The next descriptor does not implement the requested pipeline transfer.
    #[error(
        "pipeline descriptor for {expected:?} got {kind:?} from {from_placement:?} to {to_placement:?}"
    )]
    NotPipelineTransfer {
        expected: PipelineTransfer,
        kind: CollectiveKind,
        from_placement: incin_core::dist::PlacementKind,
        to_placement: incin_core::dist::PlacementKind,
    },
    /// A pipeline tensor allocation belongs to another local CUDA ordinal.
    #[error("pipeline input is on CUDA device {found}, transport owns CUDA device {expected}")]
    PipelineDevice { expected: usize, found: usize },
    /// Direct pipeline transfer requires contiguous offset-zero storage.
    #[error(
        "pipeline input layout is {layout:?} at element offset {offset}; contiguous offset-zero required"
    )]
    PipelineLayout {
        offset: usize,
        layout: incin_core::exec::LayoutClass,
    },
    /// A checked buffer could not be represented.
    #[error("invalid NCCL buffer: {0}")]
    InvalidBuffer(String),
    /// A CUDA allocation has a different byte count.
    #[error("NCCL buffer has {found} bytes, expected {expected}")]
    BufferBytes { expected: usize, found: usize },
    /// Runtime dtype differs from the next descriptor.
    #[error("NCCL buffer has dtype {found:?}, expected {expected:?}")]
    DType { expected: DTypeId, found: DTypeId },
    /// Runtime element count differs from the next descriptor.
    #[error("NCCL buffer has {found} elements, expected {expected}")]
    Elements { expected: usize, found: usize },
    /// The immutable plan has no next launch.
    #[error("collective plan is exhausted after {collectives} launches")]
    PlanExhausted { collectives: usize },
    /// Descriptor order is not the canonical zero-based sequence.
    #[error("collective sequence is {found}, expected {expected}")]
    Sequence { expected: u64, found: u64 },
    /// This transport is deliberately fixed at two network ranks.
    #[error("NCCL group has {found} ranks, expected {expected}")]
    GroupCardinality { expected: usize, found: usize },
    /// A newer collective kind is not implemented by this transport version.
    #[error("collective kind is unsupported by this NCCL transport version")]
    UnsupportedCollective,
    /// Bootstrap peer announced another world size.
    #[error("remote bootstrap world is {found}, expected {expected}")]
    WorldSize { expected: usize, found: usize },
    /// Bootstrap peer announced the wrong rank.
    #[error("remote bootstrap rank is {found}, expected {expected}")]
    RemoteRank { expected: usize, found: usize },
    /// This process's configured rank is outside the two-rank world.
    #[error("local rank {rank} is outside a world of {world}")]
    LocalRank { rank: usize, world: usize },
    /// Rank zero did not supply its NCCL id to the protocol.
    #[error("rank zero bootstrap requires an NCCL unique id")]
    MissingRootId,
    /// Rank one attempted to choose the communicator identity.
    #[error("rank one must receive, not supply, the NCCL unique id")]
    UnexpectedPeerId,
    /// Wire data did not follow the versioned protocol.
    #[error("invalid NCCL bootstrap protocol: {0}")]
    Protocol(&'static str),
    /// A fixed-size topology field could not carry its value.
    #[error("{field} has {found} bytes, maximum is {maximum}")]
    FieldTooLong {
        field: &'static str,
        maximum: usize,
        found: usize,
    },
    /// Hosts loaded different transport implementations or versions.
    #[error("transport mismatch: local {local}, remote {remote}")]
    TransportMismatch { local: String, remote: String },
    /// Zero or overflowing durations cannot form a deadline.
    #[error("NCCL timeout must be positive and fit Instant")]
    InvalidTimeout,
    /// A bounded phase reached its deadline.
    #[error("{phase} timed out after {timeout:?}")]
    Timeout {
        phase: &'static str,
        timeout: Duration,
    },
    /// Socket I/O reported a timeout.
    #[error("{operation} timed out")]
    IoTimeout { operation: &'static str },
    /// Other socket failure.
    #[error("{operation} failed ({kind:?}): {message}")]
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
        message: String,
    },
    /// CUDA driver failure.
    #[error("CUDA failure: {0}")]
    Cuda(String),
    /// NCCL failure.
    #[error("NCCL failure: {0}")]
    Nccl(String),
    /// cudarc could not dynamically load the NCCL shared library.
    #[error("cannot {operation}: NCCL is unavailable ({message})")]
    NcclUnavailable {
        operation: &'static str,
        message: String,
    },
    /// NCCL returned a negative or otherwise unrepresentable version.
    #[error("NCCL returned invalid encoded version {0}")]
    InvalidNcclVersion(i32),
}

pub(crate) fn io_error(operation: &'static str, error: io::Error) -> NcclTransportError {
    if matches!(
        error.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    ) {
        NcclTransportError::IoTimeout { operation }
    } else {
        NcclTransportError::Io {
            operation,
            kind: error.kind(),
            message: error.to_string(),
        }
    }
}

pub(crate) fn nccl_error(error: cudarc::nccl::result::NcclError) -> NcclTransportError {
    NcclTransportError::Nccl(format!("{:?}", error.0))
}

pub(crate) fn catch_nccl_panic<T>(
    operation: &'static str,
    call: impl FnOnce() -> T,
) -> Result<T, NcclTransportError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).map_err(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map_or_else(
                || {
                    payload
                        .downcast_ref::<String>()
                        .map_or("unknown loader panic", String::as_str)
                },
                |message| *message,
            )
            .to_string();
        NcclTransportError::NcclUnavailable { operation, message }
    })
}
