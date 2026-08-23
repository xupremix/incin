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
    MissingGradient {
        /// Gradient identifier involved.
        id: GradientId,
    },
    /// Caller and plan disagree on which parameter occupies this sequence.
    #[error("gradient identity is {found}, expected {expected}")]
    GradientIdentity {
        /// Expected value.
        expected: u64,
        /// Actual value.
        found: u64,
    },
    /// The next descriptor is not a DP mean all-reduce.
    #[error(
        "gradient descriptor must be Partial<Mean> -> Replicated all-reduce, got {kind:?} from {from_placement:?} to {to_placement:?}"
    )]
    NotDataParallelGradient {
        /// Collective involved in the mismatch.
        kind: CollectiveKind,
        /// Placement the transfer starts from.
        from_placement: incin_core::dist::PlacementKind,
        /// Placement the transfer targets.
        to_placement: incin_core::dist::PlacementKind,
    },
    /// A gradient allocation belongs to another local CUDA ordinal.
    #[error("gradient is on CUDA device {found}, transport owns CUDA device {expected}")]
    GradientDevice {
        /// Expected value.
        expected: usize,
        /// Actual value.
        found: usize,
    },
    /// NCCL's scalar buffer path requires a contiguous offset-zero gradient.
    #[error(
        "gradient layout is {layout:?} at element offset {offset}; contiguous offset-zero required"
    )]
    GradientLayout {
        /// Offset into the buffer where validation failed.
        offset: usize,
        /// Layout class observed.
        layout: incin_core::exec::LayoutClass,
    },
    /// Caller and plan disagree on the tensor-parallel semantic operation.
    #[error("tensor-parallel operation tag is {found}, expected {expected}")]
    TensorParallelIdentity {
        /// Expected value.
        expected: u64,
        /// Actual value.
        found: u64,
    },
    /// The next descriptor does not implement the requested TP operation.
    #[error(
        "tensor-parallel descriptor for {expected:?} got {kind:?} from {from_placement:?} to {to_placement:?}"
    )]
    NotTensorParallelOperation {
        /// Value the contract expects.
        expected: TensorParallelCollective,
        /// Collective involved in the mismatch.
        kind: CollectiveKind,
        /// Placement the transfer starts from.
        from_placement: incin_core::dist::PlacementKind,
        /// Placement the transfer targets.
        to_placement: incin_core::dist::PlacementKind,
    },
    /// A TP tensor allocation belongs to another local CUDA ordinal.
    #[error(
        "tensor-parallel input is on CUDA device {found}, transport owns CUDA device {expected}"
    )]
    /// Reported when this precondition is violated.
    TensorParallelDevice {
        /// Value the contract expects.
        expected: usize,
        /// Value actually present.
        found: usize,
    },
    /// Direct NCCL tensor execution requires contiguous offset-zero storage.
    #[error(
        "tensor-parallel input layout is {layout:?} at element offset {offset}; contiguous offset-zero required"
    )]
    TensorParallelLayout {
        /// Offset into the buffer where validation failed.
        offset: usize,
        /// Layout class observed.
        layout: incin_core::exec::LayoutClass,
    },
    /// Rank-local tensor shape disagrees with the requested logical shard.
    #[error("tensor-parallel input shape is {found:?}, expected {expected:?}")]
    TensorParallelShape {
        /// Value the contract expects.
        expected: Vec<usize>,
        /// Value actually present.
        found: Vec<usize>,
    },
    /// Requested global shape disagrees with the immutable plan.
    #[error("tensor-parallel output has {found} elements, expected {expected}")]
    TensorParallelOutputElements {
        /// Expected value.
        expected: usize,
        /// Actual value.
        found: usize,
    },
    /// Caller and plan disagree on pipeline boundary, direction, or microbatch.
    #[error("pipeline transfer tag is {found}, expected {expected}")]
    PipelineIdentity {
        /// Identity this rank expected from the plan.
        expected: u64,
        /// Identity the descriptor actually carried.
        found: u64,
    },
    /// The next descriptor does not implement the requested pipeline transfer.
    #[error(
        "pipeline descriptor for {expected:?} got {kind:?} from {from_placement:?} to {to_placement:?}"
    )]
    NotPipelineTransfer {
        /// Value the contract expects.
        expected: PipelineTransfer,
        /// Collective involved in the mismatch.
        kind: CollectiveKind,
        /// Placement the transfer starts from.
        from_placement: incin_core::dist::PlacementKind,
        /// Placement the transfer targets.
        to_placement: incin_core::dist::PlacementKind,
    },
    /// A pipeline tensor allocation belongs to another local CUDA ordinal.
    #[error("pipeline input is on CUDA device {found}, transport owns CUDA device {expected}")]
    PipelineDevice {
        /// Expected value.
        expected: usize,
        /// Actual value.
        found: usize,
    },
    /// Direct pipeline transfer requires contiguous offset-zero storage.
    #[error(
        "pipeline input layout is {layout:?} at element offset {offset}; contiguous offset-zero required"
    )]
    PipelineLayout {
        /// Offset into the buffer where validation failed.
        offset: usize,
        /// Layout class observed.
        layout: incin_core::exec::LayoutClass,
    },
    /// A checked buffer could not be represented.
    #[error("invalid NCCL buffer: {0}")]
    InvalidBuffer(String),
    /// A CUDA allocation has a different byte count.
    #[error("NCCL buffer has {found} bytes, expected {expected}")]
    BufferBytes {
        /// Expected value.
        expected: usize,
        /// Actual value.
        found: usize,
    },
    /// Runtime dtype differs from the next descriptor.
    #[error("NCCL buffer has dtype {found:?}, expected {expected:?}")]
    DType {
        /// Expected value.
        expected: DTypeId,
        /// Actual value.
        found: DTypeId,
    },
    /// Runtime element count differs from the next descriptor.
    #[error("NCCL buffer has {found} elements, expected {expected}")]
    Elements {
        /// Expected value.
        expected: usize,
        /// Actual value.
        found: usize,
    },
    /// The immutable plan has no next launch.
    #[error("collective plan is exhausted after {collectives} launches")]
    PlanExhausted {
        /// Collectives contained in the exhausted plan.
        collectives: usize,
    },
    /// Descriptor order is not the canonical zero-based sequence.
    #[error("collective sequence is {found}, expected {expected}")]
    Sequence {
        /// Expected value.
        expected: u64,
        /// Actual value.
        found: u64,
    },
    /// This transport is deliberately fixed at two network ranks.
    #[error("NCCL group has {found} ranks, expected {expected}")]
    GroupCardinality {
        /// Expected value.
        expected: usize,
        /// Actual value.
        found: usize,
    },
    /// A newer collective kind is not implemented by this transport version.
    #[error("collective kind is unsupported by this NCCL transport version")]
    UnsupportedCollective,
    /// Bootstrap peer announced another world size.
    #[error("remote bootstrap world is {found}, expected {expected}")]
    WorldSize {
        /// Expected value.
        expected: usize,
        /// Actual value.
        found: usize,
    },
    /// Bootstrap peer announced the wrong rank.
    #[error("remote bootstrap rank is {found}, expected {expected}")]
    RemoteRank {
        /// Expected value.
        expected: usize,
        /// Actual value.
        found: usize,
    },
    /// This process's configured rank is outside the two-rank world.
    #[error("local rank {rank} is outside a world of {world}")]
    LocalRank {
        /// Offending rank index.
        rank: usize,
        /// World size validated against.
        world: usize,
    },
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
        /// Field whose bound was exceeded.
        field: &'static str,
        /// Configured maximum.
        maximum: usize,
        /// Value actually present.
        found: usize,
    },
    /// Hosts loaded different transport implementations or versions.
    #[error("transport mismatch: local {local}, remote {remote}")]
    TransportMismatch {
        /// This rank's transport identity.
        local: String,
        /// The peer's transport identity.
        remote: String,
    },
    /// Zero or overflowing durations cannot form a deadline.
    #[error("NCCL timeout must be positive and fit Instant")]
    InvalidTimeout,
    /// A bounded phase reached its deadline.
    #[error("{phase} timed out after {timeout:?}")]
    Timeout {
        /// Handshake phase that timed out.
        phase: &'static str,
        /// Configured timeout duration.
        timeout: Duration,
    },
    /// Socket I/O reported a timeout.
    #[error("{operation} timed out")]
    IoTimeout {
        /// Operation or IO step that failed.
        operation: &'static str,
    },
    /// Other socket failure.
    #[error("{operation} failed ({kind:?}): {message}")]
    Io {
        /// Operation or IO step that failed.
        operation: &'static str,
        /// Collective involved in the mismatch.
        kind: io::ErrorKind,
        /// See the variant documentation.
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
        /// Operation or IO step that failed.
        operation: &'static str,
        /// See the variant documentation.
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
