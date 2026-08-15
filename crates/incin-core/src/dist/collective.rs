//! Transport-neutral collective vocabulary.
//!
//! A plan and a backend must agree on these values without either crate
//! depending on the other. Static dtype support is expressed by
//! [`CollectiveDType`]; [`validate_collective_dtype`] is its `Dyn`
//! counterpart.

use crate::dist::placement::{Max, Mean, Min, PartialReduction, Prod, Sum};
use crate::exec::ReduceOp;
use crate::shapes::Dyn;
use crate::tensor::dtype::{DType, DTypeId};

/// Dtypes with scalar collective semantics.
///
/// A static dtype absent from this trait is rejected by trait resolution.
/// [`Dyn`] implements it and must be checked with
/// [`validate_collective_dtype`] before a descriptor is minted.
pub trait CollectiveDType: DType {}

impl CollectiveDType for f32 {}
impl CollectiveDType for f64 {}
impl CollectiveDType for u8 {}
impl CollectiveDType for u32 {}
impl CollectiveDType for i64 {}
impl CollectiveDType for half::f16 {}
impl CollectiveDType for half::bf16 {}
impl CollectiveDType for Dyn {}

/// Compile-time proof that dtype `Self` supports reduction marker `R`.
///
/// Scalar integer types support exact sum/product/min/max operations, but mean
/// gradients require a floating representation. [`Dyn`] implements every
/// statically expressible reduction and is checked by
/// [`validate_collective_reduction`] after its runtime dtype is known.
pub trait CollectiveReductionDType<R: PartialReduction>: CollectiveDType {}

impl<K: CollectiveDType> CollectiveReductionDType<Sum> for K {}
impl<K: CollectiveDType> CollectiveReductionDType<Prod> for K {}
impl<K: CollectiveDType> CollectiveReductionDType<Max> for K {}
impl<K: CollectiveDType> CollectiveReductionDType<Min> for K {}
impl CollectiveReductionDType<Mean> for f32 {}
impl CollectiveReductionDType<Mean> for f64 {}
impl CollectiveReductionDType<Mean> for half::f16 {}
impl CollectiveReductionDType<Mean> for half::bf16 {}
impl CollectiveReductionDType<Mean> for Dyn {}

/// Runtime counterpart of [`CollectiveDType`].
///
/// Block-quantized values need a block/layout-aware message contract, which
/// the scalar collective interface deliberately does not claim.
pub const fn validate_collective_dtype(dtype: DTypeId) -> Result<(), CollectiveError> {
    match dtype {
        DTypeId::U8
        | DTypeId::U32
        | DTypeId::I64
        | DTypeId::BF16
        | DTypeId::F16
        | DTypeId::F32
        | DTypeId::F64
        | DTypeId::Bool => Ok(()),
        DTypeId::Q8_0 => Err(CollectiveError::UnsupportedDType { dtype }),
    }
}

/// Runtime counterpart of [`CollectiveReductionDType`].
pub fn validate_collective_reduction(dtype: DTypeId, op: ReduceOp) -> Result<(), CollectiveError> {
    validate_collective_dtype(dtype)?;
    if matches!(op, ReduceOp::Mean) && matches!(dtype, DTypeId::U8 | DTypeId::U32 | DTypeId::I64) {
        Err(CollectiveError::UnsupportedReduction { dtype, op })
    } else {
        Ok(())
    }
}

/// Stable identity and cardinality of one ordered collective group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupId {
    token: u64,
    ranks: usize,
}

impl GroupId {
    /// Build a non-empty group.
    pub const fn new(token: u64, ranks: usize) -> Result<Self, CollectiveError> {
        if ranks == 0 {
            Err(CollectiveError::EmptyGroup)
        } else {
            Ok(Self { token, ranks })
        }
    }

    /// Stable group token supplied by the planner.
    #[must_use]
    pub const fn token(self) -> u64 {
        self.token
    }

    /// Number of ordered ranks in the group.
    #[must_use]
    pub const fn ranks(self) -> usize {
        self.ranks
    }
}

/// Logical communication stream selected by a collective plan.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(u32);

impl StreamId {
    /// Build a stream identifier.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Numeric stream identifier.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A collective and the reduction semantics it carries.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollectiveKind {
    /// Reduce and return the result to every rank.
    AllReduce(ReduceOp),
    /// Concatenate rank-local shards and return the whole value to every rank.
    AllGather,
    /// Reduce complete values and scatter equal contiguous shards.
    ReduceScatter(ReduceOp),
    /// Exchange equal contiguous chunks among every rank.
    AllToAll,
    /// Transfer one buffer between two ordered ranks.
    ///
    /// This is represented as one global operation rather than separate
    /// rank-local `Send` and `Recv` descriptors so every rank hashes and
    /// preflights identical plan data before either side launches.
    SendRecv {
        /// Rank that owns the input payload.
        source: usize,
        /// Rank that receives the payload.
        destination: usize,
    },
}

impl CollectiveKind {
    /// Reverse-mode collective corresponding to this forward collective.
    #[must_use]
    pub const fn adjoint(self) -> Self {
        match self {
            Self::AllReduce(op) => Self::AllReduce(op),
            Self::AllGather => Self::ReduceScatter(ReduceOp::Sum),
            Self::ReduceScatter(_) => Self::AllGather,
            Self::AllToAll => Self::AllToAll,
            Self::SendRecv {
                source,
                destination,
            } => Self::SendRecv {
                source: destination,
                destination: source,
            },
        }
    }
}

/// Failures shared by collective planning and execution.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum CollectiveError {
    /// A collective group cannot contain zero ranks.
    #[error("a collective group must contain at least one rank")]
    EmptyGroup,
    /// The runtime dtype has no scalar collective encoding.
    #[error("dtype {dtype:?} has no scalar collective encoding")]
    UnsupportedDType {
        /// Rejected runtime dtype.
        dtype: DTypeId,
    },
    /// The number of submitted buffers differs from the group cardinality.
    #[error("collective group requires {expected} rank buffers, found {found}")]
    InputCount {
        /// Group cardinality.
        expected: usize,
        /// Submitted buffer count.
        found: usize,
    },
    /// Rank buffers disagree on their runtime dtype.
    #[error("rank {rank} has dtype {found:?}, expected {expected:?}")]
    DTypeMismatch {
        /// Rank whose buffer disagreed.
        rank: usize,
        /// Dtype established by rank zero.
        expected: DTypeId,
        /// Dtype found at `rank`.
        found: DTypeId,
    },
    /// A typed buffer's runtime dtype does not match `K`.
    #[error("buffer values have dtype {values:?}, but the tensor dtype is {typed:?}")]
    BufferDType {
        /// Dtype represented by the value payload.
        values: DTypeId,
        /// Dtype selected statically or through `Dyn`.
        typed: DTypeId,
    },
    /// Rank buffers disagree on element count.
    #[error("rank {rank} has {found} elements, expected {expected}")]
    ElementCount {
        /// Rank whose buffer disagreed.
        rank: usize,
        /// Element count established by rank zero.
        expected: usize,
        /// Element count found at `rank`.
        found: usize,
    },
    /// A scatter or all-to-all cannot divide a buffer evenly.
    #[error("{elements} elements cannot be divided across {ranks} ranks")]
    NonDivisible {
        /// Complete buffer element count.
        elements: usize,
        /// Group cardinality.
        ranks: usize,
    },
    /// A point-to-point transfer cannot send a rank to itself.
    #[error("point-to-point source and destination are both rank {rank}")]
    SamePeer {
        /// Rejected rank.
        rank: usize,
    },
    /// A point-to-point endpoint is outside its ordered group.
    #[error("point-to-point {endpoint} rank {rank} is outside a group of {ranks} ranks")]
    PeerOutOfRange {
        /// Whether this is the source or destination endpoint.
        endpoint: &'static str,
        /// Rejected rank.
        rank: usize,
        /// Group cardinality.
        ranks: usize,
    },
    /// The reduction has no meaning for this dtype.
    #[error("reduction {op:?} is unsupported for dtype {dtype:?}")]
    UnsupportedReduction {
        /// Runtime dtype.
        dtype: DTypeId,
        /// Requested reduction.
        op: ReduceOp,
    },
    /// Integer reference reduction overflowed its dtype.
    #[error("reduction {op:?} overflowed dtype {dtype:?} at element {element}")]
    ReductionOverflow {
        /// Runtime dtype.
        dtype: DTypeId,
        /// Requested reduction.
        op: ReduceOp,
        /// Element offset whose accumulation overflowed.
        element: usize,
    },
}
