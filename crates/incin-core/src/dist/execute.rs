//! Bucketed data-parallel execution lowering (issue #97, in-process slice).
//!
//! [`DataParallelPlan`](crate::dist::data_parallel::DataParallelPlan) lists
//! every gradient tensor in launch order; this module groups that ordered
//! list into all-reduce buckets. One collective per parameter is dominated
//! by launch latency on any model with more than a handful of parameters,
//! so the execution path reduces one bucket - the concatenation of its
//! tensors - per collective instead.
//!
//! The lowering is transport-neutral on purpose: it says which tensors
//! share a collective and in what order, while the caller owns running the
//! collective (the deterministic reference transport in-process, NCCL
//! across hosts). Buckets partition the plan's traversal order, which is
//! also reverse-topological order, so a bucket is launchable as soon as
//! both ranks produced its last tensor - the streaming order an
//! overlapped backward pass would consume. Wall-clock overlap with the
//! backward pass itself needs autograd streaming hooks that do not exist
//! yet; what this module guarantees is the launchable order, and the
//! rendezvous records that it launched buckets incrementally.
//!
//! Byte accounting uses the host `f64` protocol width
//! ([`all_reduce_model_gradients`](crate::dist::sync::all_reduce_model_gradients)
//! reduces `f64` vectors regardless of the parameter dtype), so a
//! [`BucketPolicy::Bytes`] budget is comparable across mixed-dtype models.

use alloc::vec::Vec;

use crate::dist::data_parallel::DataParallelPlan;
use crate::dist::plan::SequenceToken;

/// How a [`DataParallelPlan`] gradient list is grouped into all-reduce buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BucketPolicy {
    /// One collective per gradient tensor: the milestone floor, kept so a
    /// benchmark can show what bucketing buys.
    SingleTensor,
    /// Greedily fill each bucket until it holds at least this many protocol
    /// bytes (`elements * size_of::<f64>()`), then close it. A single tensor
    /// larger than the budget still gets its own bucket rather than being
    /// split: ranks could not agree on a split point without a second
    /// protocol.
    Bytes {
        /// Minimum bucket payload in `f64` protocol bytes. Must be nonzero.
        max_bytes: usize,
    },
    /// Greedily fill each bucket with this many tensors, then close it. The
    /// trailing bucket holds the remainder.
    Count {
        /// Tensors per bucket. Must be nonzero.
        max_tensors: usize,
    },
}

impl BucketPolicy {
    /// The issue's sketched default: 25 MiB buckets.
    pub const DEFAULT_BYTES: usize = 25 * 1024 * 1024;

    /// Refuses a zero budget before any plan is bucketed.
    pub fn validate(self) -> Result<(), BucketError> {
        match self {
            Self::SingleTensor => Ok(()),
            Self::Bytes { max_bytes: 0 } => Err(BucketError::ZeroBytes),
            Self::Bytes { .. } => Ok(()),
            Self::Count { max_tensors: 0 } => Err(BucketError::ZeroTensors),
            Self::Count { .. } => Ok(()),
        }
    }
}

impl Default for BucketPolicy {
    fn default() -> Self {
        Self::Bytes {
            max_bytes: Self::DEFAULT_BYTES,
        }
    }
}

/// Why a plan could not be bucketed.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketError {
    /// A [`BucketPolicy::Bytes`] budget of zero can never close: even an
    /// empty bucket already meets it, so ranks could not agree on
    /// boundaries.
    #[error("bucket byte budget must be nonzero")]
    ZeroBytes,
    /// A [`BucketPolicy::Count`] budget of zero admits no tensor per bucket.
    #[error("bucket tensor budget must be nonzero")]
    ZeroTensors,
}

/// One all-reduce bucket: a contiguous run of the plan's gradient order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientBucket {
    index: usize,
    tensors: usize,
    elements: usize,
    first_sequence: SequenceToken,
    last_sequence: SequenceToken,
}

impl GradientBucket {
    /// Position in the launch order, counting from zero.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }

    /// How many gradient tensors this bucket holds.
    #[must_use]
    pub const fn tensor_count(self) -> usize {
        self.tensors
    }

    /// Total `f64` elements across the bucket's tensors.
    #[must_use]
    pub const fn elements(self) -> usize {
        self.elements
    }

    /// Protocol bytes this bucket's collective carries.
    #[must_use]
    pub const fn bytes(self) -> usize {
        self.elements * core::mem::size_of::<f64>()
    }

    /// Sequence token of the bucket's first tensor.
    #[must_use]
    pub const fn first_sequence(self) -> SequenceToken {
        self.first_sequence
    }

    /// Sequence token of the bucket's last tensor.
    #[must_use]
    pub const fn last_sequence(self) -> SequenceToken {
        self.last_sequence
    }
}

/// A [`DataParallelPlan`] lowered to launch-ordered all-reduce buckets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketPlan {
    buckets: Vec<GradientBucket>,
    tensors: usize,
}

impl BucketPlan {
    /// The buckets in launch order.
    #[must_use]
    pub fn buckets(&self) -> &[GradientBucket] {
        &self.buckets
    }

    /// How many collectives the lowered run issues.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    /// Whether the plan lowered to no bucket. Unreachable through
    /// [`bucket_plan`]: the builder refuses an empty gradient list, so a
    /// lowered plan always issues at least one collective.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// How many gradient tensors were bucketed.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensors
    }

    /// How many per-tensor collectives bucketing saved:
    /// `tensors - buckets`, zero under [`BucketPolicy::SingleTensor`].
    #[must_use]
    pub fn collectives_saved(&self) -> usize {
        self.tensors.saturating_sub(self.buckets.len())
    }
}

/// Lower a [`DataParallelPlan`] to launch-ordered all-reduce buckets.
///
/// Greedy and deterministic: tensors are visited once, in plan order, and
/// each bucket closes as soon as the policy is satisfied, so both ranks
/// derive identical boundaries from identical plans without communicating.
///
/// # Errors
///
/// [`BucketError::ZeroBytes`] or [`BucketError::ZeroTensors`] for a zero
/// budget. A plan with no gradients cannot reach this function: the
/// builder refuses it first.
pub fn bucket_plan(
    plan: &DataParallelPlan,
    policy: BucketPolicy,
) -> Result<BucketPlan, BucketError> {
    policy.validate()?;
    let gradients = plan.gradients();

    let mut buckets = Vec::new();
    // The open bucket's accumulation: tensor count, elements, first index.
    let mut open: Option<(usize, usize, usize)> = None;

    let close = |buckets: &mut Vec<GradientBucket>,
                 start: usize,
                 end: usize,
                 gradients: &[crate::dist::data_parallel::GradientDescriptor]| {
        let tensors = end - start;
        let elements = gradients[start..end]
            .iter()
            .map(|gradient| gradient.elements())
            .sum();
        buckets.push(GradientBucket {
            index: buckets.len(),
            tensors,
            elements,
            first_sequence: gradients[start].sequence(),
            last_sequence: gradients[end - 1].sequence(),
        });
    };

    for (position, gradient) in gradients.iter().enumerate() {
        let (count, elements, start) = match open {
            Some(open) => open,
            None => (0, 0, position),
        };
        let count = count + 1;
        let elements = elements + gradient.elements();
        let closed = match policy {
            BucketPolicy::SingleTensor => true,
            BucketPolicy::Bytes { max_bytes } => {
                elements * core::mem::size_of::<f64>() >= max_bytes
            }
            BucketPolicy::Count { max_tensors } => count >= max_tensors,
        };
        if closed {
            close(&mut buckets, start, position + 1, gradients);
            open = None;
        } else {
            open = Some((count, elements, start));
        }
    }
    // The trailing partial bucket always closes: every tensor launches.
    if let Some((_, _, start)) = open {
        close(&mut buckets, start, gradients.len(), gradients);
    }

    Ok(BucketPlan {
        buckets,
        tensors: gradients.len(),
    })
}
