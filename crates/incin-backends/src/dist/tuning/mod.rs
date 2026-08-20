//! Coordinated two-rank collective-tuning contracts.
//!
//! This module does not claim to control NCCL algorithms yet. It owns the
//! transport-neutral coordination rules that must be true before a measured
//! winner may enter a cache: every candidate is legal for the same problem,
//! every rank measures that candidate, scoring uses the median of per-sample
//! maximum-rank duration, measurement buffers are unchanged, and every rank
//! votes for the exact same result.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `algorithms` is the
//! algorithm/protocol vocabulary and the `StaticCollectiveTuning` marker
//! types every candidate is built from; `problem` is what is being tuned
//! (`CollectiveTuningKey`, `CollectiveTuningProblem`, and the candidate/
//! budget descriptors); `coordination` is the round-by-round measurement
//! and commit protocol itself, plus its private scoring/hashing helpers;
//! `error` is the failure vocabulary shared across all three.

use alloc::{string::String, vec::Vec};
use core::marker::PhantomData;

use incin_core::dist::mesh::TopologyFingerprint;
use incin_core::dist::placement::PartialReduction;
use incin_core::dist::{
    CollectiveDType, CollectiveError, CollectiveKind, CollectiveReductionDType, GroupId,
    ShardDivisible,
};
use incin_core::exec::{Determinism, ReduceOp};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::{ConstDType, DTypeId};
use incin_core::typenum::{B1, IsLessOrEqual, NonZero, PowerOfTwo, U2, U32, U4294967295, Unsigned};

mod algorithms;
mod coordination;
mod error;
mod problem;

pub use algorithms::*;
pub use coordination::*;
pub use error::*;
pub use problem::*;
