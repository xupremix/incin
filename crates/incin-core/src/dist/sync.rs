//! Data-parallel gradient synchronization for single- and multi-rank training.
//!
//! [`all_reduce_model_gradients`] walks a model's parameters in
//! [`VisitParameters`] order, reads every gradient that exists, hands the
//! flat `f64` values to a [`GradientSynchronizer`], and writes the reduced
//! values back over the gradients. The synchronizer is the pluggable
//! transport boundary: this module owns the read-reduce-write protocol and
//! the dtype contract; the synchronizer owns agreeing with its peers.
//!
//! # What is proven here and what is hardware-gated
//!
//! Proven on the CPU: identity reduction through a one-rank synchronizer,
//! element-wise two-rank mean arithmetic against a scripted peer, and
//! equality between the mean of two shard gradients and the full-batch
//! gradient. Those are the guarantees of this protocol; they do not
//! require network hardware.
//!
//! The following are hardware-gated gaps - documented, not claimed, until
//! real transports wire through:
//!
//! - **No collective transport is attached.** `DST-005`'s descriptors and
//!   transports exist behind feature gates, but nothing in this module
//!   opens a communicator. A multi-rank synchronizer is a caller-supplied
//!   [`GradientSynchronizer`] implementation; none ships wired to NCCL,
//!   MPI, or any other wire protocol.
//! - **Host round-trip per tensor.** Gradients are read into `f64` host
//!   vectors, reduced, and encoded back; device-resident all-reduce is not
//!   implemented. Bucketing across the step's gradients is available
//!   through [`all_reduce_mean_batch`](GradientSynchronizer::all_reduce_mean_batch)
//!   overrides - the default loops one collective per tensor - and the
//!   reference-transport rendezvous in `incin`'s trainer uses it; overlap
//!   with the backward pass itself needs autograd streaming hooks that do
//!   not exist yet.
//! - **No per-rank data sharding.** Each rank must be handed its own data
//!   shard by the caller; this protocol only reduces the gradients those
//!   shards produce.
//! - **Gradient presence must match across ranks.** Parameters without a
//!   gradient are skipped (frozen weights, branches no loss depends on).
//!   Ranks whose models disagree about which gradients exist will issue
//!   different numbers of collectives and can hang: identical models per
//!   rank are required, and presence parity is not checked here.
//! - **Module buffers are not synchronized.** Traversal visits parameters
//!   only; batch-norm running statistics and similar buffers stay local
//!   to each rank.
//! - **Only `f32`, `f64`, `f16`, and `bf16` gradients are supported**, the
//!   same set `DataParallelDType` accepts. Anything else fails with
//!   [`SyncError::UnsupportedDType`] at the first parameter that actually
//!   carries such a gradient, before that parameter is reduced.
//! - **Reduction happens in `f64` on the host and is stored back in the
//!   parameter's own dtype**, so a multi-rank mean of `f16` gradients
//!   rounds to `f16` on write-back.
//!
//! # FSDP / ZeRO-sharded execution (#99)
//!
//! [`reduce_scatter_model_gradients`], [`mask_gradients_to_owned_shard`],
//! and [`all_gather_model_parameters`] lower the ZeRO-1 and ZeRO-2 stages
//! of an FSDP plan onto the same visitor/transport boundary. ZeRO-2
//! reduce-scatters every gradient: this rank ends the walk owning exactly
//! its contiguous slice and zeros elsewhere, so the round retains
//! `1/world_size` of the gradient bytes an all-reduce round would retain -
//! measured directly by [`ShardedGradients::retained_bytes`] against
//! [`ShardedGradients::full_bytes`]. ZeRO-1 all-reduces through
//! [`GradientSynchronizer`] and then masks gradients to the owned slice.
//! After every optimizer step [`all_gather_model_parameters`] rebuilds each
//! rank's full replica from the owners, which is what keeps non-owned
//! slices honest against rank-local optimizer effects (weight decay, for
//! instance, would otherwise drift them until the next gather).
//!
//! Proven on the CPU: reduce-scatter and all-gather arithmetic against
//! scripted two-rank peers, the `1/N`-versus-`N` byte measurement above,
//! divisibility refusals issued before the offending parameter's
//! collective, and a two-rank ZeRO-2 trajectory equal to the
//! single-device full-batch reference.
//!
//! Out of scope here, in addition to the transport gaps above:
//!
//! - **ZeRO-3 is not executable.** Parameters keep full storage on every
//!   rank; persistent parameter sharding and the gather/free lifecycle are
//!   not implemented, and the trainer that receives a ZeRO-3 request
//!   refuses rather than approximates it.
//! - **Optimizer state stays full-size per rank.** The owned-slice
//!   authority model yields the right trajectory with replicated state
//!   tensors, not the `1/N` optimizer-state memory ZeRO counts as its
//!   headline number.
//! - **Global gradient clipping is unsupported while sharded.** A norm
//!   taken over masked (partly zeroed) gradients is the wrong norm.
//! - **Every synchronized parameter must divide evenly** across the world.
//!   A length that does not is refused with
//!   [`SyncError::NonDivisibleShard`] before that parameter's collective;
//!   the padded `div_ceil` shards `FsdpPlan` can express in planning
//!   reports are not executable here.
//! - **No activation-recompute composition.** The autograd has no
//!   checkpoint/recompute machinery yet, so checkpoint x FSDP cannot be
//!   wired or tested.
//! - **Tensor- and pipeline-parallel execution are other tiers** (#99):
//!   TP waits on partitioned matmul (#85/#90); PP consumes schedule clocks
//!   end to end.

use crate::autograd::Gradients;
use crate::err::{Error, Result};
use crate::nn::param::{Param, TrainState};
use crate::nn::{ParameterVisitor, StatePath, VisitParameters};
use crate::shapes::Shape;
use crate::tensor::backend::{AutogradBackend, HostInterop, VariableBackend};
use crate::tensor::dtype::{DType, DTypeDescriptor, DTypeId};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Why a gradient synchronization round could not complete.
///
/// Every variant is a refusal: there is no variant meaning "reduced
/// partially" or "skipped a rank".
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SyncError {
    /// A parameter's gradient dtype cannot be encoded by this protocol.
    ///
    /// The supported set is `f32`, `f64`, `f16`, and `bf16` - the same
    /// dtypes `DataParallelDType` accepts - checked before that
    /// parameter's collective runs, so nothing about it is reduced.
    #[error(
        "gradient dtype {dtype:?} cannot be synchronized; only f32, f64, f16, and bf16 \
         gradients are supported"
    )]
    UnsupportedDType {
        /// The descriptor of the gradient that cannot be encoded.
        dtype: DTypeDescriptor,
    },
    /// The synchronizer's rank is not inside its own world.
    #[error("rank {rank} is out of range for world size {world_size}")]
    RankOutOfRange {
        /// The rank the synchronizer reported.
        rank: usize,
        /// The world size the synchronizer reported.
        world_size: usize,
    },
    /// The synchronizer itself refused the collective.
    ///
    /// Carries whatever the transport layer reported - a rendezvous
    /// failure, a cursor mismatch, a peer that went away.
    #[error("gradient synchronizer failed: {message}")]
    Synchronizer {
        /// The transport's own message.
        message: String,
    },
    /// A backend operation failed while reading, rebuilding, or writing a
    /// gradient - not the synchronizer's fault, and not a dtype this
    /// protocol rejected either.
    #[error("backend error while synchronizing gradients: {message}")]
    Backend {
        /// The underlying backend error, stringified.
        message: String,
    },
    /// An FSDP buffer's element count does not divide across the world.
    ///
    /// Sharded execution splits every parameter into exactly
    /// `world_size` contiguous slices of equal length; a length with a
    /// remainder has no slice layout the ranks could agree on. Checked
    /// before that parameter's collective runs, so nothing about it is
    /// reduced or gathered - the same fail-closed rule the dtype check
    /// follows. The padded (`div_ceil`) shards `FsdpPlan` can express in
    /// planning reports are not executable.
    #[error(
        "an FSDP buffer of {elements} elements does not divide across {world_size} ranks; \
         every synchronized parameter must split into equal-sized shards"
    )]
    NonDivisibleShard {
        /// The element count that could not be split.
        elements: usize,
        /// The world size the synchronizer reported.
        world_size: usize,
    },
}

/// One rank's view of an all-reduce over gradient values.
///
/// Implementations must be callable once per gradient-bearing parameter,
/// in the model's [`VisitParameters`] traversal order, and must leave
/// `values` holding the element-wise mean across every rank of
/// [`world_size`](Self::world_size). Ranks with identical models
/// therefore see identical call sequences.
///
/// The f64 host view is deliberate: it is the one representation every
/// backend can round-trip without a device-side collective. Adapters that
/// keep values on the device belong behind an implementation of this
/// trait, not in this protocol.
///
/// [`all_reduce_model_gradients`] reads the whole step's gradients, calls
/// [`all_reduce_mean_batch`](Self::all_reduce_mean_batch) once, and writes
/// the reduced values back. The default batch implementation loops
/// [`all_reduce_mean`](Self::all_reduce_mean) in order, so existing
/// implementations keep their one-collective-per-tensor behavior;
/// overriding it is how a transport buckets several tensors into one
/// collective.
///
/// # Hardware-gated gap
///
/// The in-process two-rank transport adapter lives in `incin`'s trainer
/// (`ReferenceDataParallel` under the `distributed-reference` feature -
/// outside this crate so the core stays transport-neutral); wiring a
/// real NCCL (or other collective) adapter through
/// [`all_reduce_model_gradients`] is `DST-005`'s work and is not runnable
/// on this project's current hardware. The interface, the single-rank
/// identity path, and the two-rank arithmetic against scripted peers are
/// what this row proves; the transport between real ranks is not.
pub trait GradientSynchronizer: core::fmt::Debug + Send + Sync {
    /// How many ranks take part in each collective, including this one.
    ///
    /// [`all_reduce_model_gradients`] refuses to run when [`rank`](Self::rank)
    /// is not below this value, and a trainer that has one attached
    /// refuses a plan whose device count differs.
    fn world_size(&self) -> usize;

    /// This participant's index in `0..world_size()`.
    fn rank(&self) -> usize;

    /// Replaces every value with its element-wise mean across all ranks.
    ///
    /// Called once per gradient tensor. On success `values` must hold the
    /// mean; on failure the whole synchronization round fails and the
    /// training step is refused.
    fn all_reduce_mean(&self, values: &mut [f64]) -> core::result::Result<(), SyncError>;

    /// Replaces every tensor's values with their element-wise means.
    ///
    /// Called once per step with the whole step's gradients in
    /// [`VisitParameters`] traversal order. On success each tensor must
    /// hold its mean; on failure the whole round fails and the training
    /// step is refused.
    ///
    /// The default loops [`all_reduce_mean`](Self::all_reduce_mean) in
    /// order, which is exactly the one-collective-per-tensor behavior the
    /// scripted-peer tests pin. Override it to bucket several tensors
    /// into fewer transport collectives: the batch is the whole step, so
    /// a trailing partial bucket always closes and a one-tensor batch
    /// always progresses.
    fn all_reduce_mean_batch(&self, batch: &mut [Vec<f64>]) -> core::result::Result<(), SyncError> {
        for values in batch.iter_mut() {
            self.all_reduce_mean(values)?;
        }
        Ok(())
    }
}

/// One rank's view of the FSDP collectives: reduce-scatter and all-gather.
///
/// Extends [`GradientSynchronizer`] with the two operations ZeRO-sharded
/// execution needs. Like the all-reduce path, this crate ships only the
/// protocol and the single-rank identity implementation
/// ([`crate::dist::sync`]-side tests script the two-rank arithmetic): a
/// transport-backed implementation is a caller-supplied adapter.
///
/// # Hardware-gated gap
///
/// No NCCL- (or other transport-) backed implementation ships here;
/// multi-host reduce-scatter/all-gather adapters are not runnable on this
/// project's current hardware. The interface, the one-rank identity path,
/// and the scripted two-rank arithmetic are what the CPU tests prove.
pub trait FsdpSynchronizer: GradientSynchronizer {
    /// Returns this rank's contiguous shard of the element-wise mean of
    /// `values` across [`world_size`](GradientSynchronizer::world_size)
    /// ranks.
    ///
    /// Called once per gradient-bearing parameter, in [`VisitParameters`]
    /// order, only with lengths divisible by the world size. On success
    /// the returned vector must hold exactly `values.len() / world_size`
    /// elements: this rank's slice of the mean, ordered by element index
    /// the same way the corresponding slice of `values` is. On failure the
    /// whole round fails and the training step is refused.
    fn reduce_scatter_mean(&self, values: &[f64]) -> core::result::Result<Vec<f64>, SyncError>;

    /// Concatenates this rank's shard with every peer's into the full
    /// sequence, ordered by rank.
    ///
    /// Called once per gradient-bearing parameter, in [`VisitParameters`]
    /// order, with the shard this rank would own after a reduce-scatter of
    /// that parameter. On success the returned vector must hold exactly
    /// `shard.len() * world_size` elements - the parameter's full value,
    /// rank 0's slice first. On failure the whole round fails and the
    /// training step is refused.
    fn all_gather(&self, shard: &[f64]) -> core::result::Result<Vec<f64>, SyncError>;
}

/// What one reduce-scatter round left this rank holding.
///
/// One shard per gradient-bearing parameter, in [`VisitParameters`] order -
/// the same walk [`all_reduce_model_gradients`] takes. The byte accessors
/// exist so a caller can measure the difference the issue names: a
/// reduce-scatter round retains [`retained_bytes`](Self::retained_bytes)
/// on this rank, where the all-reduce round it replaced would have
/// retained [`full_bytes`](Self::full_bytes) - `world_size` times as much.
#[derive(Debug, Clone, PartialEq)]
pub struct ShardedGradients {
    rank: usize,
    world_size: usize,
    shards: Vec<Vec<f64>>,
    full_elements: usize,
    owned_elements: usize,
}

impl ShardedGradients {
    /// This rank's index in `0..world_size()`.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// How many ranks the round reduced across.
    #[must_use]
    pub const fn world_size(&self) -> usize {
        self.world_size
    }

    /// The owned slice of each gradient, in [`VisitParameters`] order.
    #[must_use]
    pub fn shards(&self) -> &[Vec<f64>] {
        &self.shards
    }

    /// How many gradients the round sharded.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.shards.len()
    }

    /// Total elements across every gradient the round sharded.
    #[must_use]
    pub const fn full_elements(&self) -> usize {
        self.full_elements
    }

    /// Elements this rank owns after the round (`full / world_size`).
    #[must_use]
    pub const fn owned_elements(&self) -> usize {
        self.owned_elements
    }

    /// `f64` bytes this rank retains after the round.
    ///
    /// The reduce-scatter result: one rank's slice of every gradient.
    #[must_use]
    pub const fn retained_bytes(&self) -> usize {
        self.owned_elements * core::mem::size_of::<f64>()
    }

    /// `f64` bytes an all-reduce of the same gradients would have
    /// retained on this rank.
    ///
    /// The full gradients - `world_size()` times
    /// [`retained_bytes`](Self::retained_bytes) whenever the round sharded
    /// anything, which is the byte-fanout claim reduce-scatter exists to
    /// make.
    #[must_use]
    pub const fn full_bytes(&self) -> usize {
        self.full_elements * core::mem::size_of::<f64>()
    }
}

/// Synchronizes a backward pass's gradients through `sync`.
///
/// Walks `model` in [`VisitParameters`] order - the same order every rank
/// with the same model walks - reads every gradient that exists into host
/// `f64` vectors, reduces the whole step with one
/// [`all_reduce_mean_batch`](GradientSynchronizer::all_reduce_mean_batch)
/// call, and encodes the reduced values back into the parameters' dtypes.
/// Parameters without a gradient are skipped without contributing to the
/// batch; see the module docs for why ranks must agree on which those
/// are.
///
/// Call it after `backward()` and before the optimizer step.
///
/// # Errors
///
/// [`SyncError::RankOutOfRange`] when `sync`'s rank is outside its world,
/// [`SyncError::UnsupportedDType`] for a gradient dtype outside
/// `f32`/`f64`/`f16`/`bf16` - checked on the read pass, before any
/// collective runs - [`SyncError::Synchronizer`] when the synchronizer
/// refuses, and [`SyncError::Backend`] when the backend cannot read,
/// rebuild, or write a gradient.
pub fn all_reduce_model_gradients<B, M>(
    model: &M,
    grads: &mut Gradients<B>,
    sync: &dyn GradientSynchronizer,
) -> core::result::Result<(), SyncError>
where
    B: VariableBackend + AutogradBackend + HostInterop,
    M: VisitParameters<B>,
{
    let world_size = sync.world_size();
    let rank = sync.rank();
    if rank >= world_size {
        return Err(SyncError::RankOutOfRange { rank, world_size });
    }

    let mut values = {
        let mut reader = ReadVisitor {
            grads: grads.as_backend_mut(),
            values: Vec::new(),
            failure: None,
        };
        if let Err(error) = model.visit_parameters(&StatePath::root(), &mut reader) {
            return Err(reader.failure.unwrap_or_else(|| SyncError::Backend {
                message: error.to_string(),
            }));
        }
        reader.values
    };

    sync.all_reduce_mean_batch(&mut values)?;

    let mut writer = WriteVisitor {
        grads: grads.as_backend_mut(),
        values,
        index: 0,
        failure: None,
    };
    match model.visit_parameters(&StatePath::root(), &mut writer) {
        Ok(()) => Ok(()),
        Err(error) => Err(writer.failure.unwrap_or_else(|| SyncError::Backend {
            message: error.to_string(),
        })),
    }
}

/// The read pass: gradient tensors to host `f64` vectors, in traversal
/// order, refusing unsupported dtypes before any collective runs.
struct ReadVisitor<'a, B: VariableBackend + AutogradBackend + HostInterop> {
    grads: &'a mut B::Grads,
    values: Vec<Vec<f64>>,
    failure: Option<SyncError>,
}

impl<B> ReadVisitor<'_, B>
where
    B: VariableBackend + AutogradBackend + HostInterop,
{
    /// Records `error` as the typed failure and returns a traversal error
    /// carrying its message, which stops the walk.
    fn fail(&mut self, error: SyncError) -> Error {
        let message = format!("{error}");
        self.failure = Some(error);
        Error::Msg(message)
    }
}

impl<B> ParameterVisitor<B> for ReadVisitor<'_, B>
where
    B: VariableBackend + AutogradBackend + HostInterop,
{
    fn visit_param<S, K, Train>(
        &mut self,
        _path: &StatePath,
        param: &Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: Shape,
        K: DType,
        Train: TrainState,
    {
        let variable = param
            .variable_any()
            .downcast_ref::<B::Var<K>>()
            .ok_or_else(|| Error::InternalInvariant {
                operation: "synchronize gradients",
                reason: "backend variable type did not match the parameter dtype",
            })?;
        let key = B::var_as_tensor::<K>(variable)?;

        // A parameter with no gradient contributes nothing to the batch:
        // frozen weights and unused branches skip here. Ranks must skip
        // the same parameters (module docs).
        let Some(grad) = B::get_grad::<K>(&key, &*self.grads)? else {
            return Ok(());
        };

        let dtype = param.dtype_descriptor();
        if let Err(error) = ensure_supported_dtype(dtype) {
            return Err(self.fail(error));
        }

        self.values.push(B::float_to_vec1::<K>(&grad)?);
        Ok(())
    }
}

/// The write pass: reduced `f64` vectors back over the gradients, in the
/// same traversal order the read pass took them in.
struct WriteVisitor<'a, B: VariableBackend + AutogradBackend + HostInterop> {
    grads: &'a mut B::Grads,
    values: Vec<Vec<f64>>,
    index: usize,
    failure: Option<SyncError>,
}

impl<B> WriteVisitor<'_, B>
where
    B: VariableBackend + AutogradBackend + HostInterop,
{
    fn fail(&mut self, error: SyncError) -> Error {
        let message = format!("{error}");
        self.failure = Some(error);
        Error::Msg(message)
    }
}

impl<B> ParameterVisitor<B> for WriteVisitor<'_, B>
where
    B: VariableBackend + AutogradBackend + HostInterop,
{
    fn visit_param<S, K, Train>(
        &mut self,
        _path: &StatePath,
        param: &Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: Shape,
        K: DType,
        Train: TrainState,
    {
        let variable = param
            .variable_any()
            .downcast_ref::<B::Var<K>>()
            .ok_or_else(|| Error::InternalInvariant {
                operation: "synchronize gradients",
                reason: "backend variable type did not match the parameter dtype",
            })?;
        let key = B::var_as_tensor::<K>(variable)?;

        let Some(grad) = B::get_grad::<K>(&key, &*self.grads)? else {
            return Ok(());
        };

        let Some(reduced) = self.values.get(self.index) else {
            return Err(self.fail(SyncError::Backend {
                message: "gradient count changed between the synchronization read and write passes"
                    .to_string(),
            }));
        };
        self.index += 1;
        let expected = B::float_to_vec1::<K>(&grad)?.len();
        if reduced.len() != expected {
            return Err(self.fail(SyncError::Synchronizer {
                message: format!(
                    "gradient synchronizer returned {} element(s) for a gradient of \
                     {expected} element(s); the batch contract keeps lengths",
                    reduced.len(),
                ),
            }));
        }

        let dtype = param.dtype_descriptor();
        let shape = B::host_shape(&grad);
        let Some(device) = B::host_storage_device(&grad) else {
            return Err(self.fail(SyncError::Backend {
                message: format!(
                    "backend did not report the device of a gradient with dtype {dtype:?}"
                ),
            }));
        };
        let bytes = match encode_reduced(dtype, reduced) {
            Ok(bytes) => bytes,
            Err(error) => return Err(self.fail(error)),
        };
        let rebuilt = B::from_bytes::<K>(&bytes, shape.as_ref(), dtype, &device)?;
        B::set_grad::<K>(&key, self.grads, rebuilt)?;
        Ok(())
    }
}

/// Refuses a dtype outside the synchronized float set.
///
/// The one gate every synchronized walk - all-reduce, reduce-scatter,
/// mask, and gather - runs before its per-parameter work, so a parameter
/// this protocol cannot encode is refused before its collective.
fn ensure_supported_dtype(dtype: DTypeDescriptor) -> core::result::Result<(), SyncError> {
    match dtype.builtin_id() {
        Some(DTypeId::F32 | DTypeId::F64 | DTypeId::F16 | DTypeId::BF16) => Ok(()),
        _ => Err(SyncError::UnsupportedDType { dtype }),
    }
}

/// Refuses a length that does not split into equal shards.
///
/// FSDP walks only; the caller has already established the world size is
/// nonzero (the driver's rank-inside-world check runs first), so the
/// remainder test cannot divide by zero.
fn ensure_divisible(elements: usize, world_size: usize) -> core::result::Result<(), SyncError> {
    if elements.is_multiple_of(world_size) {
        Ok(())
    } else {
        Err(SyncError::NonDivisibleShard {
            elements,
            world_size,
        })
    }
}

/// Synchronizes a backward pass's gradients through `reduce_scatter`.
///
/// The ZeRO-2 execution path: walks `model` in [`VisitParameters`] order
/// and, for each parameter that has a gradient in `grads`, reads it to a
/// host `f64` vector, calls
/// [`reduce_scatter_mean`](FsdpSynchronizer::reduce_scatter_mean), and
/// replaces the gradient with the full-length buffer that holds this
/// rank's slice of the mean at `[rank * chunk .. (rank + 1) * chunk]` and
/// zeros everywhere else. Each rank therefore retains exactly
/// `elements / world_size` gradient values per parameter - the `1/N`
/// bytes [`ShardedGradients`] reports - instead of the full mean an
/// all-reduce would leave behind.
///
/// In-place, like [`all_reduce_model_gradients`]: call it after
/// `backward()` and before the optimizer step. Parameters without a
/// gradient are skipped without a collective (module docs).
///
/// The optimizer on the resulting gradients must treat non-owned slices
/// as they are: zero. Zero gradients leave Adam/SGD momentum and second
/// moments at zero on those slices, so non-owned parameters do not move
/// here; [`all_gather_model_parameters`] rebuilds them from their owners
/// after the step. Global gradient clipping over these masked gradients
/// computes the wrong norm and is unsupported.
///
/// # Errors
///
/// [`SyncError::RankOutOfRange`] when `sync`'s rank is outside its world,
/// [`SyncError::UnsupportedDType`] for a gradient dtype outside
/// `f32`/`f64`/`f16`/`bf16`, [`SyncError::NonDivisibleShard`] before the
/// first parameter whose elements do not divide across the world,
/// [`SyncError::Synchronizer`] when the synchronizer refuses, and
/// [`SyncError::Backend`] when the backend cannot read, rebuild, or write
/// a gradient.
pub fn reduce_scatter_model_gradients<B, M>(
    model: &M,
    grads: &mut Gradients<B>,
    sync: &dyn FsdpSynchronizer,
) -> core::result::Result<ShardedGradients, SyncError>
where
    B: VariableBackend + AutogradBackend + HostInterop,
    M: VisitParameters<B>,
{
    let world_size = sync.world_size();
    let rank = sync.rank();
    if rank >= world_size {
        return Err(SyncError::RankOutOfRange { rank, world_size });
    }

    let mut visitor = ScatterVisitor {
        grads: grads.as_backend_mut(),
        sync,
        rank,
        world_size,
        shards: Vec::new(),
        full_elements: 0,
        owned_elements: 0,
        failure: None,
    };
    match model.visit_parameters(&StatePath::root(), &mut visitor) {
        Ok(()) => Ok(ShardedGradients {
            rank,
            world_size,
            shards: visitor.shards,
            full_elements: visitor.full_elements,
            owned_elements: visitor.owned_elements,
        }),
        Err(error) => Err(visitor.failure.unwrap_or_else(|| SyncError::Backend {
            message: error.to_string(),
        })),
    }
}

/// The reduce-scatter's per-parameter read-shard-write worker.
struct ScatterVisitor<'a, B: VariableBackend + AutogradBackend + HostInterop> {
    grads: &'a mut B::Grads,
    sync: &'a dyn FsdpSynchronizer,
    rank: usize,
    world_size: usize,
    shards: Vec<Vec<f64>>,
    full_elements: usize,
    owned_elements: usize,
    failure: Option<SyncError>,
}

impl<B> ScatterVisitor<'_, B>
where
    B: VariableBackend + AutogradBackend + HostInterop,
{
    fn fail(&mut self, error: SyncError) -> Error {
        let message = format!("{error}");
        self.failure = Some(error);
        Error::Msg(message)
    }
}

impl<B> ParameterVisitor<B> for ScatterVisitor<'_, B>
where
    B: VariableBackend + AutogradBackend + HostInterop,
{
    fn visit_param<S, K, Train>(
        &mut self,
        _path: &StatePath,
        param: &Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: Shape,
        K: DType,
        Train: TrainState,
    {
        let variable = param
            .variable_any()
            .downcast_ref::<B::Var<K>>()
            .ok_or_else(|| Error::InternalInvariant {
                operation: "reduce-scatter gradients",
                reason: "backend variable type did not match the parameter dtype",
            })?;
        let key = B::var_as_tensor::<K>(variable)?;

        let Some(grad) = B::get_grad::<K>(&key, &*self.grads)? else {
            return Ok(());
        };

        let dtype = param.dtype_descriptor();
        if let Err(error) = ensure_supported_dtype(dtype) {
            return Err(self.fail(error));
        }

        let values = B::float_to_vec1::<K>(&grad)?;
        let elements = values.len();
        if let Err(error) = ensure_divisible(elements, self.world_size) {
            return Err(self.fail(error));
        }
        let chunk = elements / self.world_size;
        let shard = match self.sync.reduce_scatter_mean(&values) {
            Ok(shard) => shard,
            Err(error) => return Err(self.fail(error)),
        };
        if shard.len() != chunk {
            return Err(self.fail(SyncError::Synchronizer {
                message: format!(
                    "reduce-scatter returned {} element(s) for a gradient of {elements} \
                     elements across {} ranks; expected {chunk}",
                    shard.len(),
                    self.world_size,
                ),
            }));
        }

        // Full-length buffer: this rank's slice at its owned range, zeros
        // everywhere else, so an optimizer that walks the whole tensor
        // leaves non-owned slices untouched.
        let mut masked = alloc::vec![0.0_f64; elements];
        masked[self.rank * chunk..(self.rank + 1) * chunk].copy_from_slice(&shard);

        let shape = B::host_shape(&grad);
        let Some(device) = B::host_storage_device(&grad) else {
            return Err(self.fail(SyncError::Backend {
                message: format!(
                    "backend did not report the device of a gradient with dtype {dtype:?}"
                ),
            }));
        };
        let bytes = match encode_reduced(dtype, &masked) {
            Ok(bytes) => bytes,
            Err(error) => return Err(self.fail(error)),
        };
        let reduced = B::from_bytes::<K>(&bytes, shape.as_ref(), dtype, &device)?;
        B::set_grad::<K>(&key, self.grads, reduced)?;

        self.owned_elements += chunk;
        self.full_elements += elements;
        self.shards.push(shard);
        Ok(())
    }
}

/// Masks gradients in place to this rank's contiguous shard (ZeRO-1).
///
/// The ZeRO-1 companion to [`reduce_scatter_model_gradients`]: run
/// [`all_reduce_model_gradients`] first (full mean on every rank), then
/// this walk zeros every gradient element outside
/// `[rank * chunk .. (rank + 1) * chunk]`. The resulting buffer is
/// byte-for-byte what a reduce-scatter of the same gradients would leave,
/// reached through the all-reduce transport ZeRO-1 specifies.
///
/// Same visitor, dtype, and divisibility rules as the reduce-scatter walk;
/// issues no collectives of its own, so a divisibility refusal here stops
/// the step before any optimizer sees a half-masked buffer.
///
/// # Errors
///
/// As [`reduce_scatter_model_gradients`], minus the synchronizer's own
/// refusal (this walk only reads `sync`'s
/// [`rank`](GradientSynchronizer::rank) and
/// [`world_size`](GradientSynchronizer::world_size)).
pub fn mask_gradients_to_owned_shard<B, M>(
    model: &M,
    grads: &mut Gradients<B>,
    sync: &dyn GradientSynchronizer,
) -> core::result::Result<(), SyncError>
where
    B: VariableBackend + AutogradBackend + HostInterop,
    M: VisitParameters<B>,
{
    let world_size = sync.world_size();
    let rank = sync.rank();
    if rank >= world_size {
        return Err(SyncError::RankOutOfRange { rank, world_size });
    }

    let mut visitor = MaskVisitor {
        grads: grads.as_backend_mut(),
        rank,
        world_size,
        failure: None,
    };
    match model.visit_parameters(&StatePath::root(), &mut visitor) {
        Ok(()) => Ok(()),
        Err(error) => Err(visitor.failure.unwrap_or_else(|| SyncError::Backend {
            message: error.to_string(),
        })),
    }
}

/// The mask's per-parameter read-mask-write worker.
struct MaskVisitor<'a, B: VariableBackend + AutogradBackend + HostInterop> {
    grads: &'a mut B::Grads,
    rank: usize,
    world_size: usize,
    failure: Option<SyncError>,
}

impl<B> MaskVisitor<'_, B>
where
    B: VariableBackend + AutogradBackend + HostInterop,
{
    fn fail(&mut self, error: SyncError) -> Error {
        let message = format!("{error}");
        self.failure = Some(error);
        Error::Msg(message)
    }
}

impl<B> ParameterVisitor<B> for MaskVisitor<'_, B>
where
    B: VariableBackend + AutogradBackend + HostInterop,
{
    fn visit_param<S, K, Train>(
        &mut self,
        _path: &StatePath,
        param: &Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: Shape,
        K: DType,
        Train: TrainState,
    {
        let variable = param
            .variable_any()
            .downcast_ref::<B::Var<K>>()
            .ok_or_else(|| Error::InternalInvariant {
                operation: "mask gradients to owned shard",
                reason: "backend variable type did not match the parameter dtype",
            })?;
        let key = B::var_as_tensor::<K>(variable)?;

        let Some(grad) = B::get_grad::<K>(&key, &*self.grads)? else {
            return Ok(());
        };

        let dtype = param.dtype_descriptor();
        if let Err(error) = ensure_supported_dtype(dtype) {
            return Err(self.fail(error));
        }

        let mut values = B::float_to_vec1::<K>(&grad)?;
        let elements = values.len();
        if let Err(error) = ensure_divisible(elements, self.world_size) {
            return Err(self.fail(error));
        }
        let chunk = elements / self.world_size;
        let start = self.rank * chunk;
        for (index, value) in values.iter_mut().enumerate() {
            if index < start || index >= start + chunk {
                *value = 0.0;
            }
        }

        let shape = B::host_shape(&grad);
        let Some(device) = B::host_storage_device(&grad) else {
            return Err(self.fail(SyncError::Backend {
                message: format!(
                    "backend did not report the device of a gradient with dtype {dtype:?}"
                ),
            }));
        };
        let bytes = match encode_reduced(dtype, &values) {
            Ok(bytes) => bytes,
            Err(error) => return Err(self.fail(error)),
        };
        let reduced = B::from_bytes::<K>(&bytes, shape.as_ref(), dtype, &device)?;
        B::set_grad::<K>(&key, self.grads, reduced)?;
        Ok(())
    }
}

/// Rebuilds every parameter from its owners' shards.
///
/// The second half of a ZeRO step: after the optimizer commits, walk `model`
/// in [`VisitParameters`] order, take this rank's owned slice of each
/// parameter, [`all_gather`](FsdpSynchronizer::all_gather) it into the full
/// value, and write the result back over the parameter. Every rank's
/// replica is then rebuilt from the owners, which is what makes the
/// masked-gradient step safe: rank-local optimizer effects on non-owned
/// slices (weight decay, for one) are discarded here instead of
/// compounding.
///
/// Walks every parameter, unconditionally. The reduce walks consult the
/// gradient map to skip untouched parameters, but this walk runs *after*
/// the step, and an optimizer commit (`assign_var`) stores a freshly
/// computed storage whose `TensorId` no longer matches the one `backward`
/// recorded - so post-step `get_grad` would answer `None` for every
/// parameter and silently skip the whole collective. The skip is
/// unobservable here by construction. It is also unnecessary: under
/// ZeRO-1/ZeRO-2 parameters keep full storage on every rank, so a
/// parameter no step touched carries identical values on all ranks and
/// gathering it writes back exactly what was already there; and every
/// rank runs this same walk over the same model, so rendezvous counts
/// stay aligned either way.
///
/// Writes through a cloned variable handle, relying on the shared-slot
/// identity every optimizer commit relies on: the model's parameter and
/// the handle observe the same storage.
///
/// Called after the optimizer step, before the next forward pass.
///
/// # Errors
///
/// [`SyncError::RankOutOfRange`] when `sync`'s rank is outside its world,
/// [`SyncError::UnsupportedDType`] for a parameter dtype outside
/// `f32`/`f64`/`f16`/`bf16`, [`SyncError::NonDivisibleShard`] before the
/// first parameter whose elements do not divide across the world,
/// [`SyncError::Synchronizer`] when the synchronizer refuses or returns a
/// length that is not `elements`, and [`SyncError::Backend`] when the
/// backend cannot read, rebuild, or write a parameter.
pub fn all_gather_model_parameters<B, M>(
    model: &M,
    sync: &dyn FsdpSynchronizer,
) -> core::result::Result<(), SyncError>
where
    B: VariableBackend + AutogradBackend + HostInterop,
    M: VisitParameters<B>,
{
    let world_size = sync.world_size();
    let rank = sync.rank();
    if rank >= world_size {
        return Err(SyncError::RankOutOfRange { rank, world_size });
    }

    let mut visitor = GatherVisitor {
        sync,
        rank,
        world_size,
        failure: None,
    };
    match model.visit_parameters(&StatePath::root(), &mut visitor) {
        Ok(()) => Ok(()),
        Err(error) => Err(visitor.failure.unwrap_or_else(|| SyncError::Backend {
            message: error.to_string(),
        })),
    }
}

/// The gather's per-parameter read-gather-write worker.
struct GatherVisitor<'a> {
    sync: &'a dyn FsdpSynchronizer,
    rank: usize,
    world_size: usize,
    failure: Option<SyncError>,
}

impl GatherVisitor<'_> {
    fn fail(&mut self, error: SyncError) -> Error {
        let message = format!("{error}");
        self.failure = Some(error);
        Error::Msg(message)
    }
}

impl<B> ParameterVisitor<B> for GatherVisitor<'_>
where
    B: VariableBackend + AutogradBackend + HostInterop,
{
    fn visit_param<S, K, Train>(
        &mut self,
        _path: &StatePath,
        param: &Param<S, B, K, Train>,
    ) -> Result<()>
    where
        S: Shape,
        K: DType,
        Train: TrainState,
    {
        let variable = param
            .variable_any()
            .downcast_ref::<B::Var<K>>()
            .ok_or_else(|| Error::InternalInvariant {
                operation: "all-gather parameters",
                reason: "backend variable type did not match the parameter dtype",
            })?;

        let dtype = param.dtype_descriptor();
        if let Err(error) = ensure_supported_dtype(dtype) {
            return Err(self.fail(error));
        }

        let storage = B::var_as_tensor::<K>(variable)?;
        let values = B::float_to_vec1::<K>(&storage)?;
        let elements = values.len();
        if let Err(error) = ensure_divisible(elements, self.world_size) {
            return Err(self.fail(error));
        }
        let chunk = elements / self.world_size;
        let start = self.rank * chunk;
        let shard = values[start..start + chunk].to_vec();
        let full = match self.sync.all_gather(&shard) {
            Ok(full) => full,
            Err(error) => return Err(self.fail(error)),
        };
        if full.len() != elements {
            return Err(self.fail(SyncError::Synchronizer {
                message: format!(
                    "all-gather returned {} element(s) for a parameter of {elements} \
                     elements across {} ranks; expected {elements}",
                    full.len(),
                    self.world_size,
                ),
            }));
        }

        let shape = B::host_shape(&storage);
        let Some(device) = B::host_storage_device(&storage) else {
            return Err(self.fail(SyncError::Backend {
                message: format!(
                    "backend did not report the device of a parameter with dtype {dtype:?}"
                ),
            }));
        };
        let bytes = match encode_reduced(dtype, &full) {
            Ok(bytes) => bytes,
            Err(error) => return Err(self.fail(error)),
        };
        let gathered = B::from_bytes::<K>(&bytes, shape.as_ref(), dtype, &device)?;
        // A cloned handle shares the parameter's slot, so this write is
        // visible through the model - the mechanism optimizer commits use.
        let mut handle = variable.clone();
        if let Err(error) = B::assign_var::<K>(&mut handle, &gathered) {
            return Err(self.fail(SyncError::Backend {
                message: error.to_string(),
            }));
        }
        Ok(())
    }
}

/// Encodes reduced `f64` values back into dtype-native bytes.
///
/// Byte order matches [`HostInterop::to_bytes`] on the CPU backend: native
/// endian for every supported dtype, `to_bits().to_ne_bytes()` for the
/// 16-bit float types.
fn encode_reduced(
    dtype: DTypeDescriptor,
    values: &[f64],
) -> core::result::Result<Vec<u8>, SyncError> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(8));
    match dtype.builtin_id() {
        Some(DTypeId::F32) => {
            for value in values {
                bytes.extend_from_slice(&(*value as f32).to_ne_bytes());
            }
        }
        Some(DTypeId::F64) => {
            for value in values {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
        Some(DTypeId::F16) => {
            for value in values {
                bytes.extend_from_slice(&half::f16::from_f64(*value).to_bits().to_ne_bytes());
            }
        }
        Some(DTypeId::BF16) => {
            for value in values {
                bytes.extend_from_slice(&half::bf16::from_f64(*value).to_bits().to_ne_bytes());
            }
        }
        _ => {
            return Err(SyncError::UnsupportedDType { dtype });
        }
    }
    Ok(bytes)
}
