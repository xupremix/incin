//! Custom operations with hand-written backward rules.
//!
//! [`DifferentiableOp`] is the typed half of the custom-training contract:
//! implement a forward kernel and a backward rule as pure functions over one
//! backend's storage, and the blanket [`Execute`] implementation builds the
//! tape node, derives its identities, checks admission, and records. The
//! author never names a [`TensorId`](crate::exec::TensorId), never orders an
//! id vector by hand, and cannot forget to record: there is no `execute`
//! body left to forget it in.
//!
//! What the trait does not cover is deliberate. Multi-output operations keep
//! the explicit per-backend `tape_record` path — one node per output cannot
//! be derived from a single return type without specialization acrobatics, and that shape is rare enough to deserve spelling out (see
//! the polar example). Composing existing differentiable tensor operations
//! needs no trait at all: the graph is inherited. And implementing both this
//! trait and a manual [`Execute`] for the same operation on the same backend
//! is a coherence error, which is the compiler enforcing that there is one
//! execution path, not two.
//!
//! One implementation trains one dtype: the associated [`Dtype`](Self::Dtype)
//! names the storage the recipe is written against. An operation that trains
//! in two dtypes is two implementations, conventionally via a generic
//! wrapper (`Square<f32>`, `Square<f64>`), each with its own recipe.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::Any;

use super::{Execute, StorageBackend, StorageOutput};
use crate::err::{BackendError, Result as CoreResult};
use crate::exec::UnsupportedReason;
use crate::exec::catalog::Operation;
use crate::exec::{Capabilities, CapabilityQuery, SupportLevel, TapeNode, TapeStorage};
use crate::tensor::dtype::DType;

/// A backend that accepts recorded custom backward recipes.
///
/// Each training backend owns its thread-local tape; this trait is the one
/// generic seam into it. The implementations are three-line delegations to
/// the backend's `record` function (`cpu::tape_record` and siblings).
pub trait RecordingBackend<K: DType>: StorageBackend {
    /// Record a custom backward recipe on this backend's tape, under the
    /// same `GradMode` gate as every built-in kernel.
    fn record_custom(node: TapeNode<Self::Storage<K>>);
}

/// A custom operation whose backward rule is written by hand.
///
/// Implementors provide a forward kernel and a vector-Jacobian product over
/// one backend's storage type. The blanket [`Execute`] implementation turns
/// the pair into a trained operation: it downcasts the validated input
/// handles, runs [`forward`](Self::forward), builds the [`TapeNode`] from
/// the storages themselves (so identities and order cannot drift), and
/// records through [`RecordingBackend`].
///
/// The trait is per backend, and each implementation names one dtype via
/// [`Dtype`](Self::Dtype): `impl DifferentiableOp<CpuBackend> for Square`
/// with `type Dtype = f32` trains `Square` on `f32` storage only. Broader
/// coverage is answered per query by [`supports`](Self::supports), which
/// defaults to `Native` — override it to refuse what the kernel was not
/// written for.
///
/// Recipes obey the same rules as hand-built nodes: one gradient per input
/// in input order (checked again by the walk), saved values owned by move,
/// shape-matched outputs. See the deep autograd chapter for the full
/// contract.
pub trait DifferentiableOp<B>: Operation
where
    B: StorageBackend,
{
    /// The dtype whose storage this recipe is written against.
    type Dtype: DType;
    /// What forward saves for backward. Owned values, never handles: the
    /// recipe closure moves them and must be self-contained.
    type Saved: Send + Sync + 'static;

    /// Capability answer for this operation on this backend and dtype.
    /// Defaults to `Native`; override to refuse dtypes (or ranks, layouts,
    /// training modes) the kernel does not hold. The blanket [`Execute`]
    /// forwards `supports_custom` here, so planners see the same answer
    /// dispatch enforces.
    fn supports(_query: &CapabilityQuery) -> SupportLevel {
        SupportLevel::Native
    }

    /// Run the forward kernel. Returns the output storage (which mints its
    /// own identity on construction) and the saved values the recipe needs.
    #[allow(clippy::type_complexity)]
    fn forward(
        inputs: &[B::Storage<Self::Dtype>],
        attributes: &Self::Attributes,
    ) -> core::result::Result<(B::Storage<Self::Dtype>, Self::Saved), BackendError>;

    /// Map one output gradient to one gradient per input, in input order.
    /// Receives the saved values by shared reference; clone out of them,
    /// never out of the live graph.
    fn backward(
        saved: &Self::Saved,
        grad_out: &B::Storage<Self::Dtype>,
    ) -> CoreResult<Vec<B::Storage<Self::Dtype>>>;
}

impl<O, B> Execute<O> for B
where
    O: DifferentiableOp<B>,
    B: RecordingBackend<O::Dtype> + Capabilities,
    B::Storage<O::Dtype>: Any + StorageOutput + TapeStorage,
{
    type Output = B::Storage<O::Dtype>;

    fn supports_custom(&self, query: &CapabilityQuery) -> SupportLevel {
        O::supports(query)
    }

    fn execute(
        &self,
        request: super::ExecutionRequest<'_, O, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let mut owned = Vec::with_capacity(request.inputs.len());
        for handle in request.inputs {
            match handle.downcast_ref::<B::Storage<O::Dtype>>() {
                Some(storage) => owned.push(storage.clone()),
                None => {
                    return Err(BackendError::unsupported(
                        B::BACKEND_NAME,
                        UnsupportedReason::CustomOperation { operation: O::KEY },
                    ));
                }
            }
        }
        let attributes = request.operation.descriptor().attributes();
        let (out, saved) = O::forward(&owned, attributes)?;
        let mut input_ids = Vec::with_capacity(owned.len());
        for storage in &owned {
            input_ids.push(storage.id());
        }
        let node = TapeNode {
            output_id: out.id(),
            input_ids,
            backward: Box::new(move |grad_out: &B::Storage<O::Dtype>| {
                O::backward(&saved, grad_out)
            }),
        };
        B::record_custom(node);
        Ok(out)
    }
}
