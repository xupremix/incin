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
//! the explicit per-backend `tape_record` path, because one node per output
//! cannot be derived from a single return type without specialization
//! acrobatics, and that shape is rare enough to deserve spelling out (see the
//! polar example). And implementing both this trait and a manual [`Execute`]
//! for the same operation on the same backend is a coherence error, which is
//! the compiler enforcing that there is one execution path, not two.
//!
//! # Composing built-in operations
//!
//! Chaining differentiable tensor methods needs no trait at all: each built-in
//! records its own node and the graph is inherited. That is composition at the
//! *tensor* level, and it produces no custom node.
//!
//! Building a kernel *inside* [`forward`](DifferentiableOp::forward) out of
//! dispatched built-in operations is a different thing, and it is supported:
//! the blanket [`Execute`] runs `forward` under
//! [`GradMode::Disabled`](crate::exec::GradMode), so the built-ins record
//! nothing and the custom node is the single authority on this operation's
//! derivative. Without that scope both the inner nodes and the custom node
//! would carry the same output identity, the reverse walk would invoke every
//! recipe against the same output gradient, and each input would receive its
//! gradient twice. `tests/custom_op_composition.rs` pins the number.
//!
//! # One implementation, or several
//!
//! The associated [`Dtype`](Self::Dtype) names the storage one recipe is
//! written against, so a recipe written as a hand loop over one buffer variant
//! covers one dtype on one backend.
//!
//! A recipe written as dispatched built-in operations covers every backend
//! that implements them and every dtype they accept, from one implementation.
//! The dtype has to appear in the self type rather than only in the associated
//! type, because an impl type parameter that appears nowhere in the self type
//! is rejected (E0207) and `type Dtype = K` does not constrain `K`:
//!
//! ```text
//! struct ScaledSquare<K>(PhantomData<K>);
//!
//! impl<B, K> DifferentiableOp<B> for ScaledSquare<K>
//! where
//!     B: Backend + Execute<op::Mul> + RecordingBackend<K>,
//!     K: FloatDType,
//!     // ...
//! { type Dtype = K; /* ... */ }
//! ```
//!
//! Which to write is a performance question rather than a constraint of the
//! design: a fused kernel is still one implementation per backend, and a
//! composed one is not.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::Any;

use super::{Execute, StorageBackend, StorageOutput};
use crate::err::{BackendError, Result as CoreResult};
use crate::exec::UnsupportedReason;
use crate::exec::catalog::Operation;
use crate::exec::{Capabilities, CapabilityQuery, GradMode, SupportLevel, TapeNode, TapeStorage};
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
        // The forward kernel runs with recording off. A recipe author is
        // entitled to build their kernel out of built-in operations rather
        // than a hand-written loop, and those record nodes of their own
        // against the very output this impl is about to record a custom node
        // for. Both nodes would then carry the same `output_id`, the reverse
        // walk would invoke both recipes against the same output gradient,
        // and every input would receive its gradient twice. Disabling here is
        // what makes the custom node the single authority on this
        // operation's derivative, which is what declaring one means.
        //
        // `restrict` rather than `scope` for the reason recorded in D12:
        // `scope` installs a thread-local and is `std`-only, and this module
        // is not gated.
        let (out, saved) = GradMode::Disabled.restrict(|| O::forward(&owned, attributes))?;
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
