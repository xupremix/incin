//! Dispatch-time autocast: allowlisted operand casting for mixed precision.
//!
//! Issue #2's remaining slice. [`RuntimePrecisionPolicy`] reaches every eager
//! operation through [`ExecutionContext::from_scope`], but carrying the axis
//! is not enforcing it: an `active_dtype` of `bf16` must actually put `bf16`
//! operands in front of the kernels whose capability rows admit them. This
//! module is where that happens - at dispatch admission, before the
//! descriptor is inferred, so the descriptor is built from the dtypes the
//! backend will really see.
//!
//! The rules, each one a deliberate refusal to guess:
//!
//! - **Allowlist by [`SemanticProfile`], not by a hand-kept op list.**
//!   [`SemanticProfile::BinaryBroadcast`], [`SemanticProfile::MatMul`] and
//!   [`SemanticProfile::Attention`] compute *toward* the active dtype
//!   (downcast); [`SemanticProfile::Reduction`] and [`SemanticProfile::Loss`]
//!   widen their active-dtype inputs to the policy's exact accumulator so
//!   sums and losses never accumulate in the narrow dtype. Every other
//!   profile - creation, shape, transfer, optimizer steps, unary floats and
//!   scalar arithmetic among them - is never cast: master weights stay in
//!   their storage dtype, and a scalar loss chain such as
//!   `sum -> mul_scalar` stays readable as `f32`.
//! - **The capability registry gates every cast.** Before any operand is
//!   rewritten, each input that would be cast is re-queried at the *target*
//!   dtype against the operation's own identity. A backend that admits the
//!   operation only for `f32` (CPU `scaled_dot_product_attention`, for
//!   instance) simply does not get the cast - the invocation proceeds
//!   uncast, exactly as before, rather than failing or silently lying about
//!   support.
//! - **An `fp32` policy casts nothing.** [`RuntimePrecisionPolicy::fp32`]
//!   carries no `active_dtype`, so the plan short-circuits before any work:
//!   zero allocation, zero queries, bit-identical execution.
//! - **No caster, no cast.** Dispatch itself stays generic over any
//!   `Execute<O> + Capabilities` backend; the B-typed machinery is installed
//!   on a thread by [`install`](crate::exec::autocast::install) (the
//!   trainer's `fit` and `fit_scaled` do it) and removed when the returned
//!   guard drops. Outside a trainer run - or on a thread that never entered
//!   one - this module is a no-op.
//!
//! The cast itself is an ordinary [`op::ToDType`] dispatch, so it inherits
//! descriptor validation, capability admission, tape recording (a cast
//! between two tracked storages becomes a backward node that casts the
//! gradient back), and whatever execution site the backend already gives
//! `to_dtype`.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::Any;

#[cfg(feature = "std")]
use crate::dist::placement::Local;
use crate::err::BackendError;
use crate::exec::capability::Capabilities;
#[cfg(feature = "std")]
use crate::exec::capability::{CapabilityQuery, OperationIdentity, SupportLevel};
#[cfg(feature = "std")]
use crate::exec::catalog::{DTypeAttributes, catalog_entry};
#[cfg(any(feature = "std", test))]
use crate::exec::catalog::{OPERATION_CATALOG, SemanticProfile};
use crate::exec::catalog::{Operation, op};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch::CanonicalError;
#[cfg(any(feature = "std", test))]
use crate::exec::precision::PrecisionChoice;
use crate::exec::precision::RuntimePrecisionPolicy;
use crate::exec::request::TensorHandle;
use crate::shapes::error::OperationKind;
use crate::tensor::backend::{Execute, ExecuteInto, StorageBackend};
#[cfg(any(feature = "std", test))]
use crate::tensor::dtype::DTypeId;
use crate::tensor::dtype::{DTypeDescriptor, bf16, f16};

/// Which allowlisted cast, if any, applies to one operation under one policy.
#[cfg(any(feature = "std", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CastRole {
    /// Push operands toward the active compute dtype.
    Compute {
        /// Dtype every casted operand becomes.
        target: DTypeDescriptor,
    },
    /// Widen operands recorded in the active dtype up to the accumulator.
    Accumulate {
        /// Dtype every casted operand becomes.
        target: DTypeDescriptor,
        /// Active dtype; only inputs in this dtype are cast.
        from: DTypeDescriptor,
    },
}

#[cfg(any(feature = "std", test))]
impl CastRole {
    const fn target(self) -> DTypeDescriptor {
        match self {
            Self::Compute { target } | Self::Accumulate { target, .. } => target,
        }
    }
}

/// The allowlist decision: profile plus policy, nothing else.
///
/// Returns `None` for an `fp32` policy (no active dtype), for a profile
/// outside the allowlist, and for an accumulator that is not an exact dtype
/// distinct from the active one - in each of this module's terms there is
/// nothing honest to cast.
#[cfg(any(feature = "std", test))]
fn cast_role(profile: SemanticProfile, policy: RuntimePrecisionPolicy) -> Option<CastRole> {
    let active = policy.active_dtype()?;
    match profile {
        SemanticProfile::BinaryBroadcast | SemanticProfile::MatMul | SemanticProfile::Attention => {
            Some(CastRole::Compute { target: active })
        }
        SemanticProfile::Reduction | SemanticProfile::Loss => match policy.accumulator() {
            PrecisionChoice::Exact(target) if target != active => Some(CastRole::Accumulate {
                target,
                from: active,
            }),
            _ => None,
        },
        _ => None,
    }
}

/// Map a typed operation marker back to its catalog identity.
///
/// Custom operations (any `KEY` outside the `incin` namespace) resolve to
/// `None`: the allowlist is defined over the canonical catalog, and a custom
/// op that happens to share a name with one must not inherit its cast.
#[cfg(any(feature = "std", test))]
fn operation_kind<O: Operation>() -> Option<OperationKind> {
    if O::KEY.namespace != "incin" {
        return None;
    }
    let key = O::KEY;
    let name = key.name.as_ref();
    OPERATION_CATALOG
        .iter()
        .find(|row| row.name == name)
        .map(|row| row.operation)
}

/// Owned cast results paired with the dtype they were cast to.
///
/// Dispatch rebuilds a complete input slice from this before inference;
/// the boxes keep the freshly dispatched `ToDType` storages alive for as
/// long as the handles borrowing them are in flight.
#[cfg_attr(not(feature = "std"), allow(dead_code))]
pub(crate) struct CastOutcome {
    /// Dtype every casted operand was cast to.
    target: DTypeDescriptor,
    /// Per-input cast storage; `None` where the input was left untouched.
    owned: Vec<Option<Box<dyn Any>>>,
}

fn invalid(reason: &'static str) -> CanonicalError {
    CanonicalError::Backend(BackendError::InvalidInput {
        operation: OperationKind::ToDType,
        reason,
    })
}

/// True when the descriptor is one of the four host floats this module can
/// both name as a type parameter and hand back through `from_storage`.
#[cfg(any(feature = "std", test))]
const fn is_host_float(dtype: DTypeDescriptor) -> bool {
    matches!(
        dtype.builtin_id(),
        Some(DTypeId::F16 | DTypeId::BF16 | DTypeId::F32 | DTypeId::F64)
    )
}

/// Plan and, if the allowlist and the capability gate both agree, perform
/// the cast for one invocation's inputs.
///
/// `Ok(None)` is the ordinary answer: `fp32` policy, no caster installed,
/// profile outside the allowlist, every input already at the target, a
/// non-float or non-host-float operand, or the backend's capability row
/// refusing the target dtype. Only `Ok(Some(..))` carries rewritten
/// operands, and every input of such an invocation is a host float the
/// rebuild step can name.
pub(crate) fn cast_operands<O, B>(
    context: &ExecutionContext<B>,
    inputs: &[TensorHandle<'_>],
) -> Result<Option<CastOutcome>, CanonicalError>
where
    O: Operation,
    B: Execute<O> + Capabilities + 'static,
{
    let policy = context.precision_policy();
    if policy.active_dtype().is_none() || inputs.is_empty() {
        return Ok(None);
    }
    cast_allowlisted::<O, B>(context, inputs, policy)
}

/// The allowlist, capability gate, and cast itself. `std` performs the cast
/// through the installed registry; `no_std` has no registry and answers that
/// there is never one.
#[cfg(feature = "std")]
fn cast_allowlisted<O, B>(
    context: &ExecutionContext<B>,
    inputs: &[TensorHandle<'_>],
    policy: RuntimePrecisionPolicy,
) -> Result<Option<CastOutcome>, CanonicalError>
where
    O: Operation,
    B: Execute<O> + Capabilities + 'static,
{
    let Some(fns) = installed::current() else {
        return Ok(None);
    };
    let Some(operation) = operation_kind::<O>() else {
        return Ok(None);
    };
    let Some(entry) = catalog_entry(operation) else {
        return Ok(None);
    };
    let Some(role) = cast_role(entry.profile, policy) else {
        return Ok(None);
    };

    let target = role.target();
    let mut mask = Vec::with_capacity(inputs.len());
    let mut any = false;
    for handle in inputs {
        let dtype = handle.metadata().dtype;
        // One operand the rebuild step cannot name - integer, bool,
        // quantized, or a custom float without a builtin identity - and
        // the whole invocation stays uncast: the allowlist is a property
        // of the operation, not of whichever input the caller happens to
        // list first.
        if !is_host_float(dtype) {
            return Ok(None);
        }
        let casted = match role {
            CastRole::Compute { target } => dtype != target,
            CastRole::Accumulate { target: _, from } => dtype == from,
        };
        any |= casted;
        mask.push(casted);
    }
    if !any {
        return Ok(None);
    }

    // Capability gate: would this backend admit *this* operation for
    // every operand at the target dtype? All-or-nothing across the
    // slice, mirroring `dispatch::admit` exactly - same registry, same
    // training and math-mode axes, only the dtype substituted.
    let identity = OperationIdentity::Builtin(operation);
    for (handle, &casted) in inputs.iter().zip(&mask) {
        if !casted {
            continue;
        }
        let metadata = handle.metadata();
        let query = CapabilityQuery {
            operation: identity.clone(),
            dtype: target,
            layout: metadata.layout,
            rank: metadata.shape.dims().len(),
            training: context.training(),
            math_mode: context.math_mode(),
        };
        if matches!(
            context.backend().support(&query),
            SupportLevel::Unsupported(_)
        ) {
            return Ok(None);
        }
    }

    let owned = (fns.cast)(CastArgs {
        context,
        inputs,
        mask: &mask,
        target,
    })?;
    Ok(Some(CastOutcome { target, owned }))
}

/// The `std`-less twin: no registry exists, so no cast ever does.
#[cfg(not(feature = "std"))]
fn cast_allowlisted<O, B>(
    _context: &ExecutionContext<B>,
    _inputs: &[TensorHandle<'_>],
    _policy: RuntimePrecisionPolicy,
) -> Result<Option<CastOutcome>, CanonicalError>
where
    O: Operation,
    B: Execute<O> + Capabilities + 'static,
{
    Ok(None)
}

/// Rebuild a full input slice from [`cast_operands`]' output.
///
/// Untouched inputs are re-wrapped through their own storage (a handle is
/// not `Clone`, and `from_storage` re-derives metadata and the tracing
/// value exactly as the original construction did); casted inputs borrow
/// the boxes in `outcome`. The type-erased rebind travels through the same
/// thread-local registry as the cast itself, so this adds no bounds to
/// dispatch beyond what [`cast_operands`] already takes.
#[cfg(feature = "std")]
pub(crate) fn rebind<'a>(
    outcome: &'a CastOutcome,
    originals: &'a [TensorHandle<'a>],
) -> Result<Vec<TensorHandle<'a>>, CanonicalError> {
    let fns = installed::current().ok_or_else(no_caster)?;
    (fns.rebind)(outcome, originals)
}

/// The `std`-less twin of the above. It is unreachable - [`cast_operands`]
/// finds no caster and returns `Ok(None)` first - but it keeps both dispatch
/// funnels compiling without a `cfg` of their own.
#[cfg(not(feature = "std"))]
pub(crate) fn rebind<'a>(
    outcome: &'a CastOutcome,
    originals: &[TensorHandle<'a>],
) -> Result<Vec<TensorHandle<'a>>, CanonicalError> {
    let _ = (outcome, originals);
    Err(no_caster())
}

fn no_caster() -> CanonicalError {
    invalid("autocast caster is not installed on this thread")
}

/// The B-typed cast, type-erased to a plain function pointer.
///
/// `CastArgs::context` is an `&dyn Any` carrying the caller's
/// `ExecutionContext<B>`; `cast_impl` recovers it. Coercing to `&dyn Any`
/// is what forces `B: 'static` on the dispatch funnels - a bound every
/// concrete backend already satisfies through [`crate::backend_authoring::Backend`]'s
/// own `'static` supertrait, and one that demands no new implementation
/// from any backend.
#[cfg(feature = "std")]
type ErasedCaster = for<'a> fn(CastArgs<'a>) -> Result<Vec<Option<Box<dyn Any>>>, CanonicalError>;

/// The B-typed rebuild, type-erased the same way.
///
/// Late-bound over the handle lifetime so the registry can store one
/// function pointer while outcomes and inputs keep their own borrow
/// scopes.
#[cfg(feature = "std")]
type ErasedRebind = for<'a> fn(
    &'a CastOutcome,
    &'a [TensorHandle<'a>],
) -> Result<Vec<TensorHandle<'a>>, CanonicalError>;

/// Everything `cast_impl` needs, with the backend type already erased.
#[cfg(feature = "std")]
struct CastArgs<'a> {
    /// The caller's `ExecutionContext<B>`, erased.
    context: &'a dyn Any,
    /// The invocation's original inputs.
    inputs: &'a [TensorHandle<'a>],
    /// Which inputs the plan said to cast.
    mask: &'a [bool],
    /// Dtype every casted input becomes.
    target: DTypeDescriptor,
}

/// Backends that can host the autocast: `ToDType` executable into storage
/// for every dtype a policy can name, with storages the registry can hand
/// back as `Any`.
///
/// [`install`] is the only consumer. It exists as one bound so the
/// trainer's `fit` signatures carry a single name instead of eight
/// associated-type obligations, and because the blanket impl means every
/// backend that already implements `Execute<op::ToDType>` with a storage
/// output - CPU and CUDA today - gets it for free (issue #2).
pub trait AutocastBackend:
    ExecuteInto<op::ToDType, f16>
    + ExecuteInto<op::ToDType, bf16>
    + ExecuteInto<op::ToDType, f32>
    + ExecuteInto<op::ToDType, f64>
    + 'static
where
    Self: StorageBackend,
    <Self as StorageBackend>::Storage<f16>: Any,
    <Self as StorageBackend>::Storage<bf16>: Any,
    <Self as StorageBackend>::Storage<f32>: Any,
    <Self as StorageBackend>::Storage<f64>: Any,
{
}

impl<B> AutocastBackend for B
where
    B: ExecuteInto<op::ToDType, f16>
        + ExecuteInto<op::ToDType, bf16>
        + ExecuteInto<op::ToDType, f32>
        + ExecuteInto<op::ToDType, f64>
        + 'static,
    <B as StorageBackend>::Storage<f16>: Any,
    <B as StorageBackend>::Storage<bf16>: Any,
    <B as StorageBackend>::Storage<f32>: Any,
    <B as StorageBackend>::Storage<f64>: Any,
{
}

#[cfg(feature = "std")]
mod installed {
    use core::cell::RefCell;

    use super::{ErasedCaster, ErasedRebind};

    std::thread_local! {
        /// The B-typed cast/rebind pair installed on this thread, if any.
        /// Thread-local because the ambient execution policy is one: a scope
        /// on thread A must not recast operands on thread B.
        static FNS: RefCell<Option<Fns>> = const { RefCell::new(None) };
    }

    #[derive(Clone, Copy)]
    pub(super) struct Fns {
        pub cast: ErasedCaster,
        pub rebind: ErasedRebind,
    }

    pub(super) fn current() -> Option<Fns> {
        FNS.with(|cell| *cell.borrow())
    }

    pub(super) fn set(fns: Fns) -> Option<Fns> {
        FNS.with(|cell| cell.borrow_mut().replace(fns))
    }

    pub(super) fn restore(previous: Option<Fns>) {
        FNS.with(|cell| *cell.borrow_mut() = previous);
    }
}

/// Install `B`'s cast/rebind pair as this thread's autocast.
///
/// The returned guard restores whatever was installed before (including
/// nothing) when it drops - including on unwind - so nested `fit` calls
/// cannot leave a stale caster behind. Holding the guard for the duration
/// of a training loop is what makes dispatch's allowlist find a caster;
/// every other thread, and this thread outside the guard's lifetime, sees
/// no caster and casts nothing.
///
/// # Panics
///
/// Does not panic. A second [`install`] on the same thread simply stacks:
/// the inner guard restores the outer one on drop.
#[cfg(feature = "std")]
#[must_use = "dropping the guard immediately uninstalls the caster"]
pub fn install<B>() -> InstalledAutocast
where
    B: AutocastBackend,
{
    let previous = installed::set(installed::Fns {
        cast: cast_impl::<B>,
        rebind: rebind_impl::<B>,
    });
    InstalledAutocast { previous }
}

/// RAII guard returned by [`install`]; drops restore the previous caster.
#[cfg(feature = "std")]
#[must_use = "dropping this guard immediately uninstalls the autocast"]
pub struct InstalledAutocast {
    previous: Option<installed::Fns>,
}

#[cfg(feature = "std")]
impl Drop for InstalledAutocast {
    fn drop(&mut self) {
        installed::restore(self.previous.take());
    }
}

/// Dispatch one `ToDType` per casted input and box the resulting storages.
#[cfg(feature = "std")]
fn cast_impl<B>(args: CastArgs<'_>) -> Result<Vec<Option<Box<dyn Any>>>, CanonicalError>
where
    B: AutocastBackend,
{
    let context = args
        .context
        .downcast_ref::<ExecutionContext<B>>()
        .ok_or_else(|| invalid("autocast context backend type mismatch"))?;

    let mut owned = Vec::with_capacity(args.inputs.len());
    for (index, &casted) in args.mask.iter().enumerate() {
        if !casted {
            owned.push(None);
            continue;
        }
        let single = &args.inputs[index..index + 1];
        let attributes = DTypeAttributes { dtype: args.target };
        let storage: Box<dyn Any> = match args.target.builtin_id() {
            Some(DTypeId::F16) => Box::new(<B as ExecuteInto<op::ToDType, f16>>::dispatch_into(
                context, attributes, single,
            )?),
            Some(DTypeId::BF16) => Box::new(<B as ExecuteInto<op::ToDType, bf16>>::dispatch_into(
                context, attributes, single,
            )?),
            Some(DTypeId::F32) => Box::new(<B as ExecuteInto<op::ToDType, f32>>::dispatch_into(
                context, attributes, single,
            )?),
            Some(DTypeId::F64) => Box::new(<B as ExecuteInto<op::ToDType, f64>>::dispatch_into(
                context, attributes, single,
            )?),
            _ => {
                return Err(invalid(
                    "autocast target dtype is not a host float this caster can name",
                ));
            }
        };
        owned.push(Some(storage));
    }
    Ok(owned)
}

/// Rebuild every handle: casted ones from the boxes, untouched ones from
/// their original storage.
#[cfg(feature = "std")]
fn rebind_impl<'a, B>(
    outcome: &'a CastOutcome,
    originals: &'a [TensorHandle<'a>],
) -> Result<Vec<TensorHandle<'a>>, CanonicalError>
where
    B: AutocastBackend,
{
    let mut rebuilt = Vec::with_capacity(originals.len());
    for (original, casted) in originals.iter().zip(&outcome.owned) {
        let handle = match casted {
            Some(storage) => casted_handle::<B>(storage.as_ref(), outcome.target)?,
            None => original_handle::<B>(original)?,
        };
        rebuilt.push(handle);
    }
    Ok(rebuilt)
}

/// Re-wrap a storage this module just produced through `ToDType`.
#[cfg(feature = "std")]
fn casted_handle<'a, B>(
    storage: &'a dyn Any,
    target: DTypeDescriptor,
) -> Result<TensorHandle<'a>, CanonicalError>
where
    B: AutocastBackend,
{
    match target.builtin_id() {
        Some(DTypeId::F16) => {
            let s = storage
                .downcast_ref::<B::Storage<f16>>()
                .ok_or_else(|| invalid("autocast cast storage type mismatch"))?;
            Ok(TensorHandle::from_storage::<B, f16, Local>(s))
        }
        Some(DTypeId::BF16) => {
            let s = storage
                .downcast_ref::<B::Storage<bf16>>()
                .ok_or_else(|| invalid("autocast cast storage type mismatch"))?;
            Ok(TensorHandle::from_storage::<B, bf16, Local>(s))
        }
        Some(DTypeId::F32) => {
            let s = storage
                .downcast_ref::<B::Storage<f32>>()
                .ok_or_else(|| invalid("autocast cast storage type mismatch"))?;
            Ok(TensorHandle::from_storage::<B, f32, Local>(s))
        }
        Some(DTypeId::F64) => {
            let s = storage
                .downcast_ref::<B::Storage<f64>>()
                .ok_or_else(|| invalid("autocast cast storage type mismatch"))?;
            Ok(TensorHandle::from_storage::<B, f64, Local>(s))
        }
        _ => Err(invalid(
            "autocast target dtype is not a host float this caster can name",
        )),
    }
}

/// Re-wrap an input this module left untouched, through its own storage.
#[cfg(feature = "std")]
fn original_handle<'a, B>(
    original: &'a TensorHandle<'a>,
) -> Result<TensorHandle<'a>, CanonicalError>
where
    B: AutocastBackend,
{
    let dtype = original.metadata().dtype;
    match dtype.builtin_id() {
        Some(DTypeId::F16) => {
            let s = original
                .downcast_ref::<B::Storage<f16>>()
                .ok_or_else(|| invalid("autocast original storage type mismatch"))?;
            Ok(TensorHandle::from_storage::<B, f16, Local>(s))
        }
        Some(DTypeId::BF16) => {
            let s = original
                .downcast_ref::<B::Storage<bf16>>()
                .ok_or_else(|| invalid("autocast original storage type mismatch"))?;
            Ok(TensorHandle::from_storage::<B, bf16, Local>(s))
        }
        Some(DTypeId::F32) => {
            let s = original
                .downcast_ref::<B::Storage<f32>>()
                .ok_or_else(|| invalid("autocast original storage type mismatch"))?;
            Ok(TensorHandle::from_storage::<B, f32, Local>(s))
        }
        Some(DTypeId::F64) => {
            let s = original
                .downcast_ref::<B::Storage<f64>>()
                .ok_or_else(|| invalid("autocast original storage type mismatch"))?;
            Ok(TensorHandle::from_storage::<B, f64, Local>(s))
        }
        _ => Err(invalid(
            "autocast operand dtype is not a host float this caster can name",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dtype::ConstDType;

    const BF16: DTypeDescriptor = <bf16 as ConstDType>::DESCRIPTOR;
    const F16: DTypeDescriptor = <f16 as ConstDType>::DESCRIPTOR;
    const F32: DTypeDescriptor = <f32 as ConstDType>::DESCRIPTOR;

    #[test]
    fn compute_profiles_cast_toward_the_active_dtype() {
        for profile in [
            SemanticProfile::BinaryBroadcast,
            SemanticProfile::MatMul,
            SemanticProfile::Attention,
        ] {
            assert_eq!(
                cast_role(profile, RuntimePrecisionPolicy::mixed_bf16()),
                Some(CastRole::Compute { target: BF16 })
            );
            assert_eq!(
                cast_role(profile, RuntimePrecisionPolicy::mixed_f16()),
                Some(CastRole::Compute { target: F16 })
            );
        }
    }

    #[test]
    fn reduction_and_loss_widen_active_inputs_to_the_accumulator() {
        for profile in [SemanticProfile::Reduction, SemanticProfile::Loss] {
            assert_eq!(
                cast_role(profile, RuntimePrecisionPolicy::mixed_bf16()),
                Some(CastRole::Accumulate {
                    target: F32,
                    from: BF16
                })
            );
            assert_eq!(
                cast_role(profile, RuntimePrecisionPolicy::mixed_f16()),
                Some(CastRole::Accumulate {
                    target: F32,
                    from: F16
                })
            );
        }
    }

    #[test]
    fn profiles_outside_the_allowlist_never_cast() {
        for profile in [
            SemanticProfile::UnaryFloat,
            SemanticProfile::Shape,
            SemanticProfile::Transfer,
            SemanticProfile::Creation,
            SemanticProfile::Optimizer,
            SemanticProfile::Module,
        ] {
            assert_eq!(
                cast_role(profile, RuntimePrecisionPolicy::mixed_bf16()),
                None
            );
            assert_eq!(
                cast_role(profile, RuntimePrecisionPolicy::mixed_f16()),
                None
            );
        }
    }

    #[test]
    fn fp32_and_degenerate_accumulators_cast_nothing() {
        for profile in [
            SemanticProfile::BinaryBroadcast,
            SemanticProfile::MatMul,
            SemanticProfile::Attention,
            SemanticProfile::Reduction,
            SemanticProfile::Loss,
        ] {
            assert_eq!(cast_role(profile, RuntimePrecisionPolicy::fp32()), None);
        }
        // exact::<f32> has active == accumulator, so there is nothing to
        // widen; mixed_* with a Native accumulator likewise declines.
        assert_eq!(
            cast_role(
                SemanticProfile::Reduction,
                RuntimePrecisionPolicy::exact::<f32>()
            ),
            None
        );
        assert_eq!(
            cast_role(
                SemanticProfile::Reduction,
                RuntimePrecisionPolicy::mixed_bf16().with_accumulator(PrecisionChoice::Native)
            ),
            None
        );
    }

    #[test]
    fn operation_kind_maps_catalog_markers_and_rejects_custom_namespaces() {
        assert_eq!(operation_kind::<op::Add>(), Some(OperationKind::Add));
        assert_eq!(
            operation_kind::<op::MatMulExact>(),
            Some(OperationKind::MatMulExact)
        );
        assert_eq!(operation_kind::<op::SumAll>(), Some(OperationKind::SumAll));
        assert_eq!(
            operation_kind::<op::MseLoss>(),
            Some(OperationKind::MseLoss)
        );
        assert_eq!(
            operation_kind::<op::ToDType>(),
            Some(OperationKind::ToDType)
        );
        assert_eq!(
            operation_kind::<op::MulScalar>(),
            Some(OperationKind::MulScalar)
        );
    }

    #[test]
    fn host_float_predicate_matches_exactly_the_four_builtin_floats() {
        assert!(is_host_float(F16));
        assert!(is_host_float(BF16));
        assert!(is_host_float(F32));
        assert!(is_host_float(<f64 as ConstDType>::DESCRIPTOR));
        assert!(!is_host_float(<u32 as ConstDType>::DESCRIPTOR));
        assert!(!is_host_float(DTypeId::Bool.descriptor()));
        assert!(!is_host_float(DTypeId::Q8_0.descriptor()));
    }
}
