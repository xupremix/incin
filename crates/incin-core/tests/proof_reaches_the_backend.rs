//! The frontend's shape evidence reaches the backend, not just the descriptor.
//!
//! `exec::catalog`'s own unit test proves `ValidatedInvocation::infer`
//! preserves the proof level it is handed. This proves the other half: that
//! `dispatch::execute_shaped` derives the evidence from the caller's shape type
//! and that the `Validated` carrying it is the value the backend's
//! `execute_shaped` receives — alongside that same type.
//!
//! Without this, "the typed frontend's proof survives lowering" would be an
//! inference across two separately-tested links rather than an observation.
//!
//! The backend here records the proof level and allocates nothing. That is the
//! point — it is the smallest thing that can implement `Execute` and answer a
//! capability query, so the only behaviour under test is what the dispatch
//! layer passes down.

extern crate incin_core as incin;

use std::cell::Cell;

use incin_core::backend_authoring::operations::CreationAttributes;
use incin_core::backend_authoring::{Descriptor, Execute, ExecutionRequest, StorageBackend, op};
use incin_core::exec::{
    Capabilities, CapabilityQuery, ExecutionContext, GradMode, ProofLevel, SupportLevel, dispatch,
};
use incin_core::prelude::{Cpu, DTypeId, DeviceId, Dyn, Shape, ShapeBuf};

thread_local! {
    /// The proof level the backend last saw. A thread-local rather than a
    /// field because `execute` takes `&self` through a context the dispatch
    /// layer owns, so there is no `&mut` to write through.
    static OBSERVED: Cell<Option<ProofLevel>> = const { Cell::new(None) };
    static OBSERVED_STATIC_NUMEL: Cell<Option<Option<usize>>> = const { Cell::new(None) };
}

/// Records the proof level it is dispatched with and produces no storage.
#[derive(Debug, Clone, Default)]
struct RecordingBackend;

impl StorageBackend for RecordingBackend {
    const BACKEND_NAME: &'static str = "Recording";
    /// No allocation happens, so storage is a unit.
    type Storage<K: incin_core::prelude::DType> = ();
    type Device = Cpu;

    fn metadata<K: incin_core::prelude::DType>(
        _storage: &Self::Storage<K>,
    ) -> &incin_core::exec::TensorMeta {
        unreachable!("this backend is never asked for metadata: it has no operands and no outputs")
    }
}

impl Capabilities for RecordingBackend {
    /// Everything is supported. Refusal is `canonical_cpu.rs`'s subject; the
    /// only thing being measured here is what proof arrives.
    fn support(&self, _query: &CapabilityQuery) -> SupportLevel {
        SupportLevel::Native
    }
}

impl Execute<Descriptor<op::Zeros>> for RecordingBackend {
    type Output = ();

    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, Descriptor<op::Zeros>, Self>,
    ) -> Result<Self::Output, incin_core::prelude::BackendError> {
        OBSERVED.with(|slot| slot.set(Some(request.operation.proof_level())));
        OBSERVED_STATIC_NUMEL.with(|slot| slot.set(Some(ShapeTy::STATIC_NUMEL)));
        Ok(())
    }
}

/// Dispatch `op::Zeros` with the shape value `S` carries, and report the
/// proof level that reached the backend.
fn observed_proof_for<S: Shape>(expected: &incin::prelude::ShapeValue<S>) -> ProofLevel {
    OBSERVED.with(|slot| slot.set(None));
    OBSERVED_STATIC_NUMEL.with(|slot| slot.set(None));
    let context = ExecutionContext::new(RecordingBackend).with_grad_mode(GradMode::Disabled);
    dispatch::execute_shaped::<op::Zeros, _, S>(
        &context,
        CreationAttributes {
            shape: expected.dims(),
            dtype: DTypeId::F32.descriptor(),
            device: DeviceId::cpu(),
        },
        &[],
        expected,
    )
    .expect("the recording backend supports everything");
    OBSERVED
        .with(Cell::get)
        .expect("execute must have been reached")
}

fn observed_static_numel_for<S: Shape>(expected: &incin::prelude::ShapeValue<S>) -> Option<usize> {
    observed_proof_for(expected);
    OBSERVED_STATIC_NUMEL
        .with(Cell::get)
        .expect("execute must have recorded the shape constants")
}

/// A fully static shape reaches the backend as `Static`.
#[test]
fn a_static_shape_reaches_the_backend_as_static() {
    type S23 = incin::prelude::s![2, 3];
    let sv = incin::prelude::ShapeValue::<S23>::try_new(
        <S23 as Shape>::resolve(((), ((), ()))).unwrap(),
    )
    .unwrap();
    assert_eq!(observed_proof_for(&sv), ProofLevel::Static);
    assert_eq!(observed_static_numel_for(&sv), Some(6));
}

/// One runtime axis weakens the whole shape to `Mixed`.
#[test]
fn a_partial_shape_reaches_the_backend_as_mixed() {
    type SPartial = incin::prelude::s![usize, 784];
    let sv = incin::prelude::ShapeValue::<SPartial>::try_new(
        <SPartial as Shape>::resolve((4, ((), ()))).unwrap(),
    )
    .unwrap();
    assert_eq!(observed_proof_for(&sv), ProofLevel::Mixed);
}

/// A runtime rank proves nothing before the data exists.
#[test]
fn a_dynamic_shape_reaches_the_backend_as_dynamic() {
    let sv = incin::prelude::ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&[2, 3])).unwrap();
    assert_eq!(observed_proof_for(&sv), ProofLevel::Dynamic);
    assert_eq!(observed_static_numel_for(&sv), None);
}

/// `execute` — the evidence-free entry point — must keep claiming nothing.
#[test]
fn the_evidence_free_entry_point_still_claims_nothing() {
    OBSERVED.with(|slot| slot.set(None));
    let context = ExecutionContext::new(RecordingBackend).with_grad_mode(GradMode::Disabled);
    dispatch::execute::<op::Zeros, _>(
        &context,
        CreationAttributes {
            shape: vec![2, 3],
            dtype: DTypeId::F32.descriptor(),
            device: DeviceId::cpu(),
        },
        &[],
    )
    .expect("the recording backend supports everything");
    assert_eq!(OBSERVED.with(Cell::get), Some(ProofLevel::Dynamic));
}

/// ADVERSARIAL TEST: Typed shape expected does not match inferred output geometry.
/// Verification: Backend Execute must NEVER be called, and an error must be returned.
#[test]
fn adversarial_shape_mismatch_never_reaches_backend() {
    OBSERVED.with(|slot| slot.set(None));
    let context = ExecutionContext::new(RecordingBackend).with_grad_mode(GradMode::Disabled);
    type S23 = incin::prelude::s![2, 3];
    let expected = incin::prelude::ShapeValue::<S23>::try_new(
        <S23 as Shape>::resolve(((), ((), ()))).unwrap(),
    )
    .unwrap();
    let result = dispatch::execute_shaped::<op::Zeros, _, _>(
        &context,
        CreationAttributes {
            shape: vec![100], // Mismatch! Inferred [100] != expected [2, 3]
            dtype: DTypeId::F32.descriptor(),
            device: DeviceId::cpu(),
        },
        &[],
        &expected,
    );
    assert!(result.is_err(), "mismatched shape must return error");
    assert_eq!(
        OBSERVED.with(Cell::get),
        None,
        "backend execute must NEVER be reached on shape mismatch"
    );
}
