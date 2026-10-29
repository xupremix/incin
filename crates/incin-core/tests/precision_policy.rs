//! Integration coverage for `MockCpuBackend` on the documented public surface,
//! plus the precision axis of the scoped execution policy.
use std::cell::Cell;
use std::rc::Rc;

use incin_core::backend_authoring::{Execute, ExecutionRequest, StorageBackend};
use incin_core::exec::{
    Capabilities, CapabilityQuery, ExecutionContext, ExecutionPolicy, LossScaleState, LossScaling,
    PrecisionCapabilities, PrecisionChoice, PrecisionRequest, ResolvedPrecision,
    RuntimePrecisionPolicy, SupportLevel, TensorMeta, op, resolve_precision,
};
use incin_core::prelude::{BackendError, Cpu, DType, DTypeId, Error, OperationKind};

struct MockCpuBackend;

impl PrecisionCapabilities for MockCpuBackend {
    fn native_precision(&self, request: &PrecisionRequest) -> Result<ResolvedPrecision, Error> {
        if request.storage == DTypeId::F32.descriptor()
            || request.storage == DTypeId::F64.descriptor()
        {
            Ok(ResolvedPrecision::new(
                request.storage,
                request.storage,
                request.storage,
                request.output,
                LossScaling::None,
            ))
        } else if request.storage == DTypeId::F16.descriptor()
            || request.storage == DTypeId::BF16.descriptor()
        {
            Ok(ResolvedPrecision::new(
                request.storage,
                DTypeId::F32.descriptor(),
                DTypeId::F32.descriptor(),
                request.output,
                LossScaling::None,
            ))
        } else {
            Err(Error::UnsupportedDType {
                dtype: request.storage,
                backend: "MockCpu",
                op: "native_precision",
            })
        }
    }
}

#[test]
fn precision_policy_presets_and_construction() {
    let fp32 = RuntimePrecisionPolicy::fp32();
    assert_eq!(fp32.parameter(), DTypeId::F32.descriptor());
    assert_eq!(fp32.active_dtype(), None);
    assert_eq!(fp32.compute(), PrecisionChoice::Native);
    assert_eq!(fp32.accumulator(), PrecisionChoice::Native);
    assert_eq!(fp32.loss_scaling(), LossScaling::None);

    let fp16 = RuntimePrecisionPolicy::mixed_f16();
    assert_eq!(fp16.active_dtype(), Some(DTypeId::F16.descriptor()));
    assert_eq!(fp16.compute(), PrecisionChoice::Native);
    assert_eq!(
        fp16.accumulator(),
        PrecisionChoice::Exact(DTypeId::F32.descriptor())
    );

    let bf16 = RuntimePrecisionPolicy::mixed_bf16();
    assert_eq!(bf16.active_dtype(), Some(DTypeId::BF16.descriptor()));
    assert_eq!(bf16.compute(), PrecisionChoice::Native);
    assert_eq!(
        bf16.accumulator(),
        PrecisionChoice::Exact(DTypeId::F32.descriptor())
    );
}

#[test]
fn dynamic_loss_scaling_state_growth_and_overflow_backoff() {
    let policy = LossScaling::dynamic(1024.0, 2.0, 0.5, 3);
    let mut state = LossScaleState::new(policy);
    assert_eq!(state.scale(), 1024.0);

    // 2 finite steps -> scale unchanged
    state.update(false);
    state.update(false);
    assert_eq!(state.scale(), 1024.0);

    // 3rd finite step -> growth factor applied (1024 * 2 = 2048)
    state.update(false);
    assert_eq!(state.scale(), 2048.0);

    // Overflow detected -> backoff factor applied (2048 * 0.5 = 1024)
    state.update(true);
    assert_eq!(state.scale(), 1024.0);

    // Backoff cannot reduce scale below 1.0
    let min_policy = LossScaling::dynamic(1.0, 2.0, 0.5, 3);
    let mut min_state = LossScaleState::new(min_policy);
    min_state.update(true);
    assert_eq!(min_state.scale(), 1.0);
}

#[test]
fn resolve_precision_exact_mismatch_returns_unsupported_precision() {
    let backend = MockCpuBackend;
    let mut policy = RuntimePrecisionPolicy::fp32();
    // Demand compute exact F16 on a backend that computes F16 natively in F32
    policy = policy.with_compute(PrecisionChoice::Exact(DTypeId::F16.descriptor()));

    let req = PrecisionRequest::new(
        OperationKind::Pointwise,
        DTypeId::F16.descriptor(),
        DTypeId::F16.descriptor(),
        incin_core::exec::LayoutClass::Contiguous,
        1,
        false,
        incin_core::exec::MathMode::Fast,
    );

    let result = resolve_precision(&backend, policy, &req);
    assert!(result.is_err());
    if let Err(Error::UnsupportedPrecision {
        requested, role, ..
    }) = result
    {
        assert_eq!(requested, DTypeId::F16.descriptor());
        assert_eq!(role, incin_core::exec::PrecisionRole::Compute);
    } else {
        panic!("Expected Error::UnsupportedPrecision, got {:?}", result);
    }
}

#[test]
fn resolve_precision_native_success() {
    let backend = MockCpuBackend;
    let policy = RuntimePrecisionPolicy::fp32();
    let req = PrecisionRequest::new(
        OperationKind::Pointwise,
        DTypeId::F32.descriptor(),
        DTypeId::F32.descriptor(),
        incin_core::exec::LayoutClass::Contiguous,
        1,
        false,
        incin_core::exec::MathMode::Fast,
    );

    let resolved = resolve_precision(&backend, policy, &req).unwrap();
    assert_eq!(resolved.storage, DTypeId::F32.descriptor());
    assert_eq!(resolved.compute, DTypeId::F32.descriptor());
    assert_eq!(resolved.accumulator, DTypeId::F32.descriptor());
}

/// A backend that owns no device and executes nothing. The context under test
/// is generic over `StorageBackend`, so what this one does is irrelevant; it
/// exists so `ExecutionContext::from_scope` has a backend to own.
#[derive(Debug, Clone)]
struct Probe;

impl StorageBackend for Probe {
    const BACKEND_NAME: &'static str = "Probe";
    type Storage<K: DType> = ();
    type Device = Cpu;

    fn metadata<K: DType>(_: &Self::Storage<K>) -> &TensorMeta {
        unreachable!("precision-scope tests never inspect storage")
    }
}

/// A scope that installs a precision makes it what `from_scope` reads, and the
/// caller's ambient policy comes back when the scope ends. The context keeps
/// the value it read, because it is an owned snapshot rather than a live view.
#[test]
fn a_precision_scope_is_visible_to_from_scope_and_restored_on_exit() {
    let ambient = ExecutionPolicy::current();
    let target = RuntimePrecisionPolicy::mixed_bf16();
    assert_ne!(ambient.precision, target);

    let context = ambient.with_precision(target).scope(|| {
        assert_eq!(ExecutionPolicy::current().precision, target);
        let context = ExecutionContext::from_scope(Probe);
        assert_eq!(context.precision_policy(), target);
        context
    });

    assert_eq!(ExecutionPolicy::current(), ambient);
    assert_eq!(context.precision_policy(), target);
}

/// The scoped precision reaches the backend on `ExecutionRequest::context`:
/// `from_scope` reads it once, dispatch hands that context to `execute`, and
/// the backend observes the same value the scope installed.
#[test]
fn the_scoped_precision_reaches_the_backend_at_dispatch() {
    let seen = Rc::new(Cell::new(None));
    let spy = PrecisionSpy { seen: seen.clone() };
    let ambient = ExecutionPolicy::current();
    let target = RuntimePrecisionPolicy::mixed_bf16();

    ambient.with_precision(target).scope(|| {
        let context = ExecutionContext::from_scope(spy.clone());
        let attributes = incin_core::exec::catalog::CreationAttributes {
            shape: vec![1],
            dtype: DTypeId::F32.descriptor(),
            device: incin_core::prelude::DeviceId::cpu(),
        };
        incin_core::backend_authoring::execute::<op::Zeros, _>(&context, attributes, &[])
            .expect("native support launches");
    });

    assert_eq!(seen.get(), Some(target));
    assert_eq!(ExecutionPolicy::current(), ambient);
}

/// A spy that records the precision policy dispatch handed it. Mirrors the
/// `op::Zeros` spy in `dispatch_policy.rs`, narrowed to this one axis.
#[derive(Clone)]
struct PrecisionSpy {
    seen: Rc<Cell<Option<RuntimePrecisionPolicy>>>,
}

impl StorageBackend for PrecisionSpy {
    const BACKEND_NAME: &'static str = "precision-spy";
    type Storage<K: DType> = ();
    type Device = Cpu;

    fn metadata<K: DType>(_: &Self::Storage<K>) -> &TensorMeta {
        unreachable!("precision spy uses zero-input dispatch")
    }
}

impl Capabilities for PrecisionSpy {
    fn support(&self, _: &CapabilityQuery) -> SupportLevel {
        SupportLevel::Native
    }
}

impl Execute<op::Zeros> for PrecisionSpy {
    type Output = ();

    fn execute(&self, request: ExecutionRequest<'_, op::Zeros, Self>) -> Result<(), BackendError> {
        self.seen.set(Some(request.context.precision_policy()));
        Ok(())
    }
}
