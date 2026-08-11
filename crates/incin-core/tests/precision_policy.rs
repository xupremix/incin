use incin_core::exec::{
    LossScaleState, LossScaling, PrecisionCapabilities, PrecisionChoice, PrecisionRequest,
    ResolvedPrecision, RuntimePrecisionPolicy, resolve_precision,
};
use incin_core::prelude::{DTypeId, Error, OperationKind};

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
