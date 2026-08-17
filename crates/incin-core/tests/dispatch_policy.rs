//! Policy admission is checked before every canonical dispatch route launches.

use std::cell::Cell;
use std::rc::Rc;

use incin_core::backend_authoring::{Execute, ExecutionRequest, StorageBackend};
use incin_core::exec::catalog::NoAttributes;
use incin_core::exec::{
    CanonicalError, Capabilities, CapabilityQuery, ExecutionContext, FallbackPolicy,
    LogicalTensorMeta, Operation, OperationIdentity, OperationKey, PolicyViolation, SupportLevel,
    TensorMeta, op,
};
use incin_core::prelude::{BackendError, Cpu, DType, DTypeId, DeviceId, Dyn, ShapeBuf, ShapeValue};

#[derive(Debug, Clone)]
struct Custom;

impl Operation for Custom {
    type Attributes = NoAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("policy.test"),
        name: std::borrow::Cow::Borrowed("custom"),
        version: 1,
    };

    fn infer_outputs(
        _: &Self::Attributes,
        _: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, incin_core::exec::DescriptorError> {
        Ok(vec![LogicalTensorMeta {
            shape: Some(ShapeBuf::SCALAR),
            dtype: Some(DTypeId::F32.descriptor()),
            device: Some(DeviceId::cpu()),
        }])
    }
}

#[derive(Debug, Clone)]
struct MetadataFree;

impl Operation for MetadataFree {
    type Attributes = NoAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("policy.test"),
        name: std::borrow::Cow::Borrowed("metadata_free"),
        version: 1,
    };

    fn infer_outputs(
        _: &Self::Attributes,
        _: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, incin_core::exec::DescriptorError> {
        Ok(vec![LogicalTensorMeta {
            shape: Some(ShapeBuf::SCALAR),
            dtype: None,
            device: None,
        }])
    }
}

#[derive(Clone)]
struct PolicySpy {
    support: SupportLevel,
    executions: Rc<Cell<usize>>,
}

impl PolicySpy {
    fn new(support: SupportLevel) -> Self {
        Self {
            support,
            executions: Rc::new(Cell::new(0)),
        }
    }

    fn executed(&self) -> usize {
        self.executions.get()
    }

    fn record(&self) {
        self.executions.set(self.executed() + 1);
    }
}

impl StorageBackend for PolicySpy {
    const BACKEND_NAME: &'static str = "policy-spy";
    type Storage<K: DType> = ();
    type Device = Cpu;

    fn metadata<K: DType>(_: &Self::Storage<K>) -> &TensorMeta {
        unreachable!("policy tests use zero-input dispatch")
    }
}

impl Capabilities for PolicySpy {
    fn support(&self, _: &CapabilityQuery) -> SupportLevel {
        self.support.clone()
    }
}

macro_rules! spy_execute {
    ($operation:ty) => {
        impl Execute<$operation> for PolicySpy {
            type Output = ();

            fn execute(
                &self,
                _: ExecutionRequest<'_, $operation, Self>,
            ) -> Result<Self::Output, BackendError> {
                self.record();
                Ok(())
            }
        }
    };
}

spy_execute!(op::Zeros);
spy_execute!(op::TensorFromBytes);

impl Execute<Custom> for PolicySpy {
    type Output = ();

    fn supports_custom(&self, _: &CapabilityQuery) -> SupportLevel {
        self.support.clone()
    }

    fn execute(&self, _: ExecutionRequest<'_, Custom, Self>) -> Result<(), BackendError> {
        self.record();
        Ok(())
    }
}

impl Execute<MetadataFree> for PolicySpy {
    type Output = ();

    fn supports_custom_operation(
        &self,
        _: &OperationIdentity,
        _: bool,
        _: incin_core::exec::MathMode,
    ) -> SupportLevel {
        self.support.clone()
    }

    fn execute(&self, _: ExecutionRequest<'_, MetadataFree, Self>) -> Result<(), BackendError> {
        self.record();
        Ok(())
    }
}

fn zeros() -> incin_core::exec::catalog::CreationAttributes {
    incin_core::exec::catalog::CreationAttributes {
        shape: vec![1],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
    }
}

fn policy_error(
    error: CanonicalError,
    operation: OperationIdentity,
    support: SupportLevel,
    fallback: FallbackPolicy,
) {
    assert_eq!(
        error,
        CanonicalError::Policy(PolicyViolation {
            operation,
            support,
            fallback,
        })
    );
}

#[test]
fn native_and_permitted_composed_and_transfer_support_launch() {
    for (support, fallback) in [
        (SupportLevel::Native, FallbackPolicy::Deny),
        (SupportLevel::Composed, FallbackPolicy::AllowComposition),
        (SupportLevel::Fallback, FallbackPolicy::AllowTransfer),
    ] {
        let spy = PolicySpy::new(support);
        let context = ExecutionContext::new(spy.clone()).with_fallback(fallback);
        incin_core::backend_authoring::execute::<op::Zeros, _>(&context, zeros(), &[]).unwrap();
        assert_eq!(spy.executed(), 1);
    }
}

#[test]
fn normal_and_shaped_builtin_routes_refuse_before_execute() {
    let spy = PolicySpy::new(SupportLevel::Composed);
    let context = ExecutionContext::new(spy.clone()).with_fallback(FallbackPolicy::Deny);
    let error =
        incin_core::backend_authoring::execute::<op::Zeros, _>(&context, zeros(), &[]).unwrap_err();
    policy_error(
        error,
        OperationIdentity::Builtin(incin_core::prelude::OperationKind::Zeros),
        SupportLevel::Composed,
        FallbackPolicy::Deny,
    );
    assert_eq!(spy.executed(), 0);

    let expected = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&[1])).unwrap();
    let error = incin_core::backend_authoring::execute_shaped::<op::Zeros, _, Dyn>(
        &context,
        zeros(),
        &[],
        &expected,
    )
    .unwrap_err();
    policy_error(
        error,
        OperationIdentity::Builtin(incin_core::prelude::OperationKind::Zeros),
        SupportLevel::Composed,
        FallbackPolicy::Deny,
    );
    assert_eq!(spy.executed(), 0);
}

#[test]
fn unsupported_capability_remains_a_backend_refusal() {
    let spy = PolicySpy::new(SupportLevel::Unsupported(
        incin_core::exec::UnsupportedReason::Operation {
            operation: incin_core::prelude::OperationKind::Zeros,
        },
    ));
    let context = ExecutionContext::new(spy.clone());
    let error =
        incin_core::backend_authoring::execute::<op::Zeros, _>(&context, zeros(), &[]).unwrap_err();
    assert!(matches!(
        error,
        CanonicalError::Backend(BackendError::Unsupported { .. })
    ));
    assert_eq!(spy.executed(), 0);
}

#[test]
fn payload_and_custom_routes_use_the_same_policy_gate() {
    let spy = PolicySpy::new(SupportLevel::Fallback);
    let context =
        ExecutionContext::new(spy.clone()).with_fallback(FallbackPolicy::AllowComposition);
    let attributes = incin_core::exec::catalog::DataAttributes {
        shape: vec![1],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
        payload: incin_core::exec::catalog::CreationPayload::Bytes { byte_len: 4 },
    };
    let error = incin_core::backend_authoring::execute_with_payload::<op::TensorFromBytes, _>(
        &context,
        attributes,
        &[],
        Some(&[0; 4]),
    )
    .unwrap_err();
    policy_error(
        error,
        OperationIdentity::Builtin(incin_core::prelude::OperationKind::TensorFromBytes),
        SupportLevel::Fallback,
        FallbackPolicy::AllowComposition,
    );
    assert_eq!(spy.executed(), 0);

    let expected = ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&[1])).unwrap();
    let attributes = incin_core::exec::catalog::DataAttributes {
        shape: vec![1],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
        payload: incin_core::exec::catalog::CreationPayload::Bytes { byte_len: 4 },
    };
    let error = incin_core::backend_authoring::execute_shaped_with_payload::<
        op::TensorFromBytes,
        _,
        Dyn,
    >(&context, attributes, &[], &expected, Some(&[0; 4]))
    .unwrap_err();
    policy_error(
        error,
        OperationIdentity::Builtin(incin_core::prelude::OperationKind::TensorFromBytes),
        SupportLevel::Fallback,
        FallbackPolicy::AllowComposition,
    );
    assert_eq!(spy.executed(), 0);

    let spy = PolicySpy::new(SupportLevel::Composed);
    let context = ExecutionContext::new(spy.clone()).with_fallback(FallbackPolicy::Deny);
    let error = incin_core::backend_authoring::execute::<Custom, _>(&context, NoAttributes, &[])
        .unwrap_err();
    policy_error(
        error,
        OperationIdentity::Custom(Custom::KEY),
        SupportLevel::Composed,
        FallbackPolicy::Deny,
    );
    assert_eq!(spy.executed(), 0);
}

#[test]
fn metadata_free_custom_route_cannot_bypass_policy_admission() {
    let spy = PolicySpy::new(SupportLevel::Fallback);
    let context = ExecutionContext::new(spy.clone()).with_fallback(FallbackPolicy::Deny);
    let error =
        incin_core::backend_authoring::execute::<MetadataFree, _>(&context, NoAttributes, &[])
            .unwrap_err();
    policy_error(
        error,
        OperationIdentity::Custom(MetadataFree::KEY),
        SupportLevel::Fallback,
        FallbackPolicy::Deny,
    );
    assert_eq!(spy.executed(), 0);
}
