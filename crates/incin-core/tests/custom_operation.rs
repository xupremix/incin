//! Public downstream proof for the open custom operation contract.

extern crate incin_core as incin;

use core::marker::PhantomData;
use incin_backends::cpu::CpuBackendImpl;
use incin_backends::cpu::{CpuBuffer, CpuStorage};
use incin_core::backend_authoring::{
    DescriptorError, Execute, ExecutionRequest, LogicalTensorMeta, Operation, OperationKey,
    StorageBackend, execute, execute_shaped, execute_with_payload,
};
use incin_core::exec::TensorHandle;
use incin_core::exec::catalog::CreationAttributes;
use incin_core::exec::catalog::NoAttributes;
use incin_core::exec::{
    Capabilities, ExecutionContext, OperationIdentity, ProofLevel, SupportLevel, op,
};
use incin_core::prelude::{
    Backend, BackendError, Cpu, DTypeId, Device, DeviceId, Local, Shape, ShapeBuf, ShapeValue,
    TracingBackend, extract_graph,
};
use incin_core::test_utils::DummyBackend;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct IdentityAttributes {
    shape: ShapeBuf,
}

#[derive(Debug, Clone)]
struct CompanyIdentity;

impl Operation for CompanyIdentity {
    type Attributes = IdentityAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("identity"),
        version: 1,
    };

    fn infer_outputs(
        attributes: &Self::Attributes,
        _inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(vec![LogicalTensorMeta {
            shape: Some(attributes.shape.clone()),
            dtype: Some(DTypeId::F32.descriptor()),
            device: Some(incin_core::prelude::DeviceId::cpu()),
        }])
    }
}

#[derive(Debug, Clone, Default)]
struct CompanyBackend;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CompanyDevice;

impl Device for CompanyDevice {
    type Arg = ();
    type Field = PhantomData<Self>;

    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }

    fn to_incin(_: &Self::Field) -> incin_core::prelude::Result<DeviceId> {
        Ok(DeviceId::custom(0x434f_4d50_414e_5901, 7))
    }
}

#[derive(Debug, Clone)]
struct StaticRankProbe;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct StaticRankProbeAttributes {
    shape: ShapeBuf,
}

impl Operation for StaticRankProbe {
    type Attributes = StaticRankProbeAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("static_rank_probe"),
        version: 1,
    };

    fn infer_outputs(
        attributes: &Self::Attributes,
        _: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(vec![LogicalTensorMeta {
            shape: Some(attributes.shape.clone()),
            dtype: Some(DTypeId::U32.descriptor()),
            device: Some(DeviceId::cpu()),
        }])
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct PayloadAttributes;

#[derive(Debug, Clone)]
struct PayloadOperation;

impl Operation for PayloadOperation {
    type Attributes = PayloadAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("payload"),
        version: 1,
    };

    fn infer_outputs(
        _: &Self::Attributes,
        _: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone)]
struct MetadataFreeOperation;

impl Operation for MetadataFreeOperation {
    type Attributes = NoAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("metadata_free"),
        version: 1,
    };

    fn infer_outputs(
        _: &Self::Attributes,
        _: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(vec![LogicalTensorMeta {
            shape: Some(ShapeBuf::SCALAR),
            dtype: None,
            device: None,
        }])
    }
}

#[derive(Debug, Clone)]
struct CpuIdentityOperation;

impl Operation for CpuIdentityOperation {
    type Attributes = NoAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("cpu_identity"),
        version: 1,
    };

    fn infer_outputs(
        _: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

impl StorageBackend for CompanyBackend {
    const BACKEND_NAME: &'static str = "company";
    type Storage<K: incin_core::prelude::DType> = ();
    type Device = Cpu;

    fn metadata<K: incin_core::prelude::DType>(
        _storage: &Self::Storage<K>,
    ) -> &incin_core::backend_authoring::TensorMeta {
        unreachable!("the custom operation test has no input storage")
    }
}

impl Backend for CompanyBackend {
    type RawVar = ();
    type Grads = ();
    type InnerBackend = Self;

    fn shape<K: incin_core::prelude::DType>(
        _storage: &<Self as StorageBackend>::Storage<K>,
    ) -> ShapeBuf {
        ShapeBuf::SCALAR
    }
}

impl Capabilities for CompanyBackend {
    fn support(&self, query: &incin_core::exec::CapabilityQuery) -> SupportLevel {
        match &query.operation {
            OperationIdentity::Custom(key) if *key == CompanyIdentity::KEY => SupportLevel::Native,
            OperationIdentity::Custom(key) if *key == StaticRankProbe::KEY => SupportLevel::Native,
            OperationIdentity::Builtin(incin_core::prelude::OperationKind::Zeros) => {
                SupportLevel::Native
            }
            OperationIdentity::Builtin(incin_core::prelude::OperationKind::TensorFromBytes) => {
                SupportLevel::Native
            }
            other => panic!("unexpected capability query: {other:?}"),
        }
    }
}

impl Execute<StaticRankProbe> for CompanyBackend {
    type Output = i32;

    fn execute_shaped<S: Shape>(
        &self,
        _: ExecutionRequest<'_, StaticRankProbe, Self>,
    ) -> Result<Self::Output, BackendError> {
        Ok(S::RANK.map_or(-1, |rank| rank as i32))
    }
}

impl Execute<CompanyIdentity> for CompanyBackend {
    type Output = ProofLevel;

    fn execute_shaped<S: Shape>(
        &self,
        request: ExecutionRequest<'_, CompanyIdentity, Self>,
    ) -> Result<Self::Output, BackendError> {
        assert_eq!(
            request.operation.descriptor().identity(),
            &OperationIdentity::Custom(CompanyIdentity::KEY)
        );
        Ok(request.operation.proof_level())
    }
}

impl Execute<PayloadOperation> for CompanyBackend {
    type Output = Vec<u8>;

    fn execute_shaped<S: Shape>(
        &self,
        request: ExecutionRequest<'_, PayloadOperation, Self>,
    ) -> Result<Self::Output, BackendError> {
        Ok(request.payload.unwrap_or_default().to_vec())
    }
}

impl Execute<MetadataFreeOperation> for CompanyBackend {
    type Output = ProofLevel;

    fn supports_custom_operation(
        &self,
        operation: &OperationIdentity,
        _: bool,
        _: incin_core::exec::MathMode,
    ) -> SupportLevel {
        assert_eq!(
            operation,
            &OperationIdentity::Custom(MetadataFreeOperation::KEY)
        );
        SupportLevel::Unsupported(incin_core::exec::UnsupportedReason::CustomOperation {
            operation: MetadataFreeOperation::KEY,
        })
    }

    fn execute_shaped<S: Shape>(
        &self,
        request: ExecutionRequest<'_, MetadataFreeOperation, Self>,
    ) -> Result<Self::Output, BackendError> {
        Ok(request.operation.proof_level())
    }
}

impl Execute<CpuIdentityOperation> for CpuBackendImpl<Cpu> {
    type Output = CpuStorage;

    fn supports_custom(&self, query: &incin_core::exec::CapabilityQuery) -> SupportLevel {
        assert_eq!(
            query.operation,
            OperationIdentity::Custom(CpuIdentityOperation::KEY)
        );
        if query.rank > 2 {
            SupportLevel::Unsupported(incin_core::exec::UnsupportedReason::CustomOperation {
                operation: CpuIdentityOperation::KEY,
            })
        } else {
            SupportLevel::Native
        }
    }

    fn execute_shaped<S: Shape>(
        &self,
        request: ExecutionRequest<'_, CpuIdentityOperation, Self>,
    ) -> Result<Self::Output, BackendError> {
        request
            .inputs
            .first()
            .and_then(|input| input.downcast_ref::<CpuStorage>())
            .cloned()
            .ok_or(BackendError::InvalidInput {
                operation: incin_core::prelude::OperationKind::Pointwise,
                reason: "cpu identity requires one CPU input",
            })
    }
}

impl Execute<op::Zeros> for CompanyBackend {
    type Output = ProofLevel;

    fn execute_shaped<S: Shape>(
        &self,
        request: ExecutionRequest<'_, op::Zeros, Self>,
    ) -> Result<Self::Output, BackendError> {
        Ok(request.operation.proof_level())
    }
}

impl Execute<op::TensorFromBytes> for CompanyBackend {
    type Output = ProofLevel;

    fn execute_shaped<S: Shape>(
        &self,
        request: ExecutionRequest<'_, op::TensorFromBytes, Self>,
    ) -> Result<Self::Output, BackendError> {
        Ok(request.operation.proof_level())
    }
}

impl Execute<CompanyIdentity> for CpuBackendImpl<Cpu> {
    type Output = ProofLevel;

    fn supports_custom(&self, query: &incin_core::exec::CapabilityQuery) -> SupportLevel {
        assert_eq!(
            query.operation,
            OperationIdentity::Custom(CompanyIdentity::KEY)
        );
        SupportLevel::Native
    }

    fn execute_shaped<S: Shape>(
        &self,
        request: ExecutionRequest<'_, CompanyIdentity, Self>,
    ) -> Result<Self::Output, BackendError> {
        Ok(request.operation.proof_level())
    }
}

#[test]
fn downstream_custom_operation_keeps_static_shape_dispatch() {
    type S = incin::prelude::s![2, 3];
    let expected = ShapeValue::<S>::try_new(ShapeBuf::from_slice(&[2, 3])).unwrap();
    let context = ExecutionContext::new(CompanyBackend);
    let output = execute_shaped::<CompanyIdentity, _, S>(
        &context,
        IdentityAttributes {
            shape: ShapeBuf::from_slice(&[2, 3]),
        },
        &[],
        &expected,
    )
    .unwrap();
    assert_eq!(output, ProofLevel::Static);
}

#[test]
fn downstream_executor_receives_the_exact_shape_type() {
    type S = incin::prelude::s![2, 3];
    let expected = ShapeValue::<S>::try_new(ShapeBuf::from_slice(&[2, 3])).unwrap();
    let context = ExecutionContext::new(CompanyBackend);
    let static_rank = execute_shaped::<StaticRankProbe, _, S>(
        &context,
        StaticRankProbeAttributes {
            shape: ShapeBuf::from_slice(&[2, 3]),
        },
        &[],
        &expected,
    )
    .unwrap();
    assert_eq!(static_rank, 2);

    let runtime_rank = execute::<StaticRankProbe, _>(
        &context,
        StaticRankProbeAttributes {
            shape: ShapeBuf::from_slice(&[2, 3]),
        },
        &[],
    )
    .unwrap();
    assert_eq!(runtime_rank, -1);
}

#[test]
fn downstream_device_implementation_keeps_custom_identity() {
    let identity = CompanyDevice::to_incin(&PhantomData).unwrap();
    assert_eq!(identity.kind().custom_key(), Some(0x434f_4d50_414e_5901));
    assert_eq!(identity.ordinal(), 7);
}

#[test]
fn downstream_custom_operation_uses_unified_runtime_dispatch() {
    let context = ExecutionContext::new(CompanyBackend);
    let output = execute::<CompanyIdentity, _>(
        &context,
        IdentityAttributes {
            shape: ShapeBuf::from_slice(&[2, 3]),
        },
        &[],
    )
    .unwrap();
    assert_eq!(output, ProofLevel::Dynamic);
}

#[test]
fn custom_identity_is_derived_from_the_operation_key() {
    let context = ExecutionContext::new(CompanyBackend);
    let expected =
        ShapeValue::<incin::prelude::s![2, 3]>::try_new(ShapeBuf::from_slice(&[2, 3])).unwrap();
    let _ = execute_shaped::<CompanyIdentity, _, incin::prelude::s![2, 3]>(
        &context,
        IdentityAttributes {
            shape: ShapeBuf::from_slice(&[2, 3]),
        },
        &[],
        &expected,
    )
    .unwrap();
}

#[test]
fn custom_operations_are_recorded_with_their_key() {
    let _ = extract_graph();
    let context = ExecutionContext::new(TracingBackend::<CompanyBackend>::default());
    let expected =
        ShapeValue::<incin::prelude::s![2, 3]>::try_new(ShapeBuf::from_slice(&[2, 3])).unwrap();
    let _ = execute_shaped::<CompanyIdentity, _, incin::prelude::s![2, 3]>(
        &context,
        IdentityAttributes {
            shape: ShapeBuf::from_slice(&[2, 3]),
        },
        &[],
        &expected,
    )
    .unwrap();
    let graph = extract_graph();
    let node = graph.nodes.last().expect("custom trace node");
    assert_eq!(
        node.operation,
        OperationIdentity::Custom(CompanyIdentity::KEY)
    );
}

#[test]
fn downstream_backend_can_execute_a_builtin_operation() {
    type S = incin::prelude::s![2, 3];
    let expected = ShapeValue::<S>::try_new(ShapeBuf::from_slice(&[2, 3])).unwrap();
    let context = ExecutionContext::new(CompanyBackend);
    let output = execute_shaped::<op::Zeros, _, S>(
        &context,
        CreationAttributes {
            shape: vec![2, 3],
            dtype: DTypeId::F32.descriptor(),
            device: incin_core::prelude::DeviceId::cpu(),
        },
        &[],
        &expected,
    )
    .unwrap();
    assert_eq!(output, ProofLevel::Static);
}

#[test]
fn custom_operation_returns_backend_owned_storage() {
    let input = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![2.0, 4.0]), vec![2]).unwrap();
    let handle = TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&input);
    let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::default());
    let output = execute::<CpuIdentityOperation, _>(&context, NoAttributes, &[handle]).unwrap();
    assert_eq!(output.shape.as_ref(), &[2]);
    assert_eq!(output.get(&[0]), 2.0);
    assert_eq!(output.get(&[1]), 4.0);
}

#[test]
fn custom_operation_capability_can_reject_an_input_invocation() {
    let input =
        CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![1, 2, 2])
            .unwrap();
    let handle = TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&input);
    let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::default());
    let error = execute::<CpuIdentityOperation, _>(&context, NoAttributes, &[handle]).unwrap_err();
    assert!(matches!(
        error,
        incin_core::exec::CanonicalError::Backend(BackendError::Unsupported { .. })
    ));
}

#[test]
fn built_in_backend_can_compile_an_open_custom_operation() {
    fn assert_custom_execution<B>()
    where
        B: StorageBackend + Execute<CompanyIdentity>,
    {
    }

    assert_custom_execution::<DummyBackend<Cpu>>();
}

#[test]
fn built_in_cpu_backend_executes_a_downstream_operation() {
    type S = incin::prelude::s![2, 3];
    let expected = ShapeValue::<S>::try_new(ShapeBuf::from_slice(&[2, 3])).unwrap();
    let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::default());
    let output = execute_shaped::<CompanyIdentity, _, S>(
        &context,
        IdentityAttributes {
            shape: ShapeBuf::from_slice(&[2, 3]),
        },
        &[],
        &expected,
    )
    .unwrap();
    assert_eq!(output, ProofLevel::Static);
}

#[test]
fn custom_operations_receive_borrowed_execution_payloads() {
    let context = ExecutionContext::new(CompanyBackend);
    let output = execute_with_payload::<PayloadOperation, _>(
        &context,
        PayloadAttributes,
        &[],
        Some(&[1, 2, 3, 4]),
    )
    .unwrap();
    assert_eq!(output, vec![1, 2, 3, 4]);
}

#[test]
fn metadata_free_custom_operations_cannot_bypass_capability_admission() {
    let context = ExecutionContext::new(CompanyBackend);
    let error = execute::<MetadataFreeOperation, _>(&context, NoAttributes, &[]).unwrap_err();
    assert!(matches!(
        error,
        incin_core::exec::CanonicalError::Backend(BackendError::Unsupported { .. })
    ));

    let expected = ShapeValue::<incin_core::prelude::Dyn>::try_new(ShapeBuf::SCALAR).unwrap();
    let error = execute_shaped::<MetadataFreeOperation, _, incin_core::prelude::Dyn>(
        &context,
        NoAttributes,
        &[],
        &expected,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        incin_core::exec::CanonicalError::Backend(BackendError::Unsupported { .. })
    ));
}

#[test]
fn built_in_creation_requires_the_declared_payload() {
    let context = ExecutionContext::new(CompanyBackend);
    let attributes = incin_core::exec::catalog::DataAttributes {
        shape: vec![1],
        dtype: DTypeId::F32.descriptor(),
        device: incin_core::prelude::DeviceId::cpu(),
        payload: incin_core::exec::catalog::CreationPayload::Bytes { byte_len: 4 },
    };

    let error =
        execute_with_payload::<op::TensorFromBytes, _>(&context, attributes.clone(), &[], None)
            .unwrap_err();
    assert!(matches!(
        error,
        incin_core::exec::CanonicalError::Descriptor(
            incin_core::exec::DescriptorError::PayloadMissing {
                operation: incin_core::prelude::OperationKind::TensorFromBytes,
            }
        )
    ));

    let error =
        execute_with_payload::<op::TensorFromBytes, _>(&context, attributes, &[], Some(&[1, 2]))
            .unwrap_err();
    assert!(matches!(
        error,
        incin_core::exec::CanonicalError::Descriptor(
            incin_core::exec::DescriptorError::PayloadByteLength {
                operation: incin_core::prelude::OperationKind::TensorFromBytes,
                expected: 4,
                actual: 2,
            }
        )
    ));
}

#[test]
fn built_in_non_creation_rejects_an_execution_payload() {
    let context = ExecutionContext::new(CompanyBackend);
    let error = execute_with_payload::<op::Zeros, _>(
        &context,
        incin_core::exec::catalog::CreationAttributes {
            shape: vec![1],
            dtype: DTypeId::F32.descriptor(),
            device: incin_core::prelude::DeviceId::cpu(),
        },
        &[],
        Some(&[1]),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        incin_core::exec::CanonicalError::Descriptor(
            incin_core::exec::DescriptorError::UnexpectedPayload {
                operation: incin_core::prelude::OperationKind::Zeros,
            }
        )
    ));
}
