//! Public downstream proof for the open custom operation contract.

extern crate incin_core as incin;

use incin_core::backend_authoring::{
    DescriptorError, Execute, ExecutionRequest, LogicalTensorMeta, Operation, OperationKey,
    StorageBackend, execute_custom_shaped,
};
use incin_core::exec::{
    Capabilities, CustomCapabilityQuery, ExecutionContext, ProofLevel, SupportLevel,
};
use incin_core::prelude::{BackendError, Cpu, DTypeId, Shape, ShapeBuf, ShapeValue};

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

impl Capabilities for CompanyBackend {
    fn support(&self, _query: &incin_core::exec::CapabilityQuery) -> SupportLevel {
        SupportLevel::Native
    }

    fn support_custom(&self, query: &CustomCapabilityQuery) -> SupportLevel {
        assert_eq!(query.operation, CompanyIdentity::KEY);
        SupportLevel::Native
    }
}

impl Execute<incin_core::backend_authoring::Descriptor<CompanyIdentity>> for CompanyBackend {
    type Output = ProofLevel;

    fn execute_shaped<S: Shape>(
        &self,
        request: ExecutionRequest<
            '_,
            incin_core::backend_authoring::Descriptor<CompanyIdentity>,
            Self,
        >,
    ) -> Result<Self::Output, BackendError> {
        Ok(request.operation.proof_level())
    }
}

#[test]
fn downstream_custom_operation_keeps_static_shape_dispatch() {
    type S = incin::prelude::s![2, 3];
    let expected = ShapeValue::<S>::try_new(ShapeBuf::from_slice(&[2, 3])).unwrap();
    let context = ExecutionContext::new(CompanyBackend);
    let output = execute_custom_shaped::<CompanyIdentity, _, S>(
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
