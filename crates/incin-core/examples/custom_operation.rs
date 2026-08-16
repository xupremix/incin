//! A compact downstream custom-operation example.
//!
//! This uses the same `Operation`, `Descriptor`, and `Execute` path as a
//! backend crate. It returns proof metadata instead of allocating storage so
//! the authoring contract stays visible without hiding a backend kernel.

extern crate incin_core as incin;

use incin_core::backend_authoring::{
    Backend, Capabilities, DescriptorError, Execute, ExecutionContext, ExecutionRequest,
    LogicalTensorMeta, Operation, OperationKey, StorageBackend, execute_shaped,
};
use incin_core::exec::{CapabilityQuery, ProofLevel, SupportLevel};
use incin_core::prelude::{Cpu, DType, DTypeId, DeviceId, ShapeBuf, ShapeValue};
use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct BiasGeluAttributes {
    shape: ShapeBuf,
}

#[derive(Debug, Clone)]
struct BiasGelu;

impl Operation for BiasGelu {
    type Attributes = BiasGeluAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("example.org"),
        name: Cow::Borrowed("bias_gelu"),
        version: 1,
    };

    fn infer_outputs(
        attributes: &Self::Attributes,
        _inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        if attributes.shape.rank() != 2 {
            return Err(DescriptorError::InvalidAttribute {
                operation: incin_core::shapes::error::OperationKind::Pointwise,
                attribute: "shape",
                reason: "BiasGelu expects a rank-two activation",
            });
        }

        Ok(vec![LogicalTensorMeta {
            shape: Some(attributes.shape.clone()),
            dtype: Some(DTypeId::F32.descriptor()),
            device: Some(DeviceId::cpu()),
        }])
    }
}

#[derive(Debug, Clone, Default)]
struct ExampleBackend;

impl StorageBackend for ExampleBackend {
    const BACKEND_NAME: &'static str = "custom-operation-example";
    type Storage<K: DType> = ();
    type Device = Cpu;

    fn metadata<K: DType>(_: &Self::Storage<K>) -> &incin_core::backend_authoring::TensorMeta {
        unreachable!("this metadata-only example does not own tensor storage")
    }
}

impl Capabilities for ExampleBackend {
    fn support(&self, _: &CapabilityQuery) -> SupportLevel {
        SupportLevel::Native
    }
}

impl Backend for ExampleBackend {
    type InnerBackend = Self;
}

impl Execute<BiasGelu> for ExampleBackend {
    type Output = ProofLevel;

    fn execute(
        &self,
        request: ExecutionRequest<'_, BiasGelu, Self>,
    ) -> Result<Self::Output, incin_core::prelude::BackendError> {
        Ok(request.operation.proof_level())
    }
}

fn main() -> incin_core::prelude::Result<()> {
    type Shape = incin_core::prelude::s![4, 8];
    let expected = ShapeValue::<Shape>::try_new(ShapeBuf::from_slice(&[4, 8]))?;
    let result = execute_shaped::<BiasGelu, _, Shape>(
        &ExecutionContext::new(ExampleBackend),
        BiasGeluAttributes {
            shape: ShapeBuf::from_slice(&[4, 8]),
        },
        &[],
        &expected,
    )?;

    println!("BiasGelu executed with {result:?} output proof");
    Ok(())
}
