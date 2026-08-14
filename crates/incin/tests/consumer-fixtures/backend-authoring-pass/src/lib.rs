use incin::backend_authoring::operations::{
    CreationAttributes, Descriptor, NoAttributes, OPERATION_CATALOG, op,
};
use incin::backend_authoring::{
    Alignment, AutogradBackend, Backend, Capabilities, CapabilityQuery, Execute, ExecutionDescriptor,
    ExecutionRequest, Operation, OperationKey, ShapeBuf, StorageBackend,
    SupportLevel, TensorBackend, TensorMeta, VariableBackend,
};
use incin::prelude::{
    BackendError, Cpu, DType, DTypeDescriptor, DTypeId, DeviceId, Shape,
};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompanyAttributes {
    pub shape: ShapeBuf,
}

#[derive(Debug, Clone)]
pub struct CompanyOp;

impl Operation for CompanyOp {
    type Attributes = CompanyAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("identity"),
        version: 1,
    };

    fn infer_outputs(
        attributes: &CompanyAttributes,
        _inputs: &[incin::backend_authoring::LogicalTensorMeta],
    ) -> Result<Vec<incin::backend_authoring::LogicalTensorMeta>, incin::backend_authoring::DescriptorError> {
        Ok(vec![incin::backend_authoring::LogicalTensorMeta {
            shape: Some(attributes.shape.clone()),
            dtype: Some(incin::prelude::DTypeId::F32.descriptor()),
            device: Some(DeviceId::cpu()),
        }])
    }
}

#[derive(Debug, Clone, Default)]
pub struct CompanyBackend;

impl StorageBackend for CompanyBackend {
    const BACKEND_NAME: &'static str = "company";
    type Storage<K: DType> = ShapeBuf;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        let shape = storage.clone();
        Box::leak(Box::new(
            TensorMeta::contiguous(
                shape.clone(),
                K::descriptor(&K::Field::default()),
                DeviceId::cpu(),
                Alignment::of::<f32>(),
                shape.numel().expect("fixture shape"),
            )
            .expect("fixture metadata"),
        ))
    }
}

impl Capabilities for CompanyBackend {
    fn support(&self, _query: &CapabilityQuery) -> SupportLevel {
        SupportLevel::Native
    }
}

impl Execute<CompanyOp> for CompanyBackend {
    type Output = ShapeBuf;

    fn execute(
        &self,
        request: ExecutionRequest<'_, CompanyOp, Self>,
    ) -> Result<Self::Output, BackendError> {
        Ok(ShapeBuf::from_slice(
            &request.operation.descriptor().attributes().shape,
        ))
    }
}

impl Execute<op::Zeros> for CompanyBackend {
    type Output = ShapeBuf;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Zeros, Self>,
    ) -> Result<Self::Output, BackendError> {
        Ok(ShapeBuf::from_slice(
            &request.operation.descriptor().attributes().shape,
        ))
    }
}

impl Backend for CompanyBackend {
    type InnerBackend = Self;

    fn format_tensor_display<K: DType>(_: &<Self as StorageBackend>::Storage<K>) -> String {
        String::from("company")
    }

    fn format_tensor_debug<K: DType>(_: &<Self as StorageBackend>::Storage<K>) -> String {
        String::from("company")
    }

    fn to_bytes<K: DType>(_: &<Self as StorageBackend>::Storage<K>) -> incin::prelude::Result<Vec<u8>> {
        Ok(Vec::new())
    }

    fn from_bytes<K: DType>(
        _: &[u8],
        shape: &[usize],
        _: DTypeDescriptor,
        _: &DeviceId,
    ) -> incin::prelude::Result<<Self as StorageBackend>::Storage<K>> {
        Ok(ShapeBuf::from_slice(shape))
    }

}

impl VariableBackend for CompanyBackend {
    type RawVar = ShapeBuf;

    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> incin::prelude::Result<<Self as StorageBackend>::Storage<K>> {
        Ok(var.clone())
    }

    fn var_from_tensor<K: DType>(storage: &<Self as StorageBackend>::Storage<K>) -> incin::prelude::Result<Self::RawVar> {
        Ok(storage.clone())
    }

    fn assign_var<K: DType>(var: &mut Self::RawVar, storage: &<Self as StorageBackend>::Storage<K>) -> incin::prelude::Result<()> {
        *var = storage.clone();
        Ok(())
    }
}

impl AutogradBackend for CompanyBackend {
    type Grads = ();

    fn backward<K: DType>(_: &<Self as StorageBackend>::Storage<K>) -> incin::prelude::Result<Self::Grads> { Ok(()) }

    fn get_grad<K: DType>(
        _: &<Self as StorageBackend>::Storage<K>,
        _: &Self::Grads,
    ) -> incin::prelude::Result<Option<<Self as StorageBackend>::Storage<K>>> { Ok(None) }
}

pub fn accepts_backend_contract<B, O>()
where
    B: TensorBackend<f32> + StorageBackend + Execute<O>,
    O: ExecutionDescriptor + Operation,
{
}

pub fn exact_descriptor_contract(_: Option<Descriptor<op::Add>>) -> (&'static str, NoAttributes) {
    (
        OPERATION_CATALOG
            .iter()
            .find(|row| row.name == "add")
            .expect("add is in the canonical catalog")
            .name,
        NoAttributes,
    )
}

pub fn custom_backend_contract() -> ShapeBuf {
    let context = incin::backend_authoring::ExecutionContext::new(CompanyBackend);
    let attributes = CompanyAttributes {
        shape: ShapeBuf::from_slice(&[2, 3]),
    };
    incin::backend_authoring::execute::<CompanyOp, _>(&context, attributes, &[])
        .expect("custom backend operation")
}

pub fn custom_backend_runs_builtin_operation() -> ShapeBuf {
    let context = incin::backend_authoring::ExecutionContext::new(CompanyBackend);
    let attributes = CreationAttributes {
        shape: vec![2, 3],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
    };
    incin::backend_authoring::execute::<op::Zeros, _>(&context, attributes, &[])
        .expect("custom backend built-in operation")
}

pub fn built_in_operation_contract<B>()
where
    B: TensorBackend<f32> + StorageBackend + Execute<op::Zeros>,
{
}

pub fn external_device_identity_contract() -> DeviceId {
    let device = DeviceId::custom(0x434f_4d50_414e_5901, 7);
    assert_eq!(device.kind().custom_key(), Some(0x434f_4d50_414e_5901));
    assert_eq!(device.ordinal(), 7);
    device
}
