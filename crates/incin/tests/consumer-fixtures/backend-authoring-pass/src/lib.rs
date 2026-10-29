//! Integration coverage for `CompanyAttributes` on the documented public surface.
use incin::backend_authoring::operations::{
    CreationAttributes, Descriptor, NoAttributes, OPERATION_CATALOG, op,
};
use incin::backend_authoring::{
    Alignment, AutogradBackend, Backend, Capabilities, CapabilityQuery, Execute,
    ExecutionDescriptor, ExecutionRequest, Operation, OperationKey, ShapeBuf, StorageBackend,
    StorageTransfer, SupportLevel, SupportsDType, TensorBackend, TensorMeta, VariableBackend,
};
use incin::prelude::{BackendError, Cpu, DType, DTypeDescriptor, DTypeId, DeviceId, DeviceKey};

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
    ) -> Result<
        Vec<incin::backend_authoring::LogicalTensorMeta>,
        incin::backend_authoring::DescriptorError,
    > {
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

incin::backend_authoring::declare_capabilities! {
    for CompanyBackend {
        op::Zeros => company_support;
    }
}

fn company_support(_: &CompanyBackend, _: &CapabilityQuery) -> SupportLevel {
    SupportLevel::Native
}

incin::backend_authoring::declare_executors! {
    for CompanyBackend {
        CompanyOp => ShapeBuf = company_identity;
        op::Zeros => ShapeBuf = company_zeros;
    }
}

fn company_identity(
    _: &CompanyBackend,
    request: ExecutionRequest<'_, CompanyOp, CompanyBackend>,
) -> Result<ShapeBuf, BackendError> {
    Ok(ShapeBuf::from_slice(
        &request.operation.descriptor().attributes().shape,
    ))
}

fn company_zeros(
    _: &CompanyBackend,
    request: ExecutionRequest<'_, op::Zeros, CompanyBackend>,
) -> Result<ShapeBuf, BackendError> {
    Ok(ShapeBuf::from_slice(
        &request.operation.descriptor().attributes().shape,
    ))
}

impl Backend for CompanyBackend {
    type InnerBackend = Self;
}

impl incin::backend_authoring::HostInterop for CompanyBackend {
    fn host_format_display<K: DType>(_: &<Self as StorageBackend>::Storage<K>) -> String {
        String::from("company")
    }

    fn host_format_debug<K: DType>(_: &<Self as StorageBackend>::Storage<K>) -> String {
        String::from("company")
    }

    fn to_bytes<K: DType>(
        _: &<Self as StorageBackend>::Storage<K>,
    ) -> incin::prelude::Result<Vec<u8>> {
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

impl incin::backend_authoring::HostReadback for CompanyBackend {
    fn float_to_vec1<K: DType>(
        _: &<Self as StorageBackend>::Storage<K>,
    ) -> incin::prelude::Result<Vec<f64>> {
        Ok(Vec::new())
    }

    fn int_to_vec1<K: DType>(
        _: &<Self as StorageBackend>::Storage<K>,
    ) -> incin::prelude::Result<Vec<i64>> {
        Ok(Vec::new())
    }
}

impl VariableBackend for CompanyBackend {
    type Var<K: DType> = ShapeBuf;

    fn var_as_tensor<K: DType>(
        var: &Self::Var<K>,
    ) -> incin::prelude::Result<<Self as StorageBackend>::Storage<K>> {
        Ok(var.clone())
    }

    fn var_from_tensor<K: DType>(
        storage: &<Self as StorageBackend>::Storage<K>,
    ) -> incin::prelude::Result<Self::Var<K>> {
        Ok(storage.clone())
    }

    fn assign_var<K: DType>(
        var: &mut Self::Var<K>,
        storage: &<Self as StorageBackend>::Storage<K>,
    ) -> incin::prelude::Result<()> {
        *var = storage.clone();
        Ok(())
    }
}

impl AutogradBackend for CompanyBackend {
    type Grads = ();

    fn backward<K: DType>(
        _: &<Self as StorageBackend>::Storage<K>,
    ) -> incin::prelude::Result<Self::Grads> {
        Ok(())
    }

    fn get_grad<K: DType>(
        _: &<Self as StorageBackend>::Storage<K>,
        _: &Self::Grads,
    ) -> incin::prelude::Result<Option<<Self as StorageBackend>::Storage<K>>> {
        Ok(None)
    }

    /// Required since 0.1.0, so that post-backward transforms such as
    /// `clip_grad_norm` can be written once against the trait. A backend that
    /// records no gradients reports the refusal rather than accepting the
    /// write, because a silent accept turns clipping into a no-op the caller
    /// cannot detect.
    fn set_grad<K: DType>(
        _: &<Self as StorageBackend>::Storage<K>,
        _: &mut Self::Grads,
        _: <Self as StorageBackend>::Storage<K>,
    ) -> incin::prelude::Result<()> {
        Err(incin::Error::UnsupportedBackendOperation {
            op: "set_grad",
            backend: <Self as StorageBackend>::BACKEND_NAME,
        })
    }
}

/// A deliberately inference-only backend. It has no host serialization,
/// variable, or autograd implementation; `Backend` must still be enough to
/// execute a descriptor.
#[derive(Debug, Clone, Default)]
pub struct InferenceBackend;

impl StorageBackend for InferenceBackend {
    const BACKEND_NAME: &'static str = "inference-only";
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

impl Capabilities for InferenceBackend {
    fn support(&self, _query: &CapabilityQuery) -> SupportLevel {
        SupportLevel::Native
    }
}

impl<K: DType> SupportsDType<K> for InferenceBackend {
    fn resolve_dtype(
        field: &K::Field,
        _device: &DeviceId,
    ) -> incin::prelude::Result<DTypeDescriptor> {
        Ok(K::descriptor(field))
    }
}

impl Execute<op::Zeros> for InferenceBackend {
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

impl Backend for InferenceBackend {
    type InnerBackend = Self;
}

// Storage movement is an independent capability: this backend deliberately
// has no `VariableBackend` implementation.
impl StorageTransfer<Cpu> for InferenceBackend {
    type Output = Self;

    fn transfer_storage<K: DType>(
        storage: &<Self as StorageBackend>::Storage<K>,
        _dtype: &K::Field,
        _device: &<Cpu as incin::prelude::Device>::Field,
    ) -> incin::prelude::Result<<Self::Output as StorageBackend>::Storage<K>>
    where
        Self::Output: incin::backend_authoring::SupportsDType<K>,
    {
        Ok(storage.clone())
    }
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

pub fn inference_only_backend_runs_builtin_operation() -> ShapeBuf {
    let context = incin::backend_authoring::ExecutionContext::new(InferenceBackend);
    let attributes = CreationAttributes {
        shape: vec![2, 3],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
    };
    incin::backend_authoring::execute::<op::Zeros, _>(&context, attributes, &[])
        .expect("inference-only backend operation")
}

pub fn inference_only_backend_can_transfer_tensor()
-> incin::prelude::Result<incin::Tensor<incin::prelude::Dyn, InferenceBackend, f32>> {
    let tensor = incin::Tensor::<incin::prelude::Dyn, InferenceBackend, f32>::zeros(vec![2, 3])?;
    tensor.to_device(&Default::default())
}

pub fn built_in_operation_contract<B>()
where
    B: TensorBackend<f32> + StorageBackend + Execute<op::Zeros>,
{
}

pub fn external_device_identity_contract() -> DeviceId {
    let key = DeviceKey::new("acme", "npu", 1);
    let device = DeviceId::external(key, 7);
    assert_eq!(device.kind().external_key(), Some(key));
    assert_eq!(device.ordinal(), 7);
    device
}

#[cfg(test)]
mod macro_tests {
    extern crate incin as fire;

    use super::*;
    use fire::backend_authoring::operations::CanonicalOperation;
    use fire::backend_authoring::{
        DescriptorError, ExecutionContext, LogicalTensorMeta, OperationIdentity, UnsupportedReason,
        execute, execute_with_payload,
    };
    use std::sync::Mutex;

    #[derive(Default)]
    struct Probe<D> {
        device: std::marker::PhantomData<D>,
        expected: Mutex<Option<(usize, usize, usize, usize)>>,
        calls: std::sync::atomic::AtomicUsize,
        fail: bool,
    }

    impl StorageBackend for Probe<Cpu> {
        const BACKEND_NAME: &'static str = "probe";
        type Storage<K: DType> = TensorMeta;
        type Device = Cpu;

        fn metadata<K: DType>(storage: &TensorMeta) -> &TensorMeta {
            storage
        }
    }

    #[derive(Clone, Debug)]
    struct PairOp;

    impl Operation for PairOp {
        type Attributes = CompanyAttributes;
        const KEY: OperationKey = OperationKey {
            namespace: std::borrow::Cow::Borrowed("company.example"),
            name: std::borrow::Cow::Borrowed("pair"),
            version: 2,
        };

        fn infer_outputs(
            attributes: &CompanyAttributes,
            _: &[LogicalTensorMeta],
        ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
            Ok(vec![
                LogicalTensorMeta {
                    shape: Some(attributes.shape.clone()),
                    dtype: Some(DTypeId::F32.descriptor()),
                    device: Some(DeviceId::cpu()),
                },
                LogicalTensorMeta {
                    shape: Some(ShapeBuf::from_slice(&[1])),
                    dtype: Some(DTypeId::F32.descriptor()),
                    device: Some(DeviceId::cpu()),
                },
            ])
        }
    }

    fire::backend_authoring::declare_executors! {
        for Probe<Cpu> {
            op::Zeros => f64 = handlers::scalar;
            op::Ones => Vec<f64> = handlers::vector;
            PairOp => (ShapeBuf, ShapeBuf) = handlers::pair;
        }
    }

    fire::backend_authoring::declare_capabilities! {
        for Probe<Cpu> {
            op::Zeros => handlers::support;
            op::Ones => handlers::support;
        }
    }

    mod handlers {
        use super::*;

        pub(super) fn scalar(
            backend: &Probe<Cpu>,
            request: ExecutionRequest<'_, op::Zeros, Probe<Cpu>>,
        ) -> Result<f64, BackendError> {
            backend
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert!(std::ptr::eq(backend, request.context.backend()));
            if let Some(expected) = *backend.expected.lock().unwrap() {
                assert_eq!(request.operation as *const _ as usize, expected.0);
                assert_eq!(request.context as *const _ as usize, expected.1);
                assert_eq!(request.inputs.as_ptr() as usize, expected.2);
                assert_eq!(request.payload.unwrap().as_ptr() as usize, expected.3);
            }
            assert_eq!(request.operation.descriptor().attributes(), &creation());
            if backend.fail || request.payload == Some(&[0xff][..]) {
                return Err(failure());
            }
            Ok(17.0)
        }

        pub(super) fn vector(
            _: &Probe<Cpu>,
            request: ExecutionRequest<'_, op::Ones, Probe<Cpu>>,
        ) -> Result<Vec<f64>, BackendError> {
            assert_eq!(request.operation.descriptor().attributes(), &creation());
            Ok(vec![1.0, 2.0])
        }

        pub(super) fn pair(
            _: &Probe<Cpu>,
            request: ExecutionRequest<'_, PairOp, Probe<Cpu>>,
        ) -> Result<(ShapeBuf, ShapeBuf), BackendError> {
            let descriptor = request.operation.descriptor();
            assert_eq!(
                descriptor.identity(),
                &OperationIdentity::Custom(PairOp::KEY)
            );
            assert_eq!(descriptor.outputs().len(), 2);
            assert_eq!(request.payload, Some(&[4, 8][..]));
            assert!(request.context.training());
            Ok((
                descriptor.outputs()[0].shape.clone().unwrap(),
                descriptor.outputs()[1].shape.clone().unwrap(),
            ))
        }

        pub(super) fn support(_: &Probe<Cpu>, query: &CapabilityQuery) -> SupportLevel {
            assert_eq!(query.dtype, DTypeId::F32.descriptor());
            assert_eq!(query.rank, 2);
            if query.training {
                SupportLevel::Fallback
            } else {
                SupportLevel::Native
            }
        }
    }

    fn creation() -> CreationAttributes {
        CreationAttributes {
            shape: vec![2, 3],
            dtype: DTypeId::F32.descriptor(),
            device: DeviceId::cpu(),
        }
    }

    fn failure() -> BackendError {
        BackendError::unsupported(
            "original-handler",
            UnsupportedReason::MissingDeviceFeature {
                feature: "probe-kernel",
            },
        )
    }

    #[test]
    fn exact_results_include_a_true_multi_output_custom_descriptor() {
        let context = ExecutionContext::new(Probe::<Cpu>::default());
        let scalar: f64 = execute::<op::Zeros, _>(&context, creation(), &[]).unwrap();
        let vector: Vec<f64> = execute::<op::Ones, _>(&context, creation(), &[]).unwrap();
        assert_eq!(scalar, 17.0);
        assert_eq!(vector, vec![1.0, 2.0]);
        let pair: (ShapeBuf, ShapeBuf) = execute_with_payload::<PairOp, _>(
            &context.with_training(true),
            CompanyAttributes {
                shape: ShapeBuf::from_slice(&[2, 3]),
            },
            &[],
            Some(&[4, 8]),
        )
        .unwrap();
        assert_eq!(
            pair,
            (ShapeBuf::from_slice(&[2, 3]), ShapeBuf::from_slice(&[1]))
        );
        assert_eq!(custom_backend_contract(), ShapeBuf::from_slice(&[2, 3]));
        assert_eq!(
            custom_backend_runs_builtin_operation(),
            ShapeBuf::from_slice(&[2, 3])
        );
        assert_eq!(
            inference_only_backend_runs_builtin_operation(),
            ShapeBuf::from_slice(&[2, 3])
        );
        inference_only_backend_can_transfer_tensor().unwrap();
    }

    #[test]
    fn request_references_payload_and_errors_are_not_rebuilt() {
        let context = ExecutionContext::new(Probe::<Cpu>::default()).with_training(true);
        let operation = Descriptor::<op::Zeros>::infer_runtime(creation(), vec![]).unwrap();
        let inputs = [];
        let payload = [4, 8];
        *context.backend.expected.lock().unwrap() = Some((
            &operation as *const _ as usize,
            &context as *const _ as usize,
            inputs.as_ptr() as usize,
            payload.as_ptr() as usize,
        ));
        assert_eq!(
            context
                .backend
                .execute(ExecutionRequest {
                    operation: &operation,
                    inputs: &inputs,
                    context: &context,
                    payload: Some(&payload),
                })
                .unwrap(),
            17.0
        );
        *context.backend.expected.lock().unwrap() = None;
        let request = ExecutionRequest {
            operation: &operation,
            inputs: &inputs,
            context: &context,
            payload: Some(&[0xff]),
        };
        assert_eq!(context.backend.execute(request).unwrap_err(), failure());
        let mut context = context.with_training(false);
        context.backend.fail = true;
        let error: fire::Error = execute::<op::Zeros, _>(&context, creation(), &[])
            .unwrap_err()
            .into();
        match error {
            fire::Error::Backend(error) => assert_eq!(error, failure()),
            other => panic!("expected unchanged backend error, got {other:?}"),
        }
    }

    #[test]
    fn capability_queries_keep_identity_and_policy_refusals() {
        let context = ExecutionContext::new(Probe::<Cpu>::default()).with_training(true);
        let error: fire::Error = execute::<op::Zeros, _>(&context, creation(), &[])
            .unwrap_err()
            .into();
        match error {
            fire::Error::Policy(refusal) => {
                assert_eq!(refusal.operation, OperationIdentity::Builtin(op::Zeros::ID));
                assert_eq!(refusal.support, SupportLevel::Fallback);
                assert_eq!(refusal.fallback, context.fallback());
            }
            other => panic!("expected policy refusal, got {other:?}"),
        }
        assert_eq!(
            context
                .backend
                .calls
                .load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        let mut query = CapabilityQuery {
            operation: OperationIdentity::Builtin(op::Add::ID),
            dtype: DTypeId::F32.descriptor(),
            layout: CompanyBackend::metadata::<f32>(&ShapeBuf::from_slice(&[2, 3])).layout,
            rank: 2,
            training: false,
            math_mode: context.math_mode(),
        };
        assert_eq!(
            context.backend.support(&query),
            SupportLevel::Unsupported(UnsupportedReason::Operation {
                operation: op::Add::ID
            },)
        );
        query.operation = OperationIdentity::Custom(PairOp::KEY);
        assert_eq!(
            context.backend.support(&query),
            SupportLevel::Unsupported(UnsupportedReason::CustomOperation {
                operation: PairOp::KEY
            },)
        );
    }

    #[derive(Debug, Clone)]
    struct InspectInput;

    impl Operation for InspectInput {
        type Attributes = CompanyAttributes;
        const KEY: OperationKey = OperationKey {
            namespace: std::borrow::Cow::Borrowed("company.example"),
            name: std::borrow::Cow::Borrowed("inspect-input"),
            version: 1,
        };

        fn infer_outputs(
            _: &CompanyAttributes,
            inputs: &[LogicalTensorMeta],
        ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
            assert_eq!(inputs.len(), 1);
            Ok(inputs.to_vec())
        }
    }

    fire::declare_executors! {
        for InferenceBackend {
            InspectInput => ShapeBuf = inspect_input;
        }
    }

    fn inspect_input(
        backend: &InferenceBackend,
        request: ExecutionRequest<'_, InspectInput, InferenceBackend>,
    ) -> Result<ShapeBuf, BackendError> {
        assert!(std::ptr::eq(backend, request.context.backend()));
        assert_eq!(request.inputs.len(), 1);
        let storage = request.inputs[0].downcast_ref::<ShapeBuf>().unwrap();
        assert_eq!(
            storage as *const _ as usize,
            request.operation.descriptor().attributes().shape[0]
        );
        assert_eq!(request.inputs[0].metadata().shape(), storage);
        assert_eq!(
            request.operation.descriptor().inputs()[0].shape.as_ref(),
            Some(storage)
        );
        Ok(storage.clone())
    }

    #[test]
    fn input_storage_and_checked_metadata_reach_the_handler() {
        let tensor = fire::Tensor::<fire::Dyn, InferenceBackend, f32>::zeros(vec![2, 3]).unwrap();
        let output = tensor
            .apply_op::<InspectInput>(CompanyAttributes {
                shape: ShapeBuf::from_slice(&[tensor.inner() as *const _ as usize]),
            })
            .unwrap();
        assert_eq!(output.inner(), tensor.inner());
    }
}
