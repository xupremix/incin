//! Downstream-style backend authoring through the public proc macro.
//!
//! This fixture uses only the public surface: `::incin::backend_authoring`,
//! `::incin::prelude`, and `::incin_macros::define_backend_operations`. No
//! `incin_core` paths, no workspace-private macros.

use ::incin::backend_authoring::{
    Alignment, Capabilities, CapabilityQuery, DescriptorError, Execute, ExecutionContext,
    ExecutionRequest, LogicalTensorMeta, Operation, OperationIdentity, OperationKey, ShapeBuf,
    StorageBackend, SupportLevel, TensorMeta, execute, execute_with_payload,
    operations::{
        CanonicalOperation, CreationAttributes, Descriptor, FullAttributes, NoAttributes, op,
    },
};
use ::incin::prelude::{BackendError, Cpu, DType, DTypeId, DeviceId};
use ::incin_macros::define_backend_operations;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[derive(Default)]
struct ProcBackend {
    expected: Mutex<Option<(usize, usize, usize, usize)>>,
    calls: AtomicUsize,
    fail: AtomicBool,
}

impl StorageBackend for ProcBackend {
    const BACKEND_NAME: &'static str = "proc-fixture";
    type Storage<K: DType> = ShapeBuf;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &ShapeBuf) -> &TensorMeta {
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

#[derive(Debug, Clone)]
struct PairOp;

impl Operation for PairOp {
    type Attributes = NoAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("proc.fixture"),
        name: std::borrow::Cow::Borrowed("pair"),
        version: 1,
    };

    fn infer_outputs(
        _: &NoAttributes,
        _: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(vec![
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[2, 3])),
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

#[derive(Debug, Clone)]
struct MultiOp;

impl Operation for MultiOp {
    type Attributes = NoAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("proc.fixture"),
        name: std::borrow::Cow::Borrowed("multi"),
        version: 1,
    };

    fn infer_outputs(
        _: &NoAttributes,
        _: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(vec![
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[2])),
                dtype: Some(DTypeId::F32.descriptor()),
                device: Some(DeviceId::cpu()),
            },
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[3])),
                dtype: Some(DTypeId::F32.descriptor()),
                device: Some(DeviceId::cpu()),
            },
        ])
    }
}

fn zeros_storage(
    backend: &ProcBackend,
    request: ExecutionRequest<'_, op::Zeros, ProcBackend>,
) -> Result<ShapeBuf, BackendError> {
    backend.calls.fetch_add(1, Ordering::SeqCst);
    assert!(std::ptr::eq(backend, request.context.backend()));
    if let Some(expected) = *backend.expected.lock().unwrap() {
        assert_eq!(request.operation as *const _ as usize, expected.0);
        assert_eq!(request.context as *const _ as usize, expected.1);
        assert_eq!(request.inputs.as_ptr() as usize, expected.2);
        assert_eq!(request.payload.unwrap().as_ptr() as usize, expected.3);
    }
    assert_eq!(
        request.operation.descriptor().attributes(),
        &creation(),
        "the handler reads the validated descriptor instead of re-deriving metadata"
    );
    if backend.fail.load(Ordering::SeqCst) || request.payload == Some(&[0xff][..]) {
        return Err(failure());
    }
    Ok(ShapeBuf::from_slice(
        &request.operation.descriptor().attributes().shape,
    ))
}

fn ones_scalar(
    _: &ProcBackend,
    request: ExecutionRequest<'_, op::Ones, ProcBackend>,
) -> Result<f64, BackendError> {
    assert_eq!(request.operation.descriptor().attributes(), &creation());
    Ok(17.0)
}

fn full_vector(
    _: &ProcBackend,
    request: ExecutionRequest<'_, op::Full, ProcBackend>,
) -> Result<Vec<f64>, BackendError> {
    let attributes = request.operation.descriptor().attributes();
    assert_eq!(attributes.value, 2.5);
    Ok(vec![attributes.value; 3])
}

fn pair_tuple(
    _: &ProcBackend,
    request: ExecutionRequest<'_, PairOp, ProcBackend>,
) -> Result<(ShapeBuf, f64), BackendError> {
    let descriptor = request.operation.descriptor();
    assert_eq!(
        descriptor.identity(),
        &OperationIdentity::Custom(PairOp::KEY)
    );
    assert_eq!(descriptor.outputs().len(), 2);
    assert_eq!(request.payload, Some(&[4, 8][..]));
    Ok((
        descriptor.outputs()[0].shape.clone().unwrap(),
        descriptor.outputs()[1]
            .shape
            .clone()
            .unwrap()
            .numel()
            .unwrap() as f64,
    ))
}

fn multi_vec(
    _: &ProcBackend,
    request: ExecutionRequest<'_, MultiOp, ProcBackend>,
) -> Result<Vec<ShapeBuf>, BackendError> {
    Ok(request
        .operation
        .descriptor()
        .outputs()
        .iter()
        .map(|output| output.shape.clone().unwrap())
        .collect())
}

fn proc_support(_: &ProcBackend, _: &CapabilityQuery) -> SupportLevel {
    SupportLevel::Native
}

define_backend_operations! {
    for ProcBackend {
        op::Zeros => ShapeBuf = zeros_storage;
        op::Ones => f64 = ones_scalar;
        op::Full => Vec<f64> = full_vector;
        PairOp => (ShapeBuf, f64) = pair_tuple;
        MultiOp => Vec<ShapeBuf> = multi_vec;
    }
    capabilities for ProcBackend {
        op::Zeros => proc_support;
        op::Ones => proc_support;
        op::Full => proc_support;
    }
}

fn creation() -> CreationAttributes {
    CreationAttributes {
        shape: vec![2, 3],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
    }
}

fn full_creation() -> FullAttributes {
    FullAttributes {
        shape: vec![2, 3],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
        value: 2.5,
    }
}

fn failure() -> BackendError {
    BackendError::unsupported(
        "original-handler",
        ::incin::backend_authoring::UnsupportedReason::MissingDeviceFeature {
            feature: "proc-kernel",
        },
    )
}

#[test]
fn every_declared_output_shape_executes_through_dispatch() {
    let context = ExecutionContext::new(ProcBackend::default());
    let storage: ShapeBuf = execute::<op::Zeros, _>(&context, creation(), &[]).unwrap();
    assert_eq!(storage, ShapeBuf::from_slice(&[2, 3]));
    let scalar: f64 = execute::<op::Ones, _>(&context, creation(), &[]).unwrap();
    assert_eq!(scalar, 17.0);
    let vector: Vec<f64> = execute::<op::Full, _>(&context, full_creation(), &[]).unwrap();
    assert_eq!(vector, vec![2.5; 3]);
    let pair: (ShapeBuf, f64) =
        execute_with_payload::<PairOp, _>(&context, NoAttributes, &[], Some(&[4, 8])).unwrap();
    assert_eq!(pair, (ShapeBuf::from_slice(&[2, 3]), 1.0));
    let multi: Vec<ShapeBuf> = execute::<MultiOp, _>(&context, NoAttributes, &[]).unwrap();
    assert_eq!(
        multi,
        vec![ShapeBuf::from_slice(&[2]), ShapeBuf::from_slice(&[3])]
    );
}

#[test]
fn validated_request_reaches_the_handler_unmodified() {
    let mut context = ExecutionContext::new(ProcBackend::default()).with_training(true);
    let operation = Descriptor::<op::Zeros>::infer_runtime(creation(), vec![]).unwrap();
    let inputs = [];
    let payload = [4, 8];
    *context.backend_mut().expected.lock().unwrap() = Some((
        &operation as *const _ as usize,
        &context as *const _ as usize,
        inputs.as_ptr() as usize,
        payload.as_ptr() as usize,
    ));
    assert_eq!(
        context
            .backend()
            .execute(ExecutionRequest {
                operation: &operation,
                inputs: &inputs,
                context: &context,
                payload: Some(&payload),
            })
            .unwrap(),
        ShapeBuf::from_slice(&[2, 3])
    );
    assert_eq!(context.backend().calls.load(Ordering::SeqCst), 1);
}

#[test]
fn handler_errors_reach_the_caller_unrebuilt() {
    let mut context = ExecutionContext::new(ProcBackend::default());
    let operation = Descriptor::<op::Zeros>::infer_runtime(creation(), vec![]).unwrap();
    let inputs = [];
    let request = ExecutionRequest {
        operation: &operation,
        inputs: &inputs,
        context: &context,
        payload: Some(&[0xff]),
    };
    assert_eq!(context.backend().execute(request).unwrap_err(), failure());
    context.backend_mut().fail.store(true, Ordering::SeqCst);
    let error: ::incin::Error = execute::<op::Zeros, _>(&context, creation(), &[])
        .unwrap_err()
        .into();
    match error {
        ::incin::Error::Backend(error) => assert_eq!(error, failure()),
        other => panic!("expected the unchanged backend error, got {other:?}"),
    }
}

#[test]
fn capability_routing_admits_listed_and_refuses_the_rest() {
    let context = ExecutionContext::new(ProcBackend::default());
    let mut query = CapabilityQuery {
        operation: OperationIdentity::Builtin(op::Zeros::ID),
        dtype: DTypeId::F32.descriptor(),
        layout: ProcBackend::metadata::<f32>(&ShapeBuf::from_slice(&[2, 3])).layout,
        rank: 2,
        training: false,
        math_mode: context.math_mode(),
    };
    assert_eq!(context.backend().support(&query), SupportLevel::Native);
    query.operation = OperationIdentity::Builtin(op::Add::ID);
    assert_eq!(
        context.backend().support(&query),
        SupportLevel::Unsupported(::incin::backend_authoring::UnsupportedReason::Operation {
            operation: op::Add::ID
        },)
    );
}

#[test]
fn custom_operations_execute_without_capability_entries() {
    let context = ExecutionContext::new(ProcBackend::default());
    let query = CapabilityQuery {
        operation: OperationIdentity::Custom(PairOp::KEY),
        dtype: DTypeId::F32.descriptor(),
        layout: ProcBackend::metadata::<f32>(&ShapeBuf::from_slice(&[2, 3])).layout,
        rank: 2,
        training: false,
        math_mode: context.math_mode(),
    };
    assert_eq!(
        context.backend().support(&query),
        SupportLevel::Unsupported(
            ::incin::backend_authoring::UnsupportedReason::CustomOperation {
                operation: PairOp::KEY
            },
        )
    );
    let pair: (ShapeBuf, f64) =
        execute_with_payload::<PairOp, _>(&context, NoAttributes, &[], Some(&[4, 8])).unwrap();
    assert_eq!(pair, (ShapeBuf::from_slice(&[2, 3]), 1.0));
}
