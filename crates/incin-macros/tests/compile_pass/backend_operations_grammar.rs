//! `define_backend_operations!` accepts every entry form it declares: builtin
//! operations with storage, scalar, vector, and tuple outputs, plus a custom
//! operation that stays executor-only. Each shape executes through dispatch.
use ::incin::backend_authoring::{
    Alignment, ExecutionContext, ExecutionRequest, LogicalTensorMeta, Operation, OperationKey,
    ShapeBuf, StorageBackend, SupportLevel, TensorMeta, execute, execute_with_payload,
    operations::{CreationAttributes, FullAttributes, NoAttributes, op},
};
use ::incin::prelude::{BackendError, Cpu, DType, DTypeId, DeviceId};
use ::incin_macros::define_backend_operations;
use ::incin::backend_authoring::{CapabilityQuery, DescriptorError};

#[derive(Default)]
struct GrammarBackend;

impl StorageBackend for GrammarBackend {
    const BACKEND_NAME: &'static str = "grammar-fixture";
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
struct GrammarCustom;

impl Operation for GrammarCustom {
    type Attributes = NoAttributes;
    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("grammar.fixture"),
        name: std::borrow::Cow::Borrowed("custom"),
        version: 1,
    };

    fn infer_outputs(
        _: &NoAttributes,
        _: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(vec![LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&[2, 3])),
            dtype: Some(DTypeId::F32.descriptor()),
            device: Some(DeviceId::cpu()),
        }])
    }
}

fn grammar_zeros(
    _: &GrammarBackend,
    request: ExecutionRequest<'_, op::Zeros, GrammarBackend>,
) -> Result<ShapeBuf, BackendError> {
    Ok(ShapeBuf::from_slice(
        &request.operation.descriptor().attributes().shape,
    ))
}

fn grammar_ones(
    _: &GrammarBackend,
    _: ExecutionRequest<'_, op::Ones, GrammarBackend>,
) -> Result<f64, BackendError> {
    Ok(7.0)
}

fn grammar_full(
    _: &GrammarBackend,
    request: ExecutionRequest<'_, op::Full, GrammarBackend>,
) -> Result<Vec<f64>, BackendError> {
    Ok(vec![
        request.operation.descriptor().attributes().value;
        2
    ])
}

fn grammar_pair(
    _: &GrammarBackend,
    request: ExecutionRequest<'_, GrammarCustom, GrammarBackend>,
) -> Result<(ShapeBuf, f64), BackendError> {
    Ok((
        request.operation.descriptor().outputs()[0]
            .shape
            .clone()
            .unwrap(),
        1.0,
    ))
}

fn grammar_support(_: &GrammarBackend, _: &CapabilityQuery) -> SupportLevel {
    SupportLevel::Native
}

define_backend_operations! {
    for GrammarBackend {
        op::Zeros => ShapeBuf = grammar_zeros;
        op::Ones => f64 = grammar_ones;
        op::Full => Vec<f64> = grammar_full;
        GrammarCustom => (ShapeBuf, f64) = grammar_pair;
    }
    capabilities for GrammarBackend {
        op::Zeros => grammar_support;
        op::Ones => grammar_support;
        op::Full => grammar_support;
    }
}

fn creation() -> CreationAttributes {
    CreationAttributes {
        shape: vec![2, 3],
        dtype: DTypeId::F32.descriptor(),
        device: DeviceId::cpu(),
    }
}

fn main() {
    let context = ExecutionContext::new(GrammarBackend);
    let storage: ShapeBuf = execute::<op::Zeros, _>(&context, creation(), &[]).unwrap();
    assert_eq!(storage, ShapeBuf::from_slice(&[2, 3]));
    let scalar: f64 = execute::<op::Ones, _>(&context, creation(), &[]).unwrap();
    assert_eq!(scalar, 7.0);
    let vector: Vec<f64> = execute::<op::Full, _>(
        &context,
        FullAttributes {
            shape: vec![2, 3],
            dtype: DTypeId::F32.descriptor(),
            device: DeviceId::cpu(),
            value: 3.0,
        },
        &[],
    )
    .unwrap();
    assert_eq!(vector, vec![3.0; 2]);
    let pair: (ShapeBuf, f64) =
        execute_with_payload::<GrammarCustom, _>(&context, NoAttributes, &[], None).unwrap();
    assert_eq!(pair, (ShapeBuf::from_slice(&[2, 3]), 1.0));
}
