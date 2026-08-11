//! `EXE-008`: the descriptor execution contract on a third-party backend.
//!
//! Candle is not an Incin-authored backend and its tensor type carries no
//! `TensorMeta`. These cases prove a foreign backend can still join the sealed
//! descriptor contract through a storage wrapper, which is the property a
//! backend author has to be able to reproduce.
#![cfg(feature = "external-candle")]

use incin_backends::external::candle::{CandleBackend, CandleStorage};
use incin_core::backend_authoring::{Execute, ExecutionRequest};
use incin_core::exec::{
    Capabilities, CapabilityQuery, ExecutionContext, LayoutClass, MatMulRule, MatMulSpec, MathMode,
    ReshapeRule, ReshapeSpec, ShapeRule, SupportLevel, TensorHandle, Validated,
};
use incin_core::prelude::{BackendError, Cpu, DTypeId, Dyn, Local, OperationKind, Shape};
use incin_core::typenum::{U2, U3, U4, U6};

type TestBackend = CandleBackend<Cpu>;

fn field<S: Shape>(dims: &[usize]) -> ShapeBuf {
    S::try_from_dims(dims).expect("test dimensions must match the shape type")
}

fn lower(lhs: &[usize], rhs: &[usize]) -> Validated<MatMulSpec> {
    <MatMulRule as ShapeRule<(Dyn, Dyn)>>::lower(&(field::<Dyn>(lhs), field::<Dyn>(rhs)), ())
        .expect("test operands must be valid matmul shapes")
}

fn storage(shape: &[usize], values: &[f32]) -> CandleStorage {
    let tensor = candle_core::Tensor::from_slice(values, shape, &candle_core::Device::Cpu)
        .expect("test buffer must match its shape");
    CandleStorage::try_new(tensor).expect("a candle tensor must yield checked metadata")
}

fn execute(
    validated: &Validated<MatMulSpec>,
    lhs: &CandleStorage,
    rhs: &CandleStorage,
) -> Result<CandleStorage, BackendError> {
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(lhs),
        TensorHandle::from_storage::<TestBackend, f32, Local>(rhs),
    ];
    context.backend().execute(ExecutionRequest {
        operation: validated,
        inputs: &inputs,
        context: &context,
    })
}

#[test]
fn a_foreign_tensor_reports_its_own_checked_metadata() {
    let lhs = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let meta = lhs.metadata();

    assert_eq!(meta.shape().dims(), &[2, 3]);
    assert_eq!(meta.strides().strides(), &[3, 1]);
    assert_eq!(meta.dtype(), DTypeId::F32);
    assert_eq!(meta.layout(), LayoutClass::Contiguous);
}

#[test]
fn rank2_descriptor_execution_produces_the_declared_output() {
    let lhs = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let rhs = storage(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let validated = lower(&[2, 3], &[3, 2]);

    let output = execute(&validated, &lhs, &rhs).expect("a valid descriptor must execute");

    assert_eq!(output.metadata().shape().dims(), &[2, 2]);
    let values = output
        .tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(values, vec![58., 64., 139., 154.]);
}

#[test]
fn batched_descriptor_execution_broadcasts_the_batch_axis() {
    let lhs = storage(
        &[2, 2, 3],
        &[1., 2., 3., 4., 5., 6., 6., 5., 4., 3., 2., 1.],
    );
    let rhs = storage(&[1, 3, 2], &[1., 0., 0., 1., 1., 1.]);
    let validated = lower(&[2, 2, 3], &[1, 3, 2]);

    let output = execute(&validated, &lhs, &rhs).expect("a batched descriptor must execute");

    assert_eq!(output.metadata().shape().dims(), &[2, 2, 2]);
}

#[test]
fn the_binder_requires_exactly_two_inputs() {
    let lhs = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let validated = lower(&[2, 3], &[3, 2]);
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(&lhs)];

    let error = context
        .backend()
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &inputs,
            context: &context,
        })
        .expect_err("a one-operand matmul request must not execute");

    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::MatMul,
            reason: "matmul expects exactly two tensor inputs"
        }
    ));
}

#[test]
fn capabilities_refuse_a_dtype_candle_cannot_represent() {
    // Candle has no Q8_0 representation, so the adapter must report that rather
    // than accept a query it would fail to route.
    let backend = TestBackend::default();
    let query = CapabilityQuery {
        operation: OperationKind::MatMul,
        dtype: DTypeId::Q8_0,
        layout: LayoutClass::Contiguous,
        rank: 2,
        training: false,
        math_mode: MathMode::Precise,
    };

    assert!(matches!(
        backend.support(&query),
        SupportLevel::Unsupported(_)
    ));
}

fn lower_reshape_2x6_to_3x4() -> Validated<ReshapeSpec> {
    <ReshapeRule as ShapeRule<((U2, U6), (U3, U4))>>::lower(
        &(field::<(U2, U6)>(&[2, 6]), field::<(U3, U4)>(&[3, 4])),
        (),
    )
    .expect("12 elements either way")
}

fn execute_reshape(
    validated: &Validated<ReshapeSpec>,
    input: &CandleStorage,
) -> Result<CandleStorage, BackendError> {
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(input)];
    context.backend().execute(ExecutionRequest {
        operation: validated,
        inputs: &inputs,
        context: &context,
    })
}

#[test]
fn reshape_descriptor_execution_produces_the_declared_output() {
    let input = storage(
        &[2, 6],
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
    );

    let output = execute_reshape(&lower_reshape_2x6_to_3x4(), &input)
        .expect("a valid reshape descriptor must execute");

    assert_eq!(output.metadata().shape().dims(), &[3, 4]);
    let values = output
        .tensor()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(
        values,
        (1..=12).map(|value| value as f32).collect::<Vec<_>>()
    );
}

#[test]
fn the_reshape_binder_rejects_an_operand_that_disagrees_with_the_descriptor() {
    let input = storage(&[4, 3], &[1.; 12]);

    let error = execute_reshape(&lower_reshape_2x6_to_3x4(), &input)
        .expect_err("a descriptor lowered for other extents must not execute");

    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::Reshape,
            reason: "reshape input metadata does not match the validated descriptor"
        }
    ));
}

#[test]
fn capabilities_now_answer_for_the_operations_the_adapter_routes() {
    // The registry answer and the `Execute` implementations have to agree: an
    // operation claimed here that no executor routes is exactly the unearned
    // claim `EXE-005` exists to prevent.
    let backend = TestBackend::default();
    let query = |operation| CapabilityQuery {
        operation,
        dtype: DTypeId::F32,
        layout: LayoutClass::Contiguous,
        rank: 2,
        training: false,
        math_mode: MathMode::Precise,
    };

    assert!(matches!(
        backend.support(&query(OperationKind::Reshape)),
        SupportLevel::Native
    ));
    assert!(matches!(
        backend.support(&query(OperationKind::Conv2d)),
        SupportLevel::Unsupported(_)
    ));
}
