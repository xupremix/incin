//! Parity, autograd, and descriptor execution tests for the Metal compute backend on Apple Silicon / Linux fallback.

#![cfg(feature = "metal")]

use incin_backends::metal::{MetalBackendImpl, MetalStorage, MetalStorageMode};
use incin_core::exec::{
    ExecutionContext, MatMulRule, MatMulSpec, ReshapeRule, ReshapeSpec, ShapeRule, TapeStorage,
    TensorHandle, Validated,
};
use incin_core::prelude::{
    Backend, DTypeId, DeviceId, Dyn, Execute, ExecutionRequest, FloatOps, Local, NumericOps,
    ReductionOps,
};
use incin_core::typenum::{U2, U3, U4, U6};

type TestBackend = MetalBackendImpl<f32, incin_core::prelude::Metal>;

fn field<S: incin_core::shapes::Shape>(dims: &[usize]) -> S::Field {
    S::from_dyn(dims).expect("test dimensions must match shape")
}

fn lower_matmul(lhs: &[usize], rhs: &[usize]) -> Validated<MatMulSpec> {
    <MatMulRule as ShapeRule<(Dyn, Dyn)>>::lower(&(field::<Dyn>(lhs), field::<Dyn>(rhs)), ())
        .expect("test operands must be valid matmul shapes")
}

fn lower_reshape() -> Validated<ReshapeSpec> {
    <ReshapeRule as ShapeRule<((U2, U6), (U3, U4))>>::lower(
        &(field::<(U2, U6)>(&[2, 6]), field::<(U3, U4)>(&[3, 4])),
        (),
    )
    .expect("12 elements either way")
}

fn create_storage(shape: &[usize], values: &[f32]) -> MetalStorage {
    TestBackend::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32,
        &DeviceId::metal(0),
    )
    .expect("test buffer must match its shape")
}

fn read_storage(storage: &MetalStorage) -> Vec<f32> {
    let bytes = TestBackend::to_bytes::<f32>(storage).expect("readback must succeed");
    bytemuck::cast_slice(&bytes).to_vec()
}

#[test]
fn test_metal_storage_tagging_and_tape_storage_invariants() {
    let s1 = create_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let s2 = create_storage(&[2, 2], &[0.5, 0.5, 0.5, 0.5]);

    assert_ne!(s1.id(), s2.id(), "TensorId must be unique per MetalStorage");
    assert_eq!(s1.shape(), &[2, 2]);
    assert!(!s1.has_non_finite().unwrap());

    let acc = s1.accumulate(&s2).expect("accumulate should succeed");
    assert_eq!(read_storage(&acc), vec![1.5, 2.5, 3.5, 4.5]);

    let ones = s1.ones_like().expect("ones_like should succeed");
    assert_eq!(read_storage(&ones), vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_metal_private_storage_host_readback_error() {
    let meta = incin_core::exec::TensorMeta::contiguous(
        incin_core::shapes::ShapeBuf::from_slice(&[2, 2]),
        DTypeId::F32,
        DeviceId::metal(0),
        MetalStorage::alignment(),
        4,
    )
    .unwrap();

    let private_storage = MetalStorage::from_bytes(
        bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec(),
        meta,
        MetalStorageMode::Private,
        0,
    )
    .unwrap();

    assert!(private_storage.as_bytes().is_err());
}

#[test]
fn test_metal_forward_ops() {
    let a = create_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b = create_storage(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);

    let add = <TestBackend as NumericOps<TestBackend>>::add::<f32>(&a, &b).unwrap();
    assert_eq!(read_storage(&add), vec![6.0, 8.0, 10.0, 12.0]);

    let sub = <TestBackend as NumericOps<TestBackend>>::sub::<f32>(&b, &a).unwrap();
    assert_eq!(read_storage(&sub), vec![4.0, 4.0, 4.0, 4.0]);

    let mul = <TestBackend as NumericOps<TestBackend>>::mul::<f32>(&a, &b).unwrap();
    assert_eq!(read_storage(&mul), vec![5.0, 12.0, 21.0, 32.0]);

    let div = <TestBackend as NumericOps<TestBackend>>::div::<f32>(&b, &a).unwrap();
    assert_eq!(read_storage(&div), vec![5.0, 3.0, 2.3333333, 2.0]);

    let relu = <TestBackend as FloatOps<TestBackend>>::relu::<f32>(&create_storage(
        &[4],
        &[-1.0, 0.0, 2.0, -3.0],
    ))
    .unwrap();
    assert_eq!(read_storage(&relu), vec![0.0, 0.0, 2.0, 0.0]);

    let sum = <TestBackend as ReductionOps<TestBackend>>::sum_dim::<f32>(&a, 1).unwrap();
    assert_eq!(read_storage(&sum), vec![3.0, 7.0]);
}

#[test]
fn test_metal_autograd_backward() {
    let x = create_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let w = create_storage(&[2, 2], &[0.5, -1.0, 2.0, 1.5]);

    let xw = <TestBackend as NumericOps<TestBackend>>::mul::<f32>(&x, &w).unwrap();
    let loss = <TestBackend as ReductionOps<TestBackend>>::sum_all::<f32>(&xw).unwrap();

    let grads = TestBackend::backward::<f32>(&loss).expect("backward execution should succeed");
    let gx = TestBackend::get_grad::<f32>(&x, &grads)
        .expect("get_grad for x")
        .unwrap();
    let gw = TestBackend::get_grad::<f32>(&w, &grads)
        .expect("get_grad for w")
        .unwrap();

    assert_eq!(read_storage(&gx), vec![0.5, -1.0, 2.0, 1.5]);
    assert_eq!(read_storage(&gw), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_metal_descriptor_execution() {
    let lhs = create_storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let rhs = create_storage(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let validated = lower_matmul(&[2, 3], &[3, 2]);

    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&lhs),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&rhs),
    ];
    let descriptor_out = context
        .backend()
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &inputs,
            context: &context,
        })
        .expect("descriptor matmul must execute");

    assert_eq!(descriptor_out.shape(), &[2, 2]);
    assert_eq!(read_storage(&descriptor_out), vec![58., 64., 139., 154.]);

    let reshape_in = create_storage(
        &[2, 6],
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
    );
    let reshape_val = lower_reshape();
    let reshape_inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(
        &reshape_in,
    )];
    let reshape_out = context
        .backend()
        .execute(ExecutionRequest {
            operation: &reshape_val,
            inputs: &reshape_inputs,
            context: &context,
        })
        .expect("descriptor reshape must execute");

    assert_eq!(reshape_out.shape(), &[3, 4]);
    assert_eq!(read_storage(&reshape_out), read_storage(&reshape_in));
}
