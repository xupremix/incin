//! Small active proof that the CPU backend executes through canonical descriptors.

#![cfg(feature = "cpu")]

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_core::backend_authoring::StorageBackend;
use incin_core::exec::catalog::{NoAttributes, op};
use incin_core::exec::dispatch;
use incin_core::exec::{ExecutionContext, TensorHandle};

type B = CpuBackendImpl;

fn storage(values: Vec<f32>) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values), vec![2, 2]).unwrap()
}

fn integer_storage(values: Vec<i64>) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::I64(values), vec![2, 2]).unwrap()
}

#[test]
fn add_uses_the_descriptor_executor_and_preserves_shape() {
    let lhs = storage(vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = storage(vec![10.0, 20.0, 30.0, 40.0]);
    let context = ExecutionContext::new(B::new());
    let output = dispatch::execute::<op::Add, _>(
        &context,
        NoAttributes,
        &[
            TensorHandle::from_storage::<B, f32, _>(&lhs),
            TensorHandle::from_storage::<B, f32, _>(&rhs),
        ],
    )
    .unwrap();

    assert_eq!(B::shape::<f32>(&output).dims(), &[2, 2]);
    assert_eq!(output.get(&[0, 0]), 11.0);
    assert_eq!(output.get(&[1, 1]), 44.0);
}

#[test]
fn descriptor_validation_rejects_a_wrong_dtype_before_execution() {
    let lhs = storage(vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = integer_storage(vec![10, 20, 30, 40]);
    let context = ExecutionContext::new(B::new());
    let result = dispatch::execute::<op::Add, _>(
        &context,
        NoAttributes,
        &[
            TensorHandle::from_storage::<B, f32, _>(&lhs),
            TensorHandle::from_storage::<B, f32, _>(&rhs),
        ],
    );

    assert!(result.is_err());
}
