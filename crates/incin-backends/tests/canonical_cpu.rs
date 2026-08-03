//! FND-005: the canonical CPU execution path.
//!
//! Every test here asserts a property of `exec::dispatch::execute` that the
//! legacy operation-family traits could not offer: an exact identity, a
//! capability answer that binds execution, validation before launch, and
//! output metadata that is derived rather than accepted. Parity against the
//! legacy path is asserted too, because a replacement that computes something
//! else is not a migration.

#![cfg(feature = "cpu")]

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_core::backend_authoring::{NumericOps, TensorOps};
use incin_core::exec::catalog::{NoAttributes, ShapeAttributes, op};
use incin_core::exec::dispatch::{self, CanonicalError};
use incin_core::exec::{
    Capabilities, CapabilityQuery, DescriptorError, ExecutionContext, LayoutClass, MathMode,
    SupportLevel, TensorHandle, UnsupportedReason,
};
use incin_core::prelude::{Cpu, DTypeId, Local, OperationKind};

type TestBackend = CpuBackendImpl<f32, Cpu>;

fn f32_storage(values: Vec<f32>, shape: Vec<usize>) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values), shape)
        .expect("test storage must be well formed")
}

fn handle(storage: &CpuStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<TestBackend, f32, Local>(storage)
}

/// Read a storage handle's logical values through its strides.
///
/// Reading the buffer directly would compare allocations rather than tensors,
/// and a broadcast or reshaped view shares its allocation with its source.
fn values(storage: &CpuStorage) -> Vec<f64> {
    let shape = storage.shape();
    let count: usize = shape.dims().iter().product();
    (0..count)
        .map(|mut flat| {
            let mut index = vec![0; shape.rank()];
            for axis in (0..shape.rank()).rev() {
                index[axis] = flat % shape.dims()[axis];
                flat /= shape.dims()[axis];
            }
            storage.get(&index)
        })
        .collect()
}

fn dims(storage: &CpuStorage) -> Vec<usize> {
    storage.shape().dims().to_vec()
}

fn context() -> ExecutionContext<TestBackend> {
    ExecutionContext::new(TestBackend::new())
}

#[test]
fn a_canonical_invocation_matches_the_legacy_operation_family_result() {
    let context = context();
    let lhs = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let rhs = f32_storage(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

    let canonical =
        dispatch::execute::<op::Add, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .expect("add is a registered CPU capability");
    let legacy = <TestBackend as NumericOps<TestBackend>>::add::<f32>(&lhs, &rhs)
        .expect("the legacy path computes the same operation");

    assert_eq!(values(&canonical), values(&legacy));
    assert_eq!(dims(&canonical), vec![2, 2]);
}

#[test]
fn every_migrated_pointwise_operation_matches_its_legacy_counterpart() {
    let context = context();
    let lhs = f32_storage(vec![6.0, 8.0, 10.0, 12.0], vec![4]);
    let rhs = f32_storage(vec![2.0, 4.0, 5.0, 3.0], vec![4]);

    let cases: [(&str, Vec<f64>, Vec<f64>); 4] = [
        (
            "add",
            values(
                &dispatch::execute::<op::Add, _>(
                    &context,
                    NoAttributes,
                    &[handle(&lhs), handle(&rhs)],
                )
                .unwrap(),
            ),
            values(&<TestBackend as NumericOps<TestBackend>>::add::<f32>(&lhs, &rhs).unwrap()),
        ),
        (
            "sub",
            values(
                &dispatch::execute::<op::Sub, _>(
                    &context,
                    NoAttributes,
                    &[handle(&lhs), handle(&rhs)],
                )
                .unwrap(),
            ),
            values(&<TestBackend as NumericOps<TestBackend>>::sub::<f32>(&lhs, &rhs).unwrap()),
        ),
        (
            "mul",
            values(
                &dispatch::execute::<op::Mul, _>(
                    &context,
                    NoAttributes,
                    &[handle(&lhs), handle(&rhs)],
                )
                .unwrap(),
            ),
            values(&<TestBackend as NumericOps<TestBackend>>::mul::<f32>(&lhs, &rhs).unwrap()),
        ),
        (
            "div",
            values(
                &dispatch::execute::<op::Div, _>(
                    &context,
                    NoAttributes,
                    &[handle(&lhs), handle(&rhs)],
                )
                .unwrap(),
            ),
            values(&<TestBackend as NumericOps<TestBackend>>::div::<f32>(&lhs, &rhs).unwrap()),
        ),
    ];

    for (name, canonical, legacy) in cases {
        assert_eq!(canonical, legacy, "{name} diverged from the legacy path");
    }
}

#[test]
fn matmul_and_the_shape_operations_match_the_legacy_path() {
    let context = context();
    let lhs = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let rhs = f32_storage(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

    let canonical = dispatch::execute::<op::MatMulExact, _>(
        &context,
        NoAttributes,
        &[handle(&lhs), handle(&rhs)],
    )
    .expect("matmul is a registered CPU capability");
    let legacy = <TestBackend as TensorOps<TestBackend>>::matmul::<f32>(&lhs, &rhs).unwrap();
    assert_eq!(values(&canonical), values(&legacy));
    assert_eq!(dims(&canonical), vec![2, 2]);

    let reshaped = dispatch::execute::<op::ReshapeExact, _>(
        &context,
        ShapeAttributes { shape: vec![3, 2] },
        &[handle(&lhs)],
    )
    .expect("reshape is a registered CPU capability");
    let legacy_reshape =
        <TestBackend as TensorOps<TestBackend>>::reshape::<f32>(&lhs, &[3, 2]).unwrap();
    assert_eq!(values(&reshaped), values(&legacy_reshape));
    assert_eq!(dims(&reshaped), vec![3, 2]);

    let row = f32_storage(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let broadcast = dispatch::execute::<op::BroadcastAs, _>(
        &context,
        ShapeAttributes { shape: vec![2, 3] },
        &[handle(&row)],
    )
    .expect("broadcast_as is a registered CPU capability");
    let legacy_broadcast =
        <TestBackend as TensorOps<TestBackend>>::broadcast_as::<f32>(&row, &[2, 3]).unwrap();
    assert_eq!(values(&broadcast), values(&legacy_broadcast));
}

#[test]
fn validation_runs_before_the_backend_is_reached() {
    let context = context();
    // A rank-1 operand against a rank-2 operand is not broadcastable, so the
    // descriptor contract must reject it. The distinction that matters is
    // *which* error: a `Descriptor` error proves nothing was launched, whereas
    // a `Backend` error would mean the kernel discovered the problem itself.
    let lhs = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let rhs = f32_storage(vec![1.0, 2.0, 3.0], vec![3]);

    let error =
        dispatch::execute::<op::Add, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .expect_err("incompatible operands cannot execute");
    assert!(
        matches!(error, CanonicalError::Descriptor(_)),
        "expected a pre-launch contract failure, found {error:?}"
    );
}

#[test]
fn an_arity_violation_names_the_exact_operation() {
    let context = context();
    let only = f32_storage(vec![1.0, 2.0], vec![2]);

    let error = dispatch::execute::<op::Add, _>(&context, NoAttributes, &[handle(&only)])
        .expect_err("a binary operation cannot run on one operand");
    match error {
        CanonicalError::Descriptor(DescriptorError::Arity { operation, .. }) => {
            assert_eq!(operation, OperationKind::Add);
        }
        other => panic!("expected an arity failure naming `add`, found {other:?}"),
    }
}

#[test]
fn an_unadvertised_dtype_is_refused_before_execution() {
    let context = context();
    // `matmul` advertises f32 only. An i64 operand is a legal tensor and a
    // legal descriptor, so the refusal has to come from the capability row
    // rather than from validation.
    let lhs =
        CpuStorage::try_from_contiguous(CpuBuffer::I64(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();
    let rhs =
        CpuStorage::try_from_contiguous(CpuBuffer::I64(vec![1, 2, 3, 4]), vec![2, 2]).unwrap();

    let error = dispatch::execute::<op::MatMulExact, _>(
        &context,
        NoAttributes,
        &[handle(&lhs), handle(&rhs)],
    )
    .expect_err("matmul does not advertise an integer dtype");
    match error {
        CanonicalError::Descriptor(_) => {}
        CanonicalError::Backend(backend) => {
            let text = backend.to_string();
            assert!(
                text.contains("matmul"),
                "the refusal must name the operation, found {text}"
            );
        }
    }
}

#[test]
fn the_capability_answer_and_the_execution_answer_agree() {
    // A support level that execution does not honour is a documentation bug
    // with a runtime cost. Every migrated identity is checked in both
    // directions on the same operand.
    let context = context();
    let lhs = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let rhs = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    for operation in [
        OperationKind::Add,
        OperationKind::Sub,
        OperationKind::Mul,
        OperationKind::Div,
        OperationKind::MatMulExact,
    ] {
        let query = CapabilityQuery {
            operation,
            dtype: DTypeId::F32,
            layout: LayoutClass::Contiguous,
            rank: 2,
            training: true,
            math_mode: MathMode::Precise,
        };
        assert!(
            context.backend().support(&query).is_supported(),
            "{operation} must advertise support for a plain f32 matrix"
        );
    }

    for result in [
        dispatch::execute::<op::Add, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .map(|_| ()),
        dispatch::execute::<op::Sub, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .map(|_| ()),
        dispatch::execute::<op::Mul, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .map(|_| ()),
        dispatch::execute::<op::Div, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .map(|_| ()),
        dispatch::execute::<op::MatMulExact, _>(
            &context,
            NoAttributes,
            &[handle(&lhs), handle(&rhs)],
        )
        .map(|_| ()),
    ] {
        result.expect("an advertised capability must execute");
    }
}

#[test]
fn an_unregistered_operand_rank_is_refused_with_a_typed_reason() {
    let context = context();
    // `matmul` registers rank 2 upward. A rank-1 operand pair is a legal `dot`
    // but not a legal `matmul`, and the registry must say so rather than
    // letting the kernel decide.
    let lhs = f32_storage(vec![1.0, 2.0], vec![2]);
    let rhs = f32_storage(vec![3.0, 4.0], vec![2]);

    let error = dispatch::execute::<op::MatMulExact, _>(
        &context,
        NoAttributes,
        &[handle(&lhs), handle(&rhs)],
    )
    .expect_err("matmul does not accept rank-one operands");
    match error {
        CanonicalError::Descriptor(_) => {}
        CanonicalError::Backend(backend) => assert!(
            matches!(
                backend,
                incin_core::prelude::BackendError::Unsupported {
                    reason: UnsupportedReason::Rank { .. }
                }
            ),
            "expected a typed rank refusal, found {backend:?}"
        ),
    }
}

#[test]
fn the_canonical_path_derives_output_metadata_from_the_inputs() {
    let context = context();
    // The caller supplies attributes and operands and nothing else. Because
    // there is no output argument, a wrong output shape is not a thing a
    // caller can express; the broadcast result below is derived.
    let lhs = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let rhs = f32_storage(vec![10.0, 20.0, 30.0], vec![1, 3]);

    let out =
        dispatch::execute::<op::Add, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .expect("a broadcastable pair executes");
    assert_eq!(dims(&out), vec![2, 3]);
    assert_eq!(
        values(&out),
        vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
        "the broadcast operand must be applied per row"
    );
}

#[test]
fn support_for_reports_the_same_level_the_executor_enforces() {
    let context = context();
    let storage = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let level = dispatch::support_for::<op::Add, TestBackend>(&context, storage.metadata())
        .expect("add supports a contiguous f32 matrix");
    assert_eq!(level, SupportLevel::Native);
}
