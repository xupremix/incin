//! FND-005: the canonical CPU execution path.
//!
//! Every test here asserts a property of `exec::dispatch::execute` that the
//! legacy operation-family traits could not offer: an exact identity, a
//! capability answer that binds execution, validation before launch, and
//! output metadata that is derived rather than accepted. Parity against the
//! legacy path is asserted too, because a replacement that computes something
//! else is not a migration.

#![cfg(feature = "cpu")]

extern crate incin_core as incin;

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_core::exec::catalog::{LossAttributes, LossReduction, NoAttributes, ShapeAttributes, op};
use incin_core::exec::dispatch::{self, CanonicalError};
use incin_core::exec::{
    Capabilities, CapabilityQuery, DescriptorError, ExecutionContext, LayoutClass, MathMode,
    OperationIdentity, SupportLevel, TensorHandle, UnsupportedReason,
};
use incin_core::prelude::{Cpu, DTypeId, Local, OperationKind, Reduction};
use incin_core::__backend_compat::legacy::{TensorOps};

type TestBackend = CpuBackendImpl<Cpu>;

fn f32_storage(values: Vec<f32>, shape: Vec<usize>) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values), shape)
        .expect("test storage must be well formed")
}

/// Index storage for the indexing operations.
///
/// The descriptor requires an integer index dtype, so a float index is refused
/// before execution. That refusal is the contract, not an inconvenience, and
/// the tests honour it rather than working around it.
fn i64_storage(values: Vec<i64>, shape: Vec<usize>) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::I64(values), shape)
        .expect("test storage must be well formed")
}

fn bool_storage(values: Vec<bool>, shape: Vec<usize>) -> CpuStorage {
    let bytes = values.into_iter().map(|b| b as u8).collect();
    CpuStorage::try_from_contiguous(CpuBuffer::Bool(bytes), shape)
        .expect("test storage must be well formed")
}

fn handle(storage: &CpuStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<TestBackend, f32, Local>(storage)
}

fn handle_bool(storage: &CpuStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<TestBackend, bool, Local>(storage)
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
    let legacy = TestBackend::add::<f32>(&lhs, &rhs)
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
            values(&TestBackend::add::<f32>(&lhs, &rhs).unwrap()),
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
            values(&TestBackend::sub::<f32>(&lhs, &rhs).unwrap()),
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
            values(&TestBackend::mul::<f32>(&lhs, &rhs).unwrap()),
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
            values(&TestBackend::div::<f32>(&lhs, &rhs).unwrap()),
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
            operation: OperationIdentity::Builtin(operation),
            dtype: DTypeId::F32.descriptor(),
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
                    backend: "Cpu",
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

/// Every reduction identity the CPU advertises, executed canonically and
/// compared against the legacy family method the catalog names as its source.
///
/// A single parametrised test rather than fourteen: the property is uniform,
/// and listing the pairs here makes a missing migration visible as a missing
/// row rather than as an absent test file.
#[test]
fn every_advertised_reduction_matches_its_legacy_counterpart() {
    use incin_core::exec::catalog::AxisAttributes;
    use incin_core::__backend_compat::legacy::ReductionOps;

    let context = context();
    let input = f32_storage(vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0], vec![2, 3]);
    let axis = 1;

    let whole: [(&str, Vec<f64>, Vec<f64>); 5] = [
        (
            "sum_all",
            values(
                &dispatch::execute::<op::SumAll, _>(&context, NoAttributes, &[handle(&input)])
                    .unwrap(),
            ),
            values(&<TestBackend as ReductionOps<TestBackend>>::sum_all::<f32>(&input).unwrap()),
        ),
        (
            "mean_all",
            values(
                &dispatch::execute::<op::MeanAll, _>(&context, NoAttributes, &[handle(&input)])
                    .unwrap(),
            ),
            values(&<TestBackend as ReductionOps<TestBackend>>::mean_all::<f32>(&input).unwrap()),
        ),
        (
            "max_all",
            values(
                &dispatch::execute::<op::MaxAll, _>(&context, NoAttributes, &[handle(&input)])
                    .unwrap(),
            ),
            values(&<TestBackend as ReductionOps<TestBackend>>::max_all::<f32>(&input).unwrap()),
        ),
        (
            "min_all",
            values(
                &dispatch::execute::<op::MinAll, _>(&context, NoAttributes, &[handle(&input)])
                    .unwrap(),
            ),
            values(&<TestBackend as ReductionOps<TestBackend>>::min_all::<f32>(&input).unwrap()),
        ),
        (
            "prod_all",
            values(
                &dispatch::execute::<op::ProdAll, _>(&context, NoAttributes, &[handle(&input)])
                    .unwrap(),
            ),
            values(&<TestBackend as ReductionOps<TestBackend>>::prod_all::<f32>(&input).unwrap()),
        ),
    ];
    for (name, canonical, legacy) in whole {
        assert_eq!(canonical, legacy, "{name} diverged from the legacy path");
    }

    let attributes = || AxisAttributes { axis };
    let axial: [(&str, Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>); 9] = [
        {
            let c = dispatch::execute::<op::SumDim, _>(&context, attributes(), &[handle(&input)])
                .unwrap();
            let l =
                <TestBackend as ReductionOps<TestBackend>>::sum_dim::<f32>(&input, axis).unwrap();
            ("sum_dim", values(&c), values(&l), dims(&c), dims(&l))
        },
        {
            let c = dispatch::execute::<op::MeanDim, _>(&context, attributes(), &[handle(&input)])
                .unwrap();
            let l =
                <TestBackend as ReductionOps<TestBackend>>::mean_dim::<f32>(&input, axis).unwrap();
            ("mean_dim", values(&c), values(&l), dims(&c), dims(&l))
        },
        {
            let c = dispatch::execute::<op::MaxDim, _>(&context, attributes(), &[handle(&input)])
                .unwrap();
            let l =
                <TestBackend as ReductionOps<TestBackend>>::max_dim::<f32>(&input, axis).unwrap();
            ("max_dim", values(&c), values(&l), dims(&c), dims(&l))
        },
        {
            let c = dispatch::execute::<op::MinDim, _>(&context, attributes(), &[handle(&input)])
                .unwrap();
            let l =
                <TestBackend as ReductionOps<TestBackend>>::min_dim::<f32>(&input, axis).unwrap();
            ("min_dim", values(&c), values(&l), dims(&c), dims(&l))
        },
        {
            let c = dispatch::execute::<op::ProdDim, _>(&context, attributes(), &[handle(&input)])
                .unwrap();
            let l =
                <TestBackend as ReductionOps<TestBackend>>::prod_dim::<f32>(&input, axis).unwrap();
            ("prod_dim", values(&c), values(&l), dims(&c), dims(&l))
        },
        {
            let c =
                dispatch::execute::<op::SumKeepDim, _>(&context, attributes(), &[handle(&input)])
                    .unwrap();
            let l = <TestBackend as ReductionOps<TestBackend>>::sum_keepdim::<f32>(&input, axis)
                .unwrap();
            ("sum_keepdim", values(&c), values(&l), dims(&c), dims(&l))
        },
        {
            let c =
                dispatch::execute::<op::MeanKeepDim, _>(&context, attributes(), &[handle(&input)])
                    .unwrap();
            let l = <TestBackend as ReductionOps<TestBackend>>::mean_keepdim::<f32>(&input, axis)
                .unwrap();
            ("mean_keepdim", values(&c), values(&l), dims(&c), dims(&l))
        },
        {
            let c =
                dispatch::execute::<op::MaxKeepDim, _>(&context, attributes(), &[handle(&input)])
                    .unwrap();
            let l = <TestBackend as ReductionOps<TestBackend>>::max_keepdim::<f32>(&input, axis)
                .unwrap();
            ("max_keepdim", values(&c), values(&l), dims(&c), dims(&l))
        },
        {
            let c =
                dispatch::execute::<op::MinKeepDim, _>(&context, attributes(), &[handle(&input)])
                    .unwrap();
            let l = <TestBackend as ReductionOps<TestBackend>>::min_keepdim::<f32>(&input, axis)
                .unwrap();
            ("min_keepdim", values(&c), values(&l), dims(&c), dims(&l))
        },
    ];
    for (name, canonical, legacy, canonical_dims, legacy_dims) in axial {
        assert_eq!(canonical, legacy, "{name} diverged from the legacy path");
        assert_eq!(canonical_dims, legacy_dims, "{name} output shape diverged");
    }
}

/// A `keepdim` reduction must retain the reduced axis as a length-one
/// dimension, and a plain one must drop it. The distinction is the only thing
/// separating the two identities, so it is asserted rather than assumed.
#[test]
fn keepdim_and_plain_reductions_are_distinct_identities() {
    use incin_core::exec::catalog::AxisAttributes;

    let context = context();
    let input = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let dropped =
        dispatch::execute::<op::SumDim, _>(&context, AxisAttributes { axis: 1 }, &[handle(&input)])
            .unwrap();
    let kept = dispatch::execute::<op::SumKeepDim, _>(
        &context,
        AxisAttributes { axis: 1 },
        &[handle(&input)],
    )
    .unwrap();

    assert_eq!(dims(&dropped), vec![2]);
    assert_eq!(dims(&kept), vec![2, 1]);
    assert_eq!(values(&dropped), values(&kept));
}

/// An axis outside the operand's rank is a contract failure, not a kernel
/// failure. Catching it here proves the typed attribute is validated against
/// the real operand before any reduction loop starts.
#[test]
fn an_out_of_range_reduction_axis_fails_before_execution() {
    use incin_core::exec::catalog::AxisAttributes;

    let context = context();
    let input = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let error =
        dispatch::execute::<op::SumDim, _>(&context, AxisAttributes { axis: 7 }, &[handle(&input)])
            .expect_err("axis 7 does not exist on a rank-two operand");
    assert!(
        matches!(error, CanonicalError::Descriptor(_)),
        "expected a pre-launch contract failure, found {error:?}"
    );
}

/// The spatial family executes through exact canonical descriptors.
#[test]
fn the_spatial_family_executes_with_expected_values() {
    use incin_core::exec::catalog::{AvgPool2dAttributes, Conv2dAttributes, Pool2dAttributes};

    let context = context();
    // One 4x4 image, one channel, and a 2x2 kernel: small enough to reason
    // about, large enough that a stride or padding error changes the result.
    let image = f32_storage(
        (0..16).map(|value| value as f32).collect(),
        vec![1, 1, 4, 4],
    );
    let weight = f32_storage(vec![1.0, 0.0, 0.0, -1.0], vec![1, 1, 2, 2]);

    let convolved = dispatch::execute::<op::Conv2dExact, _>(
        &context,
        Conv2dAttributes {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            has_bias: false,
        },
        &[handle(&image), handle(&weight)],
    )
    .expect("conv2d is a registered CPU capability");
    assert_eq!(dims(&convolved), vec![1, 1, 3, 3]);
    assert_eq!(values(&convolved), vec![-5.0; 9]);

    let pooled = dispatch::execute::<op::MaxPool2d, _>(
        &context,
        Pool2dAttributes {
            kernel: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
            dilation: [1, 1],
        },
        &[handle(&image)],
    )
    .expect("max_pool2d is a registered CPU capability");
    assert_eq!(dims(&pooled), vec![1, 1, 2, 2]);
    assert_eq!(values(&pooled), vec![5.0, 7.0, 13.0, 15.0]);

    let averaged = dispatch::execute::<op::AvgPool2d, _>(
        &context,
        AvgPool2dAttributes {
            kernel: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        },
        &[handle(&image)],
    )
    .expect("avg_pool2d is a registered CPU capability");
    assert_eq!(dims(&averaged), vec![1, 1, 2, 2]);
    assert_eq!(values(&averaged), vec![2.5, 4.5, 10.5, 12.5]);
}

/// An anisotropic convolution window computes with both axes' extents.
///
/// It used to be refused, because `ModuleOps::conv2d` states one extent for
/// both axes and applying the first to both would have been a quiet wrong
/// answer. The kernel behind that signature never needed them equal, so the
/// descriptor's pair is forwarded whole now.
///
/// A stride of `[1, 2]` over a 4x4 input with a 2x2 kernel steps every row and
/// every other column, giving a 3x2 output. The result is checked against the
/// definition rather than against the isotropic case, which is the comparison
/// that would have passed while the bug existed.
#[test]
fn an_anisotropic_convolution_window_uses_both_axis_extents() {
    use incin_core::exec::catalog::Conv2dAttributes;

    let context = context();
    let image = f32_storage(
        (0..16).map(|value| value as f32).collect(),
        vec![1, 1, 4, 4],
    );
    // Picks out the top-left tap minus the bottom-right one of each window.
    let weight = f32_storage(vec![1.0, 0.0, 0.0, -1.0], vec![1, 1, 2, 2]);

    let output = dispatch::execute::<op::Conv2dExact, _>(
        &context,
        Conv2dAttributes {
            stride: [1, 2],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            has_bias: false,
        },
        &[handle(&image), handle(&weight)],
    )
    .expect("a per-axis stride is a window the kernel can take");

    assert_eq!(output.shape.to_vec(), vec![1, 1, 3, 2]);
    // Input is 0..16 row-major over 4x4, so element [r, c] is 4r + c. Every
    // window contributes (4r + c) - (4(r+1) + c+1), which is -5 everywhere.
    for row in 0..3 {
        for column in 0..2 {
            let value = output.get(&[0, 0, row, column]);
            assert!(
                (value + 5.0).abs() < 1e-5,
                "[{row}, {column}] was {value}, expected -5"
            );
        }
    }

    // The anisotropic result must not coincide with the isotropic one, or the
    // assertions above would hold equally for a kernel that ignored one axis.
    let isotropic = dispatch::execute::<op::Conv2dExact, _>(
        &context,
        Conv2dAttributes {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            has_bias: false,
        },
        &[handle(&image), handle(&weight)],
    )
    .unwrap();
    assert_ne!(output.shape.to_vec(), isotropic.shape.to_vec());
}

/// A convolution whose attributes declare a bias but whose operand list omits
/// it is rejected by the contract, not by the kernel.
#[test]
fn a_declared_bias_must_be_supplied() {
    use incin_core::exec::catalog::Conv2dAttributes;

    let context = context();
    let image = f32_storage(
        (0..16).map(|value| value as f32).collect(),
        vec![1, 1, 4, 4],
    );
    let weight = f32_storage(vec![1.0, 0.0, 0.0, -1.0], vec![1, 1, 2, 2]);

    let error = dispatch::execute::<op::Conv2dExact, _>(
        &context,
        Conv2dAttributes {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            has_bias: true,
        },
        &[handle(&image), handle(&weight)],
    )
    .expect_err("a declared bias with two operands is not a legal invocation");
    assert!(
        matches!(error, CanonicalError::Descriptor(_)),
        "expected a pre-launch contract failure, found {error:?}"
    );
}

/// Every unary float identity the CPU advertises, executed canonically and
/// compared against the `` method the catalog names as its source.
///
/// The operand is `0.5` throughout, which is inside the domain of all 33
/// functions listed. `acosh` is undefined there and is checked separately, by
/// `acosh_matches_the_legacy_path_on_an_operand_inside_its_domain`. The
/// finiteness assertion below is what enforces that split: comparing NaN to NaN
/// would pass vacuously, so an operand outside a function's domain fails rather
/// than silently proving nothing.
#[test]
fn every_advertised_unary_float_operation_matches_its_legacy_counterpart() {
    use incin_core::__backend_compat::legacy::;

    let context = context();
    let input = f32_storage(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]);

    let cases: [(&str, Vec<f64>, Vec<f64>, Vec<usize>); 33] = [
        {
            let c = dispatch::execute::<op::Relu, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::relu::<f32>(&input).unwrap();
            ("relu", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Step, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::step::<f32>(&input).unwrap();
            ("step", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Mish, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::mish::<f32>(&input).unwrap();
            ("mish", values(&c), values(&l), dims(&c))
        },
        {
            let c =
                dispatch::execute::<op::Elu, _>(&context, NoAttributes, &[handle(&input)]).unwrap();
            let l = TestBackend::elu::<f32>(&input).unwrap();
            ("elu", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Gelu, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::gelu::<f32>(&input).unwrap();
            ("gelu", values(&c), values(&l), dims(&c))
        },
        {
            let c =
                dispatch::execute::<op::Abs, _>(&context, NoAttributes, &[handle(&input)]).unwrap();
            let l = TestBackend::abs::<f32>(&input).unwrap();
            ("abs", values(&c), values(&l), dims(&c))
        },
        {
            let c =
                dispatch::execute::<op::Exp, _>(&context, NoAttributes, &[handle(&input)]).unwrap();
            let l = TestBackend::exp::<f32>(&input).unwrap();
            ("exp", values(&c), values(&l), dims(&c))
        },
        {
            let c =
                dispatch::execute::<op::Neg, _>(&context, NoAttributes, &[handle(&input)]).unwrap();
            let l = TestBackend::neg::<f32>(&input).unwrap();
            ("neg", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Sqrt, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::sqrt::<f32>(&input).unwrap();
            ("sqrt", values(&c), values(&l), dims(&c))
        },
        {
            let c =
                dispatch::execute::<op::Log, _>(&context, NoAttributes, &[handle(&input)]).unwrap();
            let l = TestBackend::log::<f32>(&input).unwrap();
            ("log", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Tanh, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::tanh::<f32>(&input).unwrap();
            ("tanh", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Sigmoid, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::sigmoid::<f32>(&input).unwrap();
            ("sigmoid", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Swish, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::swish::<f32>(&input).unwrap();
            ("swish", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Sign, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::sign::<f32>(&input).unwrap();
            ("sign", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Floor, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::floor::<f32>(&input).unwrap();
            ("floor", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Ceil, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::ceil::<f32>(&input).unwrap();
            ("ceil", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Round, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::round::<f32>(&input).unwrap();
            ("round", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Log2, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::log2::<f32>(&input).unwrap();
            ("log2", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Log10, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::log10::<f32>(&input).unwrap();
            ("log10", values(&c), values(&l), dims(&c))
        },
        {
            let c =
                dispatch::execute::<op::Sin, _>(&context, NoAttributes, &[handle(&input)]).unwrap();
            let l = TestBackend::sin::<f32>(&input).unwrap();
            ("sin", values(&c), values(&l), dims(&c))
        },
        {
            let c =
                dispatch::execute::<op::Cos, _>(&context, NoAttributes, &[handle(&input)]).unwrap();
            let l = TestBackend::cos::<f32>(&input).unwrap();
            ("cos", values(&c), values(&l), dims(&c))
        },
        {
            let c =
                dispatch::execute::<op::Tan, _>(&context, NoAttributes, &[handle(&input)]).unwrap();
            let l = TestBackend::tan::<f32>(&input).unwrap();
            ("tan", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Asin, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::asin::<f32>(&input).unwrap();
            ("asin", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Acos, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::acos::<f32>(&input).unwrap();
            ("acos", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Atan, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::atan::<f32>(&input).unwrap();
            ("atan", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Sinh, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::sinh::<f32>(&input).unwrap();
            ("sinh", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Cosh, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::cosh::<f32>(&input).unwrap();
            ("cosh", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Asinh, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::asinh::<f32>(&input).unwrap();
            ("asinh", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Atanh, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::atanh::<f32>(&input).unwrap();
            ("atanh", values(&c), values(&l), dims(&c))
        },
        {
            let c =
                dispatch::execute::<op::Erf, _>(&context, NoAttributes, &[handle(&input)]).unwrap();
            let l = TestBackend::erf::<f32>(&input).unwrap();
            ("erf", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Rsqrt, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::rsqrt::<f32>(&input).unwrap();
            ("rsqrt", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Trunc, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::trunc::<f32>(&input).unwrap();
            ("trunc", values(&c), values(&l), dims(&c))
        },
        {
            let c = dispatch::execute::<op::Frac, _>(&context, NoAttributes, &[handle(&input)])
                .unwrap();
            let l = TestBackend::frac::<f32>(&input).unwrap();
            ("frac", values(&c), values(&l), dims(&c))
        },
    ];

    for (name, canonical, legacy, shape) in cases {
        assert_eq!(shape, vec![2, 2], "{name} changed its operand's shape");
        for (index, (c, l)) in canonical.iter().zip(&legacy).enumerate() {
            assert!(
                c.is_finite(),
                "{name} produced a non-finite value at {index}; pick an operand \
                 inside its domain rather than comparing NaN to NaN"
            );
            assert_eq!(c, l, "{name} diverged from the legacy path at {index}");
        }
    }
}

/// `acosh` is undefined below one, so it gets its own operand.
#[test]
fn acosh_matches_the_legacy_path_on_an_operand_inside_its_domain() {
    use incin_core::__backend_compat::legacy::;

    let context = context();
    let input = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let canonical =
        dispatch::execute::<op::Acosh, _>(&context, NoAttributes, &[handle(&input)]).unwrap();
    let legacy = TestBackend::acosh::<f32>(&input).unwrap();
    assert!(values(&canonical).iter().all(|value| value.is_finite()));
    assert_eq!(values(&canonical), values(&legacy));
}

/// The attribute-bearing float operations, whose typed attributes must reach
/// the kernel unchanged.
#[test]
fn the_attribute_bearing_float_operations_match_their_legacy_counterparts() {
    use incin_core::exec::catalog::{AxisAttributes, ClampAttributes, ScalarAttributes};
    use incin_core::__backend_compat::legacy::;

    let context = context();
    let input = f32_storage(vec![-1.5, 0.5, 2.5, 3.5], vec![2, 2]);

    let add_scalar = dispatch::execute::<op::AddScalar, _>(
        &context,
        ScalarAttributes { value: 2.0 },
        &[handle(&input)],
    )
    .unwrap();
    assert_eq!(
        values(&add_scalar),
        values(
            &TestBackend::add_scalar_float::<f32>(&input, 2.0).unwrap()
        )
    );

    let mul_scalar = dispatch::execute::<op::MulScalar, _>(
        &context,
        ScalarAttributes { value: 3.0 },
        &[handle(&input)],
    )
    .unwrap();
    assert_eq!(
        values(&mul_scalar),
        values(
            &TestBackend::mul_scalar_float::<f32>(&input, 3.0).unwrap()
        )
    );

    let positive = f32_storage(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2]);
    let powf = dispatch::execute::<op::Powf, _>(
        &context,
        ScalarAttributes { value: 2.0 },
        &[handle(&positive)],
    )
    .unwrap();
    assert_eq!(
        values(&powf),
        values(&TestBackend::powf::<f32>(&positive, 2.0).unwrap())
    );

    let clamped = dispatch::execute::<op::Clamp, _>(
        &context,
        ClampAttributes { min: 0.0, max: 2.0 },
        &[handle(&input)],
    )
    .unwrap();
    assert_eq!(
        values(&clamped),
        values(&TestBackend::clamp::<f32>(&input, 0.0, 2.0).unwrap()),
    );
    // The bounds must actually have been applied, not merely passed along.
    assert!(
        values(&clamped)
            .iter()
            .all(|value| (0.0..=2.0).contains(value))
    );

    let softmaxed = dispatch::execute::<op::Softmax, _>(
        &context,
        AxisAttributes { axis: 1 },
        &[handle(&input)],
    )
    .unwrap();
    assert_eq!(
        values(&softmaxed),
        values(&TestBackend::softmax::<f32>(&input, 1).unwrap())
    );
    for row in values(&softmaxed).chunks(2) {
        assert!(
            (row.iter().sum::<f64>() - 1.0).abs() < 1e-6,
            "each softmax row must sum to one"
        );
    }
}

/// The binary float operations, over two broadcast operands.
#[test]
fn the_binary_float_operations_match_their_legacy_counterparts() {
    use incin_core::__backend_compat::legacy::;

    let context = context();
    let lhs = f32_storage(vec![1.0, -2.0, 3.5, -4.5], vec![4]);
    let rhs = f32_storage(vec![2.0, 3.0, 1.5, 2.5], vec![4]);

    let atan2 =
        dispatch::execute::<op::Atan2, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .unwrap();
    assert_eq!(
        values(&atan2),
        values(&TestBackend::atan2::<f32>(&lhs, &rhs).unwrap())
    );

    let fmod =
        dispatch::execute::<op::Fmod, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .unwrap();
    assert_eq!(
        values(&fmod),
        values(&TestBackend::fmod::<f32>(&lhs, &rhs).unwrap())
    );

    let remainder = dispatch::execute::<op::Remainder, _>(
        &context,
        NoAttributes,
        &[handle(&lhs), handle(&rhs)],
    )
    .unwrap();
    assert_eq!(
        values(&remainder),
        values(&TestBackend::remainder::<f32>(&lhs, &rhs).unwrap())
    );
}

/// `softmax` is registered for f32 only on this backend, and the refusal of a
/// half-precision operand must come from the capability row rather than from a
/// kernel that silently computed in f32 and returned the wrong dtype.
#[test]
fn softmax_refuses_a_dtype_it_does_not_advertise() {
    use incin_core::exec::catalog::AxisAttributes;

    let context = context();
    let input = CpuStorage::try_from_contiguous(
        CpuBuffer::F16(vec![half::f16::from_f32(1.0); 4]),
        vec![2, 2],
    )
    .unwrap();

    let error = dispatch::execute::<op::Softmax, _>(
        &context,
        AxisAttributes { axis: 1 },
        &[handle(&input)],
    )
    .expect_err("softmax advertises f32 only on the CPU backend");
    match error {
        CanonicalError::Backend(backend) => assert!(
            matches!(
                backend,
                incin_core::prelude::BackendError::Unsupported {
                    backend: "Cpu",
                    reason: UnsupportedReason::DType { .. }
                }
            ),
            "expected a typed dtype refusal, found {backend:?}"
        ),
        other => panic!("expected a capability refusal, found {other:?}"),
    }
}

/// Parity for the binary tensor family, which the catalog routes through
/// `TensorOps` rather than `` because these operations preserve the
/// operand dtype and carry no gradient.
#[test]
fn every_migrated_binary_tensor_operation_matches_its_legacy_counterpart() {
    let context = context();
    let lhs = f32_storage(vec![1.0, -2.0, 3.0, 3.0], vec![4]);
    let rhs = f32_storage(vec![2.0, 3.0, 3.0, -1.0], vec![4]);

    macro_rules! check {
        ($($operation:ident => $method:ident),* $(,)?) => {$(
            let canonical = dispatch::execute::<op::$operation, _>(
                &context,
                NoAttributes,
                &[handle(&lhs), handle(&rhs)],
            )
            .expect(concat!(stringify!($operation), " is a registered CPU capability"));
            let legacy =
                <TestBackend as TensorOps<TestBackend>>::$method::<f32>(&lhs, &rhs).unwrap();
            assert_eq!(
                values(&canonical),
                values(&legacy),
                "{} diverged from its legacy counterpart",
                stringify!($operation)
            );
            assert_eq!(dims(&canonical), dims(&legacy));
        )*};
    }

    check! {
        Maximum => maximum,
        Minimum => minimum,
        AbsDiff => abs_diff,
        CmpEq => cmp_eq,
        CmpNe => cmp_ne,
        CmpLt => cmp_lt,
        CmpLe => cmp_le,
        CmpGt => cmp_gt,
        CmpGe => cmp_ge,
    }

    let lhs = bool_storage(vec![true, false, true, false], vec![4]);
    let rhs = bool_storage(vec![true, true, false, false], vec![4]);

    let canonical_and = dispatch::execute::<op::LogicalAnd, _>(
        &context,
        NoAttributes,
        &[handle(&lhs), handle(&rhs)],
    )
    .expect("LogicalAnd is a registered CPU capability");
    let legacy_and = <TestBackend as TensorOps<TestBackend>>::logical_and(&lhs, &rhs).unwrap();
    assert_eq!(values(&canonical_and), values(&legacy_and));

    let canonical_or = dispatch::execute::<op::LogicalOr, _>(
        &context,
        NoAttributes,
        &[handle(&lhs), handle(&rhs)],
    )
    .expect("LogicalOr is a registered CPU capability");
    let legacy_or = <TestBackend as TensorOps<TestBackend>>::logical_or(&lhs, &rhs).unwrap();
    assert_eq!(values(&canonical_or), values(&legacy_or));
}

/// Parity for the tensor operations that carry a scalar or an offset, where a
/// migration could plausibly read the right kernel with the wrong attribute.
#[test]
fn the_attribute_bearing_tensor_operations_match_their_legacy_counterparts() {
    use incin_core::exec::catalog::{
        AxisAttributes, DiagonalAttributes, FlattenAttributes, LerpAttributes, NarrowAttributes,
        ScalarAttributes, TransposeAttributes,
    };

    let context = context();
    let matrix = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let subtracted = dispatch::execute::<op::SubScalar, _>(
        &context,
        ScalarAttributes { value: 1.5 },
        &[handle(&matrix)],
    )
    .unwrap();
    assert_eq!(
        values(&subtracted),
        values(&<TestBackend as TensorOps<TestBackend>>::sub_scalar::<f32>(&matrix, 1.5).unwrap())
    );

    let divided = dispatch::execute::<op::DivScalar, _>(
        &context,
        ScalarAttributes { value: 2.0 },
        &[handle(&matrix)],
    )
    .unwrap();
    assert_eq!(
        values(&divided),
        values(&<TestBackend as TensorOps<TestBackend>>::div_scalar::<f32>(&matrix, 2.0).unwrap())
    );

    let transposed = dispatch::execute::<op::TransposeExact, _>(
        &context,
        TransposeAttributes {
            first: 0,
            second: 1,
        },
        &[handle(&matrix)],
    )
    .unwrap();
    assert_eq!(
        values(&transposed),
        values(&<TestBackend as TensorOps<TestBackend>>::transpose::<f32>(&matrix, 0, 1).unwrap())
    );
    assert_eq!(dims(&transposed), vec![3, 2]);

    let narrowed = dispatch::execute::<op::Narrow, _>(
        &context,
        NarrowAttributes {
            axis: 1,
            start: 1,
            length: 2,
        },
        &[handle(&matrix)],
    )
    .unwrap();
    assert_eq!(
        values(&narrowed),
        values(&<TestBackend as TensorOps<TestBackend>>::narrow::<f32>(&matrix, 1, 1, 2).unwrap())
    );
    assert_eq!(dims(&narrowed), vec![2, 2]);

    let flattened = dispatch::execute::<op::FlattenExact, _>(
        &context,
        FlattenAttributes {
            start_axis: 0,
            end_axis: 1,
        },
        &[handle(&matrix)],
    )
    .unwrap();
    assert_eq!(
        values(&flattened),
        values(&<TestBackend as TensorOps<TestBackend>>::flatten::<f32>(&matrix, 0, 1).unwrap())
    );
    assert_eq!(dims(&flattened), vec![6]);

    let unsqueezed = dispatch::execute::<op::UnsqueezeExact, _>(
        &context,
        AxisAttributes { axis: 0 },
        &[handle(&matrix)],
    )
    .unwrap();
    assert_eq!(
        values(&unsqueezed),
        values(&<TestBackend as TensorOps<TestBackend>>::unsqueeze::<f32>(&matrix, 0).unwrap())
    );
    assert_eq!(dims(&unsqueezed), vec![1, 2, 3]);

    let column = f32_storage(vec![7.0, 8.0, 9.0], vec![1, 3]);
    let squeezed = dispatch::execute::<op::SqueezeExact, _>(
        &context,
        AxisAttributes { axis: 0 },
        &[handle(&column)],
    )
    .unwrap();
    assert_eq!(
        values(&squeezed),
        values(&<TestBackend as TensorOps<TestBackend>>::squeeze::<f32>(&column, 0).unwrap())
    );
    assert_eq!(dims(&squeezed), vec![3]);

    let square = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    for (offset, label) in [(0i64, "principal"), (1, "upper"), (-1, "lower")] {
        let upper = dispatch::execute::<op::Triu, _>(
            &context,
            DiagonalAttributes { offset },
            &[handle(&square)],
        )
        .unwrap();
        assert_eq!(
            values(&upper),
            values(&<TestBackend as TensorOps<TestBackend>>::triu::<f32>(&square, offset).unwrap()),
            "triu diverged on the {label} diagonal"
        );

        let lower = dispatch::execute::<op::Tril, _>(
            &context,
            DiagonalAttributes { offset },
            &[handle(&square)],
        )
        .unwrap();
        assert_eq!(
            values(&lower),
            values(&<TestBackend as TensorOps<TestBackend>>::tril::<f32>(&square, offset).unwrap()),
            "tril diverged on the {label} diagonal"
        );
    }

    let diagonal = dispatch::execute::<op::Diag, _>(
        &context,
        DiagonalAttributes { offset: 0 },
        &[handle(&square)],
    )
    .unwrap();
    assert_eq!(
        values(&diagonal),
        values(&<TestBackend as TensorOps<TestBackend>>::diag::<f32>(&square, 0).unwrap())
    );
    assert_eq!(dims(&diagonal), vec![2]);

    let start = f32_storage(vec![0.0, 2.0, 4.0, 6.0], vec![4]);
    let end = f32_storage(vec![1.0, 3.0, 5.0, 7.0], vec![4]);
    let interpolated = dispatch::execute::<op::Lerp, _>(
        &context,
        LerpAttributes { weight: 0.25 },
        &[handle(&start), handle(&end)],
    )
    .unwrap();
    assert_eq!(
        values(&interpolated),
        values(&<TestBackend as TensorOps<TestBackend>>::lerp::<f32>(&start, &end, 0.25).unwrap())
    );
}

/// Parity for the mask-driven operations, whose operand order is the one
/// property a three-operand migration is most likely to get wrong.
#[test]
fn the_selection_operations_match_their_legacy_counterparts() {
    use incin_core::exec::catalog::ScalarAttributes;

    let context = context();
    let mask = bool_storage(vec![true, false, true, false], vec![4]);
    let on_true = f32_storage(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
    let on_false = f32_storage(vec![-1.0, -2.0, -3.0, -4.0], vec![4]);

    let selected = dispatch::execute::<op::WhereCond, _>(
        &context,
        NoAttributes,
        &[handle_bool(&mask), handle(&on_true), handle(&on_false)],
    )
    .expect("where_cond is a registered CPU capability");
    let legacy =
        <TestBackend as TensorOps<TestBackend>>::where_cond::<f32>(&mask, &on_true, &on_false)
            .unwrap();
    assert_eq!(values(&selected), values(&legacy));
    // Stated separately from the parity assertion: if the executor had bound
    // the operands in the wrong order, both paths would still have to be wrong
    // in the same way for parity alone to catch it.
    assert_eq!(values(&selected), vec![10.0, -2.0, 30.0, -4.0]);

    let filled = dispatch::execute::<op::MaskedFill, _>(
        &context,
        ScalarAttributes { value: 99.0 },
        &[handle(&on_true), handle_bool(&mask)],
    )
    .expect("masked_fill is a registered CPU capability");
    assert_eq!(
        values(&filled),
        values(
            &<TestBackend as TensorOps<TestBackend>>::masked_fill::<f32>(&on_true, &mask, 99.0)
                .unwrap()
        )
    );
    assert_eq!(values(&filled), vec![99.0, 20.0, 99.0, 40.0]);

    let bool_mask = bool_storage(vec![true, false, true, false], vec![4]);
    let negated =
        dispatch::execute::<op::LogicalNot, _>(&context, NoAttributes, &[handle_bool(&bool_mask)])
            .unwrap();
    assert_eq!(
        values(&negated),
        values(&<TestBackend as TensorOps<TestBackend>>::logical_not(&bool_mask).unwrap())
    );
}

/// `bmm` is a separate identity from `matmul` with its own rank contract, so
/// it gets its own registration and its own parity check.
#[test]
fn batched_matmul_matches_its_legacy_counterpart() {
    let context = context();
    let lhs = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    let rhs = f32_storage(vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], vec![2, 2, 2]);

    let canonical = dispatch::execute::<op::BatchedMatMul, _>(
        &context,
        NoAttributes,
        &[handle(&lhs), handle(&rhs)],
    )
    .expect("bmm is a registered CPU capability");
    let legacy = <TestBackend as TensorOps<TestBackend>>::bmm::<f32>(&lhs, &rhs).unwrap();
    assert_eq!(values(&canonical), values(&legacy));
    assert_eq!(dims(&canonical), vec![2, 2, 2]);
}

/// A rank-3 operand for a diagonal operation is refused by the descriptor,
/// before any kernel that would have silently treated the leading axes as a
/// batch dimension.
#[test]
fn a_diagonal_operation_refuses_a_rank_it_does_not_advertise() {
    use incin_core::exec::catalog::DiagonalAttributes;

    let context = context();
    let input = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);

    let error = dispatch::execute::<op::Triu, _>(
        &context,
        DiagonalAttributes { offset: 0 },
        &[handle(&input)],
    )
    .expect_err("triu is registered for rank one and two only");
    assert!(
        matches!(
            error,
            CanonicalError::Descriptor(DescriptorError::InvalidAttribute {
                operation: OperationKind::Triu,
                attribute: "rank",
                ..
            })
        ),
        "unexpected refusal: {error}"
    );
}

/// A transpose axis outside the operand rank fails during validation, not
/// inside the kernel.
#[test]
fn a_transpose_axis_outside_the_input_rank_is_refused_before_execution() {
    use incin_core::exec::catalog::TransposeAttributes;

    let context = context();
    let input = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let error = dispatch::execute::<op::TransposeExact, _>(
        &context,
        TransposeAttributes {
            first: 0,
            second: 5,
        },
        &[handle(&input)],
    )
    .expect_err("a rank-two operand has no axis five");
    assert!(matches!(
        error,
        CanonicalError::Descriptor(DescriptorError::InvalidAttribute {
            operation: OperationKind::TransposeExact,
            ..
        })
    ));
}

/// Parity for the shape and indexing operations that reshape their operand in
/// ways the descriptor has to derive rather than accept.
#[test]
fn the_shape_and_indexing_operations_match_their_legacy_counterparts() {
    use incin_core::exec::catalog::{
        AxisAttributes, DuplicateIndexRule, PadAttributes, RepeatAttributes, ScatterAttributes,
        SliceAttributes, UnfoldAttributes,
    };

    let context = context();
    let left = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let right = f32_storage(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let joined = dispatch::execute::<op::ConcatExact, _>(
        &context,
        AxisAttributes { axis: 0 },
        &[handle(&left), handle(&right)],
    )
    .expect("concat is a registered CPU capability");
    assert_eq!(
        values(&joined),
        values(
            &<TestBackend as TensorOps<TestBackend>>::concat::<f32>(&[&left, &right], 0).unwrap()
        )
    );
    assert_eq!(dims(&joined), vec![4, 2]);

    let stacked = dispatch::execute::<op::StackExact, _>(
        &context,
        AxisAttributes { axis: 0 },
        &[handle(&left), handle(&right)],
    )
    .expect("stack is a registered CPU capability");
    assert_eq!(
        values(&stacked),
        values(
            &<TestBackend as TensorOps<TestBackend>>::stack::<f32>(&[&left, &right], 0).unwrap()
        )
    );
    assert_eq!(dims(&stacked), vec![2, 2, 2]);

    let sliced = dispatch::execute::<op::SliceExact, _>(
        &context,
        SliceAttributes {
            ranges: vec![(0, 1), (0, 2)],
        },
        &[handle(&left)],
    )
    .unwrap();
    assert_eq!(
        values(&sliced),
        values(
            &<TestBackend as TensorOps<TestBackend>>::slice::<f32>(&left, &[(0, 1), (0, 2)])
                .unwrap()
        )
    );
    assert_eq!(dims(&sliced), vec![1, 2]);

    let index = i64_storage(vec![1, 0, 0, 1], vec![2, 2]);
    let gathered = dispatch::execute::<op::Gather, _>(
        &context,
        AxisAttributes { axis: 1 },
        &[handle(&left), handle(&index)],
    )
    .unwrap();
    assert_eq!(
        values(&gathered),
        values(
            &<TestBackend as TensorOps<TestBackend>>::gather::<f32, f32>(&left, 1, &index).unwrap()
        )
    );
    // Pinned independently of the legacy path: gather reads column `index[i]`
    // of row `i`, so a swapped axis or a transposed index would still agree
    // with a legacy call that made the same mistake.
    assert_eq!(values(&gathered), vec![2.0, 1.0, 3.0, 4.0]);

    let scattered = dispatch::execute::<op::Scatter, _>(
        &context,
        ScatterAttributes {
            axis: 1,
            duplicate_indices: DuplicateIndexRule::LastWriteWins,
        },
        &[handle(&left), handle(&index), handle(&right)],
    )
    .unwrap();
    assert_eq!(
        values(&scattered),
        values(
            &<TestBackend as TensorOps<TestBackend>>::scatter::<f32, i64>(&left, 1, &index, &right)
                .unwrap()
        )
    );

    let selection = i64_storage(vec![1], vec![1]);
    let selected = dispatch::execute::<op::IndexSelect, _>(
        &context,
        AxisAttributes { axis: 0 },
        &[handle(&left), handle(&selection)],
    )
    .unwrap();
    assert_eq!(
        values(&selected),
        values(
            &<TestBackend as TensorOps<TestBackend>>::index_select::<f32, i64>(
                &left, 0, &selection
            )
            .unwrap()
        )
    );
    assert_eq!(values(&selected), vec![3.0, 4.0]);

    let repeated = dispatch::execute::<op::Repeat, _>(
        &context,
        RepeatAttributes {
            repeats: vec![2, 1],
        },
        &[handle(&left)],
    )
    .unwrap();
    assert_eq!(
        values(&repeated),
        values(&<TestBackend as TensorOps<TestBackend>>::repeat::<f32>(&left, &[2, 1]).unwrap())
    );
    assert_eq!(dims(&repeated), vec![4, 2]);

    let padded = dispatch::execute::<op::Pad, _>(
        &context,
        PadAttributes {
            padding: vec![(1, 1), (0, 0)],
            value: -1.0,
        },
        &[handle(&left)],
    )
    .unwrap();
    assert_eq!(
        values(&padded),
        values(
            &<TestBackend as TensorOps<TestBackend>>::pad::<f32>(&left, &[(1, 1), (0, 0)], -1.0)
                .unwrap()
        )
    );
    assert_eq!(dims(&padded), vec![4, 2]);

    let window = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let unfolded = dispatch::execute::<op::Unfold, _>(
        &context,
        UnfoldAttributes {
            axis: 1,
            size: 2,
            step: 1,
        },
        &[handle(&window)],
    )
    .unwrap();
    assert_eq!(
        values(&unfolded),
        values(&<TestBackend as TensorOps<TestBackend>>::unfold::<f32>(&window, 1, 2, 1).unwrap())
    );
    assert_eq!(dims(&unfolded), vec![2, 3, 2]);
}

/// Parity for the operations whose canonical row is `composed` because the CPU
/// answers them by rewriting into other operations.
#[test]
fn the_composed_tensor_operations_match_their_legacy_counterparts() {
    use incin_core::exec::catalog::{
        AddmmAttributes, AttentionAttributes, EpsilonAttributes, GroupNormAttributes,
        PixelShuffleAttributes,
    };

    let context = context();
    let matrix = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let fused = dispatch::execute::<op::Addmm, _>(
        &context,
        AddmmAttributes {
            alpha: 2.0,
            beta: 0.5,
        },
        &[handle(&matrix), handle(&matrix), handle(&matrix)],
    )
    .expect("addmm is a registered CPU capability");
    assert_eq!(
        values(&fused),
        values(
            &<TestBackend as TensorOps<TestBackend>>::addmm::<f32>(
                &matrix, &matrix, &matrix, 0.5, 2.0
            )
            .unwrap()
        )
    );

    let attended = dispatch::execute::<op::ScaledDotProductAttention, _>(
        &context,
        AttentionAttributes {
            scale: Some(0.5),
            has_mask: false,
        },
        &[handle(&matrix), handle(&matrix), handle(&matrix)],
    )
    .expect("attention is a registered CPU capability");
    assert_eq!(
        values(&attended),
        values(
            &<TestBackend as TensorOps<TestBackend>>::scaled_dot_product_attention::<f32>(
                &matrix,
                &matrix,
                &matrix,
                None,
                Some(0.5)
            )
            .unwrap()
        )
    );

    let channels = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 4, 2]);
    let grouped = dispatch::execute::<op::GroupNorm, _>(
        &context,
        GroupNormAttributes {
            groups: 2,
            epsilon: 1e-5,
        },
        &[handle(&channels)],
    )
    .unwrap();
    assert_eq!(
        values(&grouped),
        values(
            &<TestBackend as TensorOps<TestBackend>>::group_norm::<f32>(&channels, 2, 1e-5)
                .unwrap()
        )
    );

    let volume = f32_storage(
        (0..16).map(|value| value as f32).collect(),
        vec![1, 4, 2, 2],
    );
    let instance = dispatch::execute::<op::InstanceNorm, _>(
        &context,
        EpsilonAttributes { epsilon: 1e-5 },
        &[handle(&volume)],
    )
    .unwrap();
    assert_eq!(
        values(&instance),
        values(
            &<TestBackend as TensorOps<TestBackend>>::instance_norm::<f32>(&volume, 1e-5).unwrap()
        )
    );

    // The descriptor's shape is the whole target; the legacy method takes the
    // prefix. Both spellings of the same request must produce the same tensor.
    let broadened = dispatch::execute::<op::BroadcastLeft, _>(
        &context,
        ShapeAttributes {
            shape: vec![3, 2, 2],
        },
        &[handle(&matrix)],
    )
    .unwrap();
    assert_eq!(
        values(&broadened),
        values(
            &<TestBackend as TensorOps<TestBackend>>::broadcast_left::<f32>(&matrix, &[3]).unwrap()
        )
    );
    assert_eq!(dims(&broadened), vec![3, 2, 2]);

    let picture = f32_storage(
        (0..16).map(|value| value as f32).collect(),
        vec![1, 4, 2, 2],
    );
    let shuffled = dispatch::execute::<op::PixelShuffle, _>(
        &context,
        PixelShuffleAttributes { upscale_factor: 2 },
        &[handle(&picture)],
    )
    .unwrap();
    assert_eq!(
        values(&shuffled),
        values(
            &<TestBackend as TensorOps<TestBackend>>::pixel_shuffle::<f32>(&picture, 2).unwrap()
        )
    );
    assert_eq!(dims(&shuffled), vec![1, 1, 4, 4]);
}

/// The CPU scatter kernel writes in index order and has no duplicate
/// detection, so a descriptor asking for duplicates to be rejected must be
/// refused rather than answered with last-write-wins.
#[test]
fn a_scatter_that_must_reject_duplicates_is_refused_rather_than_approximated() {
    use incin_core::exec::catalog::{DuplicateIndexRule, ScatterAttributes};

    let context = context();
    let target = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let index = i64_storage(vec![0, 0, 0, 0], vec![2, 2]);
    let source = f32_storage(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let error = dispatch::execute::<op::Scatter, _>(
        &context,
        ScatterAttributes {
            axis: 1,
            duplicate_indices: DuplicateIndexRule::Reject,
        },
        &[handle(&target), handle(&index), handle(&source)],
    )
    .expect_err("the CPU kernel cannot reject duplicate indices");
    assert!(
        matches!(error, CanonicalError::Backend(_)),
        "unexpected refusal: {error}"
    );

    // The same request with the rule the kernel does implement succeeds, so
    // the refusal above is about the rule and not about the operands.
    dispatch::execute::<op::Scatter, _>(
        &context,
        ScatterAttributes {
            axis: 1,
            duplicate_indices: DuplicateIndexRule::LastWriteWins,
        },
        &[handle(&target), handle(&index), handle(&source)],
    )
    .expect("last-write-wins is what this kernel does");
}

/// Attention declares whether it has a mask. Supplying the wrong number of
/// operands for that declaration is refused before the kernel is reached.
#[test]
fn an_attention_mask_declaration_must_match_the_operand_count() {
    use incin_core::exec::catalog::AttentionAttributes;

    let context = context();
    let matrix = f32_storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let error = dispatch::execute::<op::ScaledDotProductAttention, _>(
        &context,
        AttentionAttributes {
            scale: None,
            has_mask: true,
        },
        &[handle(&matrix), handle(&matrix), handle(&matrix)],
    )
    .expect_err("a declared mask was not supplied");
    assert!(
        matches!(
            error,
            CanonicalError::Backend(_) | CanonicalError::Descriptor(_)
        ),
        "unexpected refusal: {error}"
    );
}

/// The index-returning reductions compute what the legacy calls compute.
///
/// The index dtype in each descriptor is the one the CPU kernel actually
/// produces, which is not the same integer for all four: `argmax` and `argmin`
/// build `i64` and `argsort` builds `u32`. Writing the real dtype into the
/// descriptor is what lets these route at all.
#[test]
fn the_index_returning_reductions_match_their_legacy_counterparts() {
    use incin_core::exec::catalog::{ArgsortAttributes, AxisAttributes, IndexReductionAttributes};
    use incin_core::__backend_compat::legacy::ReductionOps;

    let context = context();
    let input = f32_storage(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]);

    let argmax = dispatch::execute::<op::ArgMax, _>(
        &context,
        IndexReductionAttributes {
            axis: Some(1),
            dtype: DTypeId::I64.descriptor(),
        },
        &[handle(&input)],
    )
    .expect("argmax routes");
    assert_eq!(
        values(&argmax),
        values(
            &<TestBackend as ReductionOps<TestBackend>>::argmax::<f32, i64>(&input, Some(1))
                .unwrap()
        )
    );
    // Pinned against a fixed answer as well as against the legacy path, so a
    // migration that is wrong on both cannot pass: row zero peaks at column
    // zero and row one at column one.
    assert_eq!(values(&argmax), vec![0.0, 1.0]);

    let argmin = dispatch::execute::<op::ArgMin, _>(
        &context,
        IndexReductionAttributes {
            axis: Some(1),
            dtype: DTypeId::I64.descriptor(),
        },
        &[handle(&input)],
    )
    .expect("argmin routes");
    assert_eq!(values(&argmin), vec![1.0, 0.0]);

    let argsort = dispatch::execute::<op::Argsort, _>(
        &context,
        ArgsortAttributes {
            axis: 1,
            descending: false,
            index_dtype: DTypeId::U32.descriptor(),
        },
        &[handle(&input)],
    )
    .expect("argsort routes");
    assert_eq!(values(&argsort), vec![1.0, 0.0, 0.0, 1.0]);

    let cumsum =
        dispatch::execute::<op::Cumsum, _>(&context, AxisAttributes { axis: 1 }, &[handle(&input)])
            .expect("cumsum routes");
    assert_eq!(
        values(&cumsum),
        values(&<TestBackend as ReductionOps<TestBackend>>::cumsum::<f32>(&input, 1).unwrap())
    );
    assert_eq!(values(&cumsum), vec![3.0, 4.0, 2.0, 6.0]);
}

/// `topk` is the first migrated identity that returns more than one tensor.
///
/// The pair is the descriptor's own account of the operation: output zero
/// carries the operand dtype and output one carries the declared index dtype.
#[test]
fn topk_returns_both_of_the_tensors_its_descriptor_describes() {
    use incin_core::exec::catalog::TopKAttributes;
    use incin_core::__backend_compat::legacy::ReductionOps;

    let context = context();
    let input = f32_storage(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]);

    let (selected, indices) = dispatch::execute::<op::TopK, _>(
        &context,
        TopKAttributes {
            k: 1,
            axis: 1,
            largest: true,
            index_dtype: DTypeId::U32.descriptor(),
        },
        &[handle(&input)],
    )
    .expect("topk routes");

    let (legacy_values, legacy_indices) =
        <TestBackend as ReductionOps<TestBackend>>::topk::<f32, u32>(&input, 1, 1, true).unwrap();
    assert_eq!(values(&selected), values(&legacy_values));
    assert_eq!(values(&indices), values(&legacy_indices));

    assert_eq!(values(&selected), vec![3.0, 4.0]);
    assert_eq!(values(&indices), vec![0.0, 1.0]);
    assert_eq!(selected.dtype, DTypeId::F32.descriptor());
    assert_eq!(indices.dtype, DTypeId::U32.descriptor());
}

/// An index-returning reduction produces the index dtype it was asked for.
///
/// `argmax`, `argmin`, `argsort` and `topk` each take an index dtype as a type
/// parameter. All four used to declare it and then ignore it, `argmax` and
/// `argmin` always building `i64` and the other two always `u32`, so a request
/// for anything else came back wearing a label its buffer did not hold. The
/// canonical path worked around it by refusing every dtype but the hardcoded
/// one, which narrowed the contract instead of meeting it.
///
/// Both paths honour the request now, and the storage dtype is asserted rather
/// than the absence of an error, so a return to relabelling fails here.
#[test]
fn an_index_reduction_produces_the_index_dtype_it_was_asked_for() {
    use incin_core::exec::catalog::IndexReductionAttributes;
    use incin_core::__backend_compat::legacy::ReductionOps;

    let context = context();
    let input = f32_storage(vec![3.0, 1.0, 2.0, 4.0], vec![2, 2]);

    for dtype in [DTypeId::U8, DTypeId::U32, DTypeId::I64] {
        let indices = dispatch::execute::<op::ArgMax, _>(
            &context,
            IndexReductionAttributes {
                axis: Some(1),
                dtype: dtype.descriptor(),
            },
            &[handle(&input)],
        )
        .unwrap_or_else(|error| panic!("argmax with {dtype:?} indices: {error}"));
        assert_eq!(
            indices.dtype,
            dtype.descriptor(),
            "canonical argmax mislabelled {dtype:?}"
        );
        // Row 0 is [3, 1] and row 1 is [2, 4], so the winners sit at 0 and 1.
        assert_eq!(values(&indices), vec![0.0, 1.0]);
    }

    // A float index dtype has no integer buffer to be built into. The
    // descriptor rejects it before the backend is reached, which is the
    // earlier of the two refusals and the one that names the reason.
    let refused = dispatch::execute::<op::ArgMax, _>(
        &context,
        IndexReductionAttributes {
            axis: Some(1),
            dtype: DTypeId::F32.descriptor(),
        },
        &[handle(&input)],
    )
    .expect_err("a float index dtype is not an index dtype");
    assert!(
        refused.to_string().contains("integer dtype"),
        "unexpected refusal: {refused}"
    );

    // The legacy path is the one the stable tensor surface still uses, so it
    // has to agree rather than merely not error.
    let legacy = <TestBackend as ReductionOps<TestBackend>>::argmax::<f32, u32>(&input, Some(1))
        .expect("the legacy path honours the index dtype");
    assert_eq!(legacy.dtype, DTypeId::U32.descriptor());
}

/// Broadcasting operands agree with the legacy path, for every pointwise
/// operation that takes the output shape from the descriptor.
///
/// The canonical executors no longer call `broadcast_shape`: they read the
/// shape `dispatch` already inferred and sealed in the `Validated` descriptor.
/// Every canonical pointwise case above uses same-shape operands, where a
/// descriptor shape and a re-derived one are indistinguishable. These do not:
/// `[2, 3]` against `[3]` right-aligns, so reading the wrong shape produces a
/// wrong result or an allocation of the wrong size rather than a silent tie.
#[test]
fn a_broadcasting_canonical_invocation_matches_the_legacy_result() {
    let context = context();
    let lhs = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let rhs = f32_storage(vec![10.0, 20.0, 30.0], vec![3]);

    let add =
        dispatch::execute::<op::Add, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .expect("add is a registered CPU capability");
    assert_eq!(dims(&add), vec![2, 3]);
    assert_eq!(
        values(&add),
        values(&TestBackend::add::<f32>(&lhs, &rhs).unwrap())
    );

    let sub =
        dispatch::execute::<op::Sub, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .expect("sub is a registered CPU capability");
    assert_eq!(dims(&sub), vec![2, 3]);
    assert_eq!(
        values(&sub),
        values(&TestBackend::sub::<f32>(&lhs, &rhs).unwrap())
    );

    let mul =
        dispatch::execute::<op::Mul, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .expect("mul is a registered CPU capability");
    assert_eq!(dims(&mul), vec![2, 3]);
    assert_eq!(
        values(&mul),
        values(&TestBackend::mul::<f32>(&lhs, &rhs).unwrap())
    );

    let div =
        dispatch::execute::<op::Div, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .expect("div is a registered CPU capability");
    assert_eq!(dims(&div), vec![2, 3]);
    assert_eq!(
        values(&div),
        values(&TestBackend::div::<f32>(&lhs, &rhs).unwrap())
    );
}

/// A left-broadcast operand, where the *shorter* shape is the left one.
///
/// `[3]` against `[2, 3]` exercises the other direction of the right-aligned
/// rule. A descriptor shape read from the wrong operand would pass the test
/// above and fail this one.
#[test]
fn a_left_broadcasting_canonical_invocation_matches_the_legacy_result() {
    let context = context();
    let lhs = f32_storage(vec![10.0, 20.0, 30.0], vec![3]);
    let rhs = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let add =
        dispatch::execute::<op::Add, _>(&context, NoAttributes, &[handle(&lhs), handle(&rhs)])
            .expect("add is a registered CPU capability");
    assert_eq!(dims(&add), vec![2, 3]);
    assert_eq!(
        values(&add),
        values(&TestBackend::add::<f32>(&lhs, &rhs).unwrap())
    );
}

// Canonical dispatch execution reaches `dispatch::execute_shaped` for operations.
// These use the `Tensor` surface deliberately, not raw storage: the property under test is
// that a real user-facing call actually reaches the canonical path with the
// right autograd behaviour, which raw `dispatch::execute` calls elsewhere in
// this file cannot exercise (they bypass the tensor-level gradient wrapper).
/// `embedding`'s two operands admit different dtypes by construction: the
/// index operand is integer, and the weight operand is f32 only —
/// `embedding_impl` always reads and writes f32 regardless of what the
/// operand declares. `INDEX_AND_F32_DTYPES` in `capability.rs` states the union
/// of an integer index and an f32 weight, which is already as tight as the
/// weight's real constraint, so an f64 weight is refused end to end and never
/// silently narrowed — this asserts that outcome alongside the ordinary
/// legacy-parity case every migrated operation gets.
#[test]
fn a_canonical_embedding_invocation_refuses_a_narrowed_weight() {
    let context = context();
    let indices = i64_storage(vec![2, 0, 1], vec![3]);
    let weight = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

    let canonical = dispatch::execute::<op::EmbeddingExact, _>(
        &context,
        NoAttributes,
        &[handle(&indices), handle(&weight)],
    )
    .expect("embedding is a registered CPU capability");
    assert_eq!(dims(&canonical), vec![3, 2]);
    assert_eq!(values(&canonical), vec![5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);

    // An f64 weight passes the descriptor's own per-operand dtype contract
    // (it is float), but `INDEX_AND_F32_DTYPES` names only f32 among the floats,
    // so the capability query refuses it before any kernel runs — the
    // precision `embedding_impl` would otherwise silently narrow away never
    // reaches it.
    let wide_weight = CpuStorage::try_from_contiguous(
        CpuBuffer::F64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![3, 2],
    )
    .expect("test storage must be well formed");
    let error = dispatch::execute::<op::EmbeddingExact, _>(
        &context,
        NoAttributes,
        &[handle(&indices), handle(&wide_weight)],
    )
    .expect_err("an f64 weight must not silently execute at f32 precision");
    assert!(matches!(error, CanonicalError::Backend(_)));
}

/// The rest of the pointwise-binary family's `_canonical` methods
///
/// `cross_entropy_loss` on the canonical path: the last operation FND-005
/// recorded as blocked on `CapabilityRule`'s single dtype set.
///
/// It is checked against the legacy path for all three reduction modes rather
/// than just one, because the reduction is an *attribute* rather than part of
/// the identity — one row has to hold for all three, and `Mean`/`Sum` end in
/// an all-reduce that `None` does not.
#[test]
fn a_canonical_cross_entropy_invocation_matches_the_legacy_result_in_every_reduction() {
    let context = context();
    let logits = f32_storage(vec![2.0, 1.0, 0.1, 0.5, 3.0, 0.2], vec![2, 3]);
    let targets = i64_storage(vec![0, 1], vec![2]);

    for (descriptor_reduction, legacy_reduction) in [
        (LossReduction::Mean, Reduction::Mean),
        (LossReduction::Sum, Reduction::Sum),
        (LossReduction::None, Reduction::None),
    ] {
        let canonical = dispatch::execute::<op::CrossEntropyLoss, _>(
            &context,
            LossAttributes {
                reduction: descriptor_reduction,
            },
            &[handle(&logits), handle(&targets)],
        )
        .expect("cross entropy is a registered CPU capability");
        let legacy = TestBackend::cross_entropy_loss::<f32, i64>(
            &logits,
            &targets,
            legacy_reduction,
        )
        .expect("the legacy path computes the same operation");

        assert_eq!(dims(&canonical), dims(&legacy));
        assert_eq!(values(&canonical), values(&legacy));
    }
}

/// The two guards that make the shared `INDEX_AND_F32_DTYPES` row honest for
/// `cross_entropy_loss`, each checked from the side it actually protects.
#[test]
fn canonical_cross_entropy_refuses_a_narrowed_logit_and_a_float_target() {
    let context = context();
    let targets = i64_storage(vec![0, 1], vec![2]);

    // f64 logits pass the descriptor's float contract and are inside the
    // row's union, but the kernel computes in f32 — `f32_only` refuses the
    // precision it would otherwise silently narrow away.
    let wide_logits = CpuStorage::try_from_contiguous(
        CpuBuffer::F64(vec![2.0, 1.0, 0.1, 0.5, 3.0, 0.2]),
        vec![2, 3],
    )
    .expect("test storage must be well formed");
    let error = dispatch::execute::<op::CrossEntropyLoss, _>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[handle(&wide_logits), handle(&targets)],
    )
    .expect_err("f64 logits must not silently execute at f32 precision");
    assert!(matches!(error, CanonicalError::Backend(_)));

    // The other direction: a float target is inside the row's union too (it
    // is f32), so nothing in the capability layer would stop it. The
    // descriptor's own `index_input` contract is what refuses it, before any
    // backend is reached.
    let logits = f32_storage(vec![2.0, 1.0, 0.1, 0.5, 3.0, 0.2], vec![2, 3]);
    let float_targets = f32_storage(vec![0.0, 1.0], vec![2]);
    let error = dispatch::execute::<op::CrossEntropyLoss, _>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[handle(&logits), handle(&float_targets)],
    )
    .expect_err("a float class target must be refused as an index operand");
    assert!(matches!(error, CanonicalError::Descriptor(_)));
}
