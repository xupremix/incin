#![cfg(feature = "cpu")]

extern crate incin_core as incin;

use core::hint::black_box;
use core::mem::size_of;
use std::time::Instant;

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_core::backend_authoring::{
    Backend, Execute, ExecutionRequest, ModuleOps, StorageBackend, TensorOps,
};
use incin_core::exec::{
    Alignment, Conv2dArgs, Conv2dRule, Conv2dSpec, ExecutionContext, MatMulRule, MatMulSpec,
    Pool2dRule, Pool2dSpec, PoolOp, ReduceAllRule, ReduceAtRule, ReduceKeepAtRule, ReduceOp,
    ReductionSpec, ReshapeRule, ReshapeSpec, ShapeRule, TensorHandle, TensorMeta, Validated,
};
use incin_core::prelude::{
    BackendError, Cpu, DType, DTypeId, DeviceId, Dyn, Local, OperationKind, Shape, ShapeBuf, s,
};
use incin_core::shapes::idx::{Here, Next};
use incin_core::shapes::shape::{DimCons, Nil};
use incin_core::typenum::{U0, U1, U2, U3, U4};

type TestBackend = CpuBackendImpl<Cpu>;
type R2 = DimCons<U2, DimCons<U3, Nil>>;

fn field<S: Shape>(dims: &[usize]) -> ShapeBuf {
    S::try_from_dims(dims).expect("test dimensions must match the shape type")
}

fn lower(lhs: &[usize], rhs: &[usize]) -> Validated<MatMulSpec> {
    <MatMulRule as ShapeRule<(Dyn, Dyn)>>::lower(&(field::<Dyn>(lhs), field::<Dyn>(rhs)), ())
        .expect("test operands must be valid matmul shapes")
}

fn f32_storage(shape: &[usize], values: &[f32]) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values.to_vec()), shape.to_vec())
        .expect("test buffer must match its shape")
}

fn f64_storage(shape: &[usize], values: &[f64]) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::F64(values.to_vec()), shape.to_vec())
        .expect("test buffer must match its shape")
}

fn u32_storage(shape: &[usize], values: &[u32]) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::U32(values.to_vec()), shape.to_vec())
        .expect("test buffer must match its shape")
}

fn lower_reshape_2x6_to_3x4() -> Validated<ReshapeSpec> {
    <ReshapeRule as ShapeRule<(s![2, 6], s![3, 4])>>::lower(
        &(field::<s![2, 6]>(&[2, 6]), field::<s![3, 4]>(&[3, 4])),
        (),
    )
    .expect("12 elements either way")
}

fn execute_reshape<K: DType>(
    backend: &TestBackend,
    context: &ExecutionContext<TestBackend>,
    validated: &Validated<ReshapeSpec>,
    input: &CpuStorage,
) -> Result<CpuStorage, BackendError> {
    let inputs = [TensorHandle::from_storage::<TestBackend, K, Local>(input)];
    backend.execute(ExecutionRequest {
        operation: validated,
        inputs: &inputs,
        context,
    })
}

fn values(storage: &CpuStorage) -> Vec<f64> {
    let shape = storage.shape();
    let count = shape.dims().iter().product();
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

fn execute(
    backend: &TestBackend,
    context: &ExecutionContext<TestBackend>,
    validated: &Validated<MatMulSpec>,
    lhs: &CpuStorage,
    rhs: &CpuStorage,
) -> Result<CpuStorage, BackendError> {
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(lhs),
        TensorHandle::from_storage::<TestBackend, f32, Local>(rhs),
    ];
    backend.execute(ExecutionRequest {
        operation: validated,
        inputs: &inputs,
        context,
    })
}

fn assert_storage_eq(lhs: &CpuStorage, rhs: &CpuStorage) {
    assert_eq!(lhs.shape().dims(), rhs.shape().dims());
    let lhs_values = values(lhs);
    let rhs_values = values(rhs);
    assert_eq!(lhs_values.len(), rhs_values.len());
    for (index, (left, right)) in lhs_values.into_iter().zip(rhs_values).enumerate() {
        assert!(
            (left - right).abs() <= 1e-6,
            "value {index} differs: {left} versus {right}"
        );
    }
}

#[test]
fn rank2_descriptor_execution_matches_the_legacy_path() {
    let lhs = f32_storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let rhs = f32_storage(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let validated = lower(&[2, 3], &[3, 2]);
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());

    let legacy = <TestBackend as TensorOps<TestBackend>>::matmul::<f32>(&lhs, &rhs).unwrap();
    let descriptor = execute(&backend, &context, &validated, &lhs, &rhs).unwrap();

    assert_storage_eq(&descriptor, &legacy);
    assert_eq!(values(&descriptor), vec![58., 64., 139., 154.]);
}

#[test]
fn batched_broadcast_descriptor_execution_matches_the_legacy_path() {
    let lhs = f32_storage(
        &[2, 2, 3],
        &[1., 2., 3., 4., 5., 6., 6., 5., 4., 3., 2., 1.],
    );
    let rhs = f32_storage(&[1, 3, 2], &[1., 0., 0., 1., 1., 1.]);
    let validated = lower(&[2, 2, 3], &[1, 3, 2]);
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());

    let legacy = <TestBackend as TensorOps<TestBackend>>::matmul::<f32>(&lhs, &rhs).unwrap();
    let descriptor = execute(&backend, &context, &validated, &lhs, &rhs).unwrap();

    assert_storage_eq(&descriptor, &legacy);
    assert_eq!(descriptor.shape().dims(), &[2, 2, 2]);
}

#[test]
fn strided_view_descriptor_execution_matches_the_legacy_path() {
    let lhs_base = f32_storage(&[3, 2], &[1., 4., 2., 5., 3., 6.]);
    let lhs = lhs_base.transpose(0, 1).unwrap();
    let rhs = f32_storage(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let validated = lower(&[2, 3], &[3, 2]);
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());

    let legacy = <TestBackend as TensorOps<TestBackend>>::matmul::<f32>(&lhs, &rhs).unwrap();
    let descriptor = execute(&backend, &context, &validated, &lhs, &rhs).unwrap();

    assert_eq!(lhs.layout, incin_core::exec::LayoutClass::Strided);
    assert_storage_eq(&descriptor, &legacy);
}

fn forward_and_backward(use_descriptor: bool) -> (CpuStorage, CpuStorage, CpuStorage) {
    let lhs = f32_storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let rhs = f32_storage(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let output = if use_descriptor {
        let validated = lower(&[2, 3], &[3, 2]);
        let backend = TestBackend::new();
        let context = ExecutionContext::new(TestBackend::new());
        execute(&backend, &context, &validated, &lhs, &rhs).unwrap()
    } else {
        <TestBackend as TensorOps<TestBackend>>::matmul::<f32>(&lhs, &rhs).unwrap()
    };
    let grads = <TestBackend as Backend>::backward::<f32>(&output).unwrap();
    let lhs_grad = <TestBackend as Backend>::get_grad::<f32>(&lhs, &grads)
        .unwrap()
        .expect("lhs must receive a gradient");
    let rhs_grad = <TestBackend as Backend>::get_grad::<f32>(&rhs, &grads)
        .unwrap()
        .expect("rhs must receive a gradient");
    (output, lhs_grad, rhs_grad)
}

#[test]
fn descriptor_execution_preserves_forward_and_backward_parity() {
    let legacy = forward_and_backward(false);
    let descriptor = forward_and_backward(true);
    assert_storage_eq(&descriptor.0, &legacy.0);
    assert_storage_eq(&descriptor.1, &legacy.1);
    assert_storage_eq(&descriptor.2, &legacy.2);
}

#[derive(Clone)]
struct ForeignStorage {
    metadata: TensorMeta,
}

struct ForeignBackend;

impl StorageBackend for ForeignBackend {
    const BACKEND_NAME: &'static str = "Foreign";
    type Storage<K: DType> = ForeignStorage;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        &storage.metadata
    }
}

#[test]
fn binder_rejects_wrong_count_storage_dtype_and_descriptor_binding() {
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());
    let validated = lower(&[2, 3], &[3, 2]);
    let lhs = f32_storage(&[2, 3], &[1.; 6]);
    let rhs = f32_storage(&[3, 2], &[1.; 6]);

    let one_input = [TensorHandle::from_storage::<TestBackend, f32, Local>(&lhs)];
    let error = backend
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &one_input,
            context: &context,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::MatMulExact,
            reason: "matmul expects exactly two tensor inputs"
        }
    ));

    let foreign = ForeignStorage {
        metadata: TensorMeta::contiguous(
            ShapeBuf::from_slice(&[2, 3]),
            DTypeId::F32.descriptor(),
            DeviceId::cpu(),
            Alignment::of::<f32>(),
            6,
        )
        .unwrap(),
    };
    let foreign_inputs = [
        TensorHandle::from_storage::<ForeignBackend, f32, Local>(&foreign),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&rhs),
    ];
    let error = backend
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &foreign_inputs,
            context: &context,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            reason: "matmul input is not CPU storage",
            ..
        }
    ));

    let lhs_f64 = f64_storage(&[2, 3], &[1.; 6]);
    let rhs_f64 = f64_storage(&[3, 2], &[1.; 6]);
    let f64_inputs = [
        TensorHandle::from_storage::<TestBackend, f64, Local>(&lhs_f64),
        TensorHandle::from_storage::<TestBackend, f64, Local>(&rhs_f64),
    ];
    let error = backend
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &f64_inputs,
            context: &context,
        })
        .unwrap_err();
    if let BackendError::Unsupported {
        backend,
        reason: incin_core::exec::UnsupportedReason::DType { operation, dtype },
    } = error
    {
        assert_eq!(backend, "Cpu");
        assert_eq!(operation, OperationKind::MatMulExact);
        assert_eq!(dtype, DTypeId::F64.descriptor());
    } else {
        panic!("expected BackendError::Unsupported, got {:?}", error);
    }

    let wrong_lhs = f32_storage(&[2, 4], &[1.; 8]);
    let wrong_rhs = f32_storage(&[4, 2], &[1.; 8]);
    let wrong_inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&wrong_lhs),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&wrong_rhs),
    ];
    let error = backend
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &wrong_inputs,
            context: &context,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            reason: "matmul lhs metadata does not match the validated descriptor",
            ..
        }
    ));
}

#[test]
fn reshape_descriptor_execution_matches_the_legacy_path() {
    let input = f32_storage(
        &[2, 6],
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
    );
    let validated = lower_reshape_2x6_to_3x4();
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());

    let legacy = <TestBackend as TensorOps<TestBackend>>::reshape::<f32>(&input, &[3, 4]).unwrap();
    let descriptor = execute_reshape::<f32>(&backend, &context, &validated, &input).unwrap();

    assert_eq!(descriptor.shape().dims(), &[3, 4]);
    assert_storage_eq(&descriptor, &legacy);
    assert_eq!(
        values(&descriptor),
        (1..=12).map(f64::from).collect::<Vec<_>>()
    );
}

#[test]
fn reshape_descriptor_execution_materializes_a_strided_view() {
    // A transposed source is not contiguous, so the reshape cannot be a
    // re-addressing; the CPU registry classifies exactly this case as
    // `Composed`, and the executor must still produce the materialized result
    // rather than refuse it.
    let base = f32_storage(
        &[6, 2],
        &[1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12.],
    );
    let input = base.transpose(0, 1).unwrap();
    let validated = lower_reshape_2x6_to_3x4();
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());

    assert_eq!(input.layout, incin_core::exec::LayoutClass::Strided);
    let legacy = <TestBackend as TensorOps<TestBackend>>::reshape::<f32>(&input, &[3, 4]).unwrap();
    let descriptor = execute_reshape::<f32>(&backend, &context, &validated, &input).unwrap();

    assert_storage_eq(&descriptor, &legacy);
    assert_eq!(
        values(&descriptor),
        (1..=12).map(f64::from).collect::<Vec<_>>()
    );
}

#[test]
fn reshape_descriptor_execution_preserves_backward_parity() {
    let gradient = |use_descriptor: bool| {
        let input = f32_storage(&[2, 6], &[1.; 12]);
        let output = if use_descriptor {
            let backend = TestBackend::new();
            let context = ExecutionContext::new(TestBackend::new());
            execute_reshape::<f32>(&backend, &context, &lower_reshape_2x6_to_3x4(), &input).unwrap()
        } else {
            <TestBackend as TensorOps<TestBackend>>::reshape::<f32>(&input, &[3, 4]).unwrap()
        };
        let grads = <TestBackend as Backend>::backward::<f32>(&output).unwrap();
        <TestBackend as Backend>::get_grad::<f32>(&input, &grads)
            .unwrap()
            .expect("the reshaped input must receive a gradient")
    };

    assert_storage_eq(&gradient(true), &gradient(false));
}

#[test]
fn reshape_descriptor_execution_accepts_an_integer_dtype() {
    // Reshaping a `u32` tensor is a legal inference-time operation. The binder
    // must not ask the registry for trainability the request never claimed,
    // because no backend registers an integer dtype as trainable.
    let input = u32_storage(&[2, 6], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());

    let descriptor =
        execute_reshape::<u32>(&backend, &context, &lower_reshape_2x6_to_3x4(), &input).unwrap();

    assert_eq!(descriptor.shape().dims(), &[3, 4]);
    assert_eq!(descriptor.metadata().dtype(), DTypeId::U32.descriptor());
}

#[test]
fn reshape_binder_rejects_wrong_count_storage_and_descriptor_binding() {
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());
    let validated = lower_reshape_2x6_to_3x4();
    let input = f32_storage(&[2, 6], &[1.; 12]);

    let two_inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&input),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&input),
    ];
    let error = backend
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &two_inputs,
            context: &context,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::ReshapeExact,
            reason: "reshape expects exactly one tensor input"
        }
    ));

    let foreign = ForeignStorage {
        metadata: TensorMeta::contiguous(
            ShapeBuf::from_slice(&[2, 6]),
            DTypeId::F32.descriptor(),
            DeviceId::cpu(),
            Alignment::of::<f32>(),
            12,
        )
        .unwrap(),
    };
    let foreign_inputs = [TensorHandle::from_storage::<ForeignBackend, f32, Local>(
        &foreign,
    )];
    let error = backend
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &foreign_inputs,
            context: &context,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            reason: "reshape input is not CPU storage",
            ..
        }
    ));

    // Same element count, different input shape: the descriptor was proved
    // against `[2, 6]`, so `[4, 3]` is not the tensor it authorizes.
    let wrong_input = f32_storage(&[4, 3], &[1.; 12]);
    let wrong_inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(
        &wrong_input,
    )];
    let error = backend
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &wrong_inputs,
            context: &context,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            reason: "reshape input metadata does not match the validated descriptor",
            ..
        }
    ));
}

#[test]
fn request_layer_has_fixed_borrowed_footprint_and_reports_timing() {
    let word = size_of::<usize>();
    assert_eq!(size_of::<TestBackend>(), 0);
    assert_eq!(size_of::<TensorHandle<'_>>(), 3 * word);
    assert_eq!(
        size_of::<ExecutionRequest<'_, MatMulSpec, TestBackend>>(),
        4 * word
    );

    let lhs = f32_storage(&[32, 32], &[1.; 32 * 32]);
    let rhs = f32_storage(&[32, 32], &[1.; 32 * 32]);
    let validated = lower(&[32, 32], &[32, 32]);
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());
    const RUNS: u32 = 16;

    let warm_legacy = <TestBackend as TensorOps<TestBackend>>::matmul::<f32>(&lhs, &rhs).unwrap();
    let warm_descriptor = execute(&backend, &context, &validated, &lhs, &rhs).unwrap();
    <TestBackend as Backend>::backward::<f32>(&warm_descriptor).unwrap();

    let measure_legacy = || {
        let start = Instant::now();
        let mut output = None;
        for _ in 0..RUNS {
            output = Some(black_box(
                <TestBackend as TensorOps<TestBackend>>::matmul::<f32>(&lhs, &rhs).unwrap(),
            ));
        }
        let elapsed = start.elapsed();
        <TestBackend as Backend>::backward::<f32>(output.as_ref().unwrap()).unwrap();
        (elapsed, output.unwrap())
    };
    let measure_descriptor = || {
        let start = Instant::now();
        let mut output = None;
        for _ in 0..RUNS {
            output = Some(black_box(
                execute(&backend, &context, &validated, &lhs, &rhs).unwrap(),
            ));
        }
        let elapsed = start.elapsed();
        <TestBackend as Backend>::backward::<f32>(output.as_ref().unwrap()).unwrap();
        (elapsed, output.unwrap())
    };

    // Run both orders and add the durations so one path cannot win merely by
    // always running first or second on a warmed processor.
    let (legacy_first, legacy) = measure_legacy();
    let (descriptor_second, descriptor) = measure_descriptor();
    let (descriptor_first, _) = measure_descriptor();
    let (legacy_second, _) = measure_legacy();
    let legacy_elapsed = legacy_first + legacy_second;
    let descriptor_elapsed = descriptor_first + descriptor_second;

    assert_storage_eq(&descriptor, &legacy);
    assert_storage_eq(&warm_descriptor, &warm_legacy);
    eprintln!(
        "cpu executor timing (non-gating, {} warmed runs): legacy={:?}/call, descriptor={:?}/call",
        RUNS * 2,
        legacy_elapsed / (RUNS * 2),
        descriptor_elapsed / (RUNS * 2),
    );
}

/// A dense 3x3 convolution with padding 1, over a 1x2x4x4 input to 3 channels.
type Conv3x3 = Conv2dRule<U3, U3, U1, U1, U1>;
type ConvInput = s![1, 2, 4, 4];

fn lower_conv2d() -> Validated<Conv2dSpec> {
    <Conv3x3 as ShapeRule<ConvInput>>::lower(
        &field::<ConvInput>(&[1, 2, 4, 4]),
        Conv2dArgs::dense(3),
    )
    .expect("a 3x3 window with padding 1 fits a 4x4 input")
}

/// The same window over the same input, but four output channels in two groups
/// so each filter sees one of the two input channels.
type Conv3x3Grouped = Conv2dRule<U4, U3, U1, U1, U1>;

fn lower_conv2d_grouped() -> Validated<Conv2dSpec> {
    <Conv3x3Grouped as ShapeRule<ConvInput>>::lower(
        &field::<ConvInput>(&[1, 2, 4, 4]),
        Conv2dArgs {
            out_channels: 4,
            groups: 2,
        },
    )
    .expect("2 groups divide both 2 input and 4 output channels")
}

fn execute_conv2d(
    backend: &TestBackend,
    context: &ExecutionContext<TestBackend>,
    validated: &Validated<Conv2dSpec>,
    input: &CpuStorage,
    weight: &CpuStorage,
    bias: Option<&CpuStorage>,
) -> Result<CpuStorage, BackendError> {
    let mut inputs = vec![
        TensorHandle::from_storage::<TestBackend, f32, Local>(input),
        TensorHandle::from_storage::<TestBackend, f32, Local>(weight),
    ];
    if let Some(bias) = bias {
        inputs.push(TensorHandle::from_storage::<TestBackend, f32, Local>(bias));
    }
    backend.execute(ExecutionRequest {
        operation: validated,
        inputs: &inputs,
        context,
    })
}

fn conv_operands() -> (CpuStorage, CpuStorage, CpuStorage) {
    let input: Vec<f32> = (0..32).map(|value| value as f32 * 0.25 - 4.0).collect();
    let weight: Vec<f32> = (0..54).map(|value| value as f32 * 0.1 - 2.5).collect();
    (
        f32_storage(&[1, 2, 4, 4], &input),
        f32_storage(&[3, 2, 3, 3], &weight),
        f32_storage(&[3], &[0.5, -0.25, 1.0]),
    )
}

#[test]
fn conv2d_descriptor_execution_matches_the_legacy_path() {
    let (input, weight, bias) = conv_operands();
    let validated = lower_conv2d();
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());

    let legacy = <TestBackend as ModuleOps<TestBackend>>::conv2d::<f32>(
        &input,
        &weight,
        Some(&bias),
        1,
        1,
        1,
        1,
    )
    .unwrap();
    let descriptor =
        execute_conv2d(&backend, &context, &validated, &input, &weight, Some(&bias)).unwrap();

    assert_eq!(descriptor.shape().dims(), &[1, 3, 4, 4]);
    assert_storage_eq(&descriptor, &legacy);
}

#[test]
fn conv2d_descriptor_execution_without_a_bias_is_the_two_operand_form() {
    let (input, weight, _) = conv_operands();
    let validated = lower_conv2d();
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());

    let legacy =
        <TestBackend as ModuleOps<TestBackend>>::conv2d::<f32>(&input, &weight, None, 1, 1, 1, 1)
            .unwrap();
    let descriptor = execute_conv2d(&backend, &context, &validated, &input, &weight, None).unwrap();

    assert_storage_eq(&descriptor, &legacy);
}

#[test]
fn conv2d_descriptor_execution_preserves_backward_parity() {
    let gradients = |use_descriptor: bool| {
        let (input, weight, bias) = conv_operands();
        let output = if use_descriptor {
            let backend = TestBackend::new();
            let context = ExecutionContext::new(TestBackend::new());
            execute_conv2d(
                &backend,
                &context,
                &lower_conv2d(),
                &input,
                &weight,
                Some(&bias),
            )
            .unwrap()
        } else {
            <TestBackend as ModuleOps<TestBackend>>::conv2d::<f32>(
                &input,
                &weight,
                Some(&bias),
                1,
                1,
                1,
                1,
            )
            .unwrap()
        };
        let grads = <TestBackend as Backend>::backward::<f32>(&output).unwrap();
        let input_grad = <TestBackend as Backend>::get_grad::<f32>(&input, &grads)
            .unwrap()
            .expect("the convolved input must receive a gradient");
        let weight_grad = <TestBackend as Backend>::get_grad::<f32>(&weight, &grads)
            .unwrap()
            .expect("the filter bank must receive a gradient");
        (input_grad, weight_grad)
    };

    let descriptor = gradients(true);
    let legacy = gradients(false);
    assert_storage_eq(&descriptor.0, &legacy.0);
    assert_storage_eq(&descriptor.1, &legacy.1);
}

#[test]
fn conv2d_grouping_comes_from_the_descriptor_not_the_operand_shapes() {
    // With 2 groups each of the 4 filters sees one input channel, so the weight
    // is [4, 1, 3, 3]. A [4, 2, 3, 3] weight is a legal filter bank for the
    // ungrouped convolution and must still fail against *this* descriptor,
    // because grouping is the descriptor's fact and not the operands'.
    let (input, _, _) = conv_operands();
    let grouped_values: Vec<f32> = (0..36).map(|value| value as f32 * 0.1 - 1.75).collect();
    let ungrouped_values: Vec<f32> = (0..72).map(|value| value as f32 * 0.05 - 1.75).collect();
    let grouped = f32_storage(&[4, 1, 3, 3], &grouped_values);
    let ungrouped = f32_storage(&[4, 2, 3, 3], &ungrouped_values);
    let validated = lower_conv2d_grouped();
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());

    let descriptor =
        execute_conv2d(&backend, &context, &validated, &input, &grouped, None).unwrap();
    let legacy =
        <TestBackend as ModuleOps<TestBackend>>::conv2d::<f32>(&input, &grouped, None, 1, 1, 1, 2)
            .unwrap();
    assert_eq!(descriptor.shape().dims(), &[1, 4, 4, 4]);
    assert_storage_eq(&descriptor, &legacy);

    let error = execute_conv2d(&backend, &context, &validated, &input, &ungrouped, None)
        .expect_err("an ungrouped weight must not bind to a grouped descriptor");
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::Conv2dExact,
            reason: "conv2d weight metadata does not match the validated descriptor"
        }
    ));
}

#[test]
fn conv2d_binder_rejects_a_wrong_operand_count_and_a_mismatched_input() {
    let (input, weight, bias) = conv_operands();
    let validated = lower_conv2d();
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());

    let one_input = [TensorHandle::from_storage::<TestBackend, f32, Local>(
        &input,
    )];
    let error = backend
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &one_input,
            context: &context,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::Conv2dExact,
            reason: "conv2d expects an input and a weight, and optionally a bias"
        }
    ));

    let four_inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&input),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&weight),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&bias),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&bias),
    ];
    let error = backend
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &four_inputs,
            context: &context,
        })
        .unwrap_err();
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::Conv2dExact,
            ..
        }
    ));

    let wrong_input = f32_storage(&[1, 2, 5, 5], &[1.; 50]);
    let error = execute_conv2d(&backend, &context, &validated, &wrong_input, &weight, None)
        .expect_err("a 5x5 input must not bind to a descriptor proved for 4x4");
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            reason: "conv2d input metadata does not match the validated descriptor",
            ..
        }
    ));
}

// --- reduction and pooling ---------------------------------------------------

fn execute_one<O: incin_core::exec::OperationSpec>(
    backend: &TestBackend,
    context: &ExecutionContext<TestBackend>,
    validated: &Validated<O>,
    input: &CpuStorage,
) -> Result<CpuStorage, BackendError>
where
    TestBackend: Execute<O, Output = CpuStorage>,
{
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(input)];
    backend.execute(ExecutionRequest {
        operation: validated,
        inputs: &inputs,
        context,
    })
}

fn lower_reduce(op: ReduceOp) -> Validated<ReductionSpec> {
    <ReduceAtRule<Next<Here>> as ShapeRule<R2>>::lower(&field::<R2>(&[2, 3]), op)
        .expect("axis 1 is in range")
}

#[test]
fn a_reduction_descriptor_routes_to_the_accumulation_it_names() {
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());
    let input = f32_storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);

    // One geometry, five answers. Before the descriptor named its operator this
    // call had no way to tell them apart.
    for (op, expected) in [
        (ReduceOp::Sum, [6.0, 15.0]),
        (ReduceOp::Mean, [2.0, 5.0]),
        (ReduceOp::Max, [3.0, 6.0]),
        (ReduceOp::Min, [1.0, 4.0]),
        (ReduceOp::Prod, [6.0, 120.0]),
    ] {
        let validated = lower_reduce(op);
        assert_eq!(validated.descriptor().op, op);
        let output = execute_one(&backend, &context, &validated, &input)
            .unwrap_or_else(|error| panic!("{op} over axis 1 must execute: {error:?}"));

        assert_eq!(output.shape().dims(), &[2], "{op} drops the reduced axis");
        assert_eq!(values(&output), expected.to_vec(), "{op} over axis 1");
    }
}

/// A reduction over every axis executes, and matches reducing them one by one.
///
/// `ReductionSpec` has always accepted a contiguous run of axes, but the binder
/// refused anything wider than one and no rule produced one, so the schema
/// described an operation nothing could execute. `ReduceAllRule` produces it and
/// the executor collapses it a step at a time. The reference is the same data
/// reduced along axis 1 and then axis 0, which is what "associative" means here.
#[test]
fn a_reduction_over_every_axis_collapses_the_whole_run() {
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());
    let input = f32_storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);

    for (op, expected) in [
        (ReduceOp::Sum, 21.0),
        (ReduceOp::Mean, 3.5),
        (ReduceOp::Max, 6.0),
        (ReduceOp::Min, 1.0),
        (ReduceOp::Prod, 720.0),
    ] {
        let validated = <ReduceAllRule as ShapeRule<R2>>::lower(&field::<R2>(&[2, 3]), op)
            .expect("every axis is in range");
        let spec = validated.descriptor();
        assert_eq!(spec.axes.axes().count(), 2, "{op} names both axes");
        assert_eq!(spec.reduced, 6, "{op} collapses every element");

        let output = execute_one(&backend, &context, &validated, &input)
            .unwrap_or_else(|error| panic!("{op} over every axis must execute: {error:?}"));

        // Rank 0, not `[1]`. EXE-005 found WGPU reporting the latter.
        assert!(
            output.shape().dims().is_empty(),
            "{op} produces a scalar, not a rank-1 stand-in"
        );
        assert_eq!(values(&output), vec![expected], "{op} over every axis");
    }
}

#[test]
fn a_product_that_keeps_its_axis_is_composed_rather_than_refused() {
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());
    let input = f32_storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);

    // `ReductionOps` has a `prod_dim` and no `prod_keepdim`, so this is the one
    // reduction the executor cannot route with a single call.
    let validated = <ReduceKeepAtRule<Next<Here>> as ShapeRule<R2>>::lower(
        &field::<R2>(&[2, 3]),
        ReduceOp::Prod,
    )
    .expect("axis 1 is in range");
    let output = execute_one(&backend, &context, &validated, &input)
        .expect("a kept product must still execute");

    assert_eq!(output.shape().dims(), &[2, 1]);
    assert_eq!(values(&output), vec![6.0, 120.0]);
}

#[test]
fn a_reduction_will_not_bind_an_operand_of_the_wrong_geometry() {
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());
    let validated = lower_reduce(ReduceOp::Sum);

    // Six elements either way, so only the per-axis extents catch this.
    let transposed = f32_storage(&[3, 2], &[1., 2., 3., 4., 5., 6.]);
    let error = execute_one(&backend, &context, &validated, &transposed)
        .expect_err("a 3x2 operand must not bind to a descriptor proved for 2x3");

    assert!(matches!(
        error,
        BackendError::InvalidInput {
            reason: "reduction input metadata does not match the validated descriptor",
            ..
        }
    ));
}

fn lower_pool(op: PoolOp) -> Validated<Pool2dSpec> {
    <Pool2dRule<U2, U2, U0, U1> as ShapeRule<s![1, 1, 4, 4]>>::lower(
        &field::<s![1, 1, 4, 4]>(&[1, 1, 4, 4]),
        op,
    )
    .expect("a 2x2 window strided by 2 tiles a 4x4 input")
}

#[test]
fn a_pool_descriptor_routes_to_the_accumulation_it_names() {
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());
    let input = f32_storage(
        &[1, 1, 4, 4],
        &(1..=16).map(|v| v as f32).collect::<Vec<_>>(),
    );

    for (op, expected) in [
        (PoolOp::Max, vec![6.0, 8.0, 14.0, 16.0]),
        (PoolOp::Average, vec![3.5, 5.5, 11.5, 13.5]),
    ] {
        let validated = lower_pool(op);
        assert_eq!(validated.descriptor().op, op);
        let output = execute_one(&backend, &context, &validated, &input)
            .unwrap_or_else(|error| panic!("{op} pooling must execute: {error:?}"));

        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
        assert_eq!(values(&output), expected, "{op} pooling over a 2x2 window");
    }
}

#[test]
fn a_dilated_average_pool_is_refused_rather_than_quietly_densified() {
    let backend = TestBackend::new();
    let context = ExecutionContext::new(TestBackend::new());
    let input = f32_storage(&[1, 1, 8, 8], &[1.; 64]);

    let dilated = |op| {
        <Pool2dRule<U2, U2, U0, U2> as ShapeRule<s![1, 1, 8, 8]>>::lower(
            &field::<s![1, 1, 8, 8]>(&[1, 1, 8, 8]),
            op,
        )
        .expect("a dilated 2x2 window fits an 8x8 input")
    };

    // The geometry is legal for either operator; only one has a kernel for it.
    let max = dilated(PoolOp::Max);
    execute_one(&backend, &context, &max, &input).expect("max pooling takes a dilation");

    let average = dilated(PoolOp::Average);
    let error = execute_one(&backend, &context, &average, &input)
        .expect_err("average pooling has nowhere to put a dilation");
    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::AvgPool2d,
            reason: "average pooling has no dilated form; the routed kernel takes no dilation"
        }
    ));
}
