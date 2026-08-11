//! `EXE-008`: the descriptor execution contract on real WGPU hardware.
//!
//! `EXE-007` proved the contract for CPU. These cases prove the same sealed
//! `Validated<MatMulSpec>` binds and executes against a second, genuinely
//! different backend — so the descriptor path is a shared contract rather than
//! a CPU-shaped one — and that the binder's rejections are enforced per backend
//! rather than inherited from the CPU implementation.
#![cfg(feature = "wgpu")]

extern crate incin_core as incin;

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{Execute, ExecutionRequest, ModuleOps, TensorOps};
use incin_core::exec::{
    Conv2dArgs, Conv2dRule, Conv2dSpec, ExecutionContext, MatMulRule, MatMulSpec, Pool2dRule,
    PoolOp, ReduceAtRule, ReduceOp, ReshapeRule, ReshapeSpec, ShapeRule, TensorHandle, Validated,
};
use incin_core::prelude::{
    Backend, BackendError, DTypeId, DeviceId, Dyn, Local, OperationKind, Shape, ShapeBuf, WgpuN, s,
};
use incin_core::shapes::idx::{Here, Next};
use incin_core::shapes::shape::{DimCons, Nil};
use incin_core::typenum::{U0, U1, U2, U3};

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as incin_core::backend_authoring::StorageBackend>::Storage<f32>;
type R2 = DimCons<U2, DimCons<U3, Nil>>;

fn field<S: Shape>(dims: &[usize]) -> ShapeBuf {
    S::try_from_dims(dims).expect("test dimensions must match the shape type")
}

fn lower(lhs: &[usize], rhs: &[usize]) -> Validated<MatMulSpec> {
    <MatMulRule as ShapeRule<(Dyn, Dyn)>>::lower(&(field::<Dyn>(lhs), field::<Dyn>(rhs)), ())
        .expect("test operands must be valid matmul shapes")
}

fn storage(shape: &[usize], values: &[f32]) -> TestStorage {
    TestBackend::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32.into(),
        &DeviceId::wgpu(0),
    )
    .expect("test buffer must match its shape")
}

fn read(storage: &TestStorage) -> Vec<f32> {
    let bytes = TestBackend::to_bytes::<f32>(storage).expect("readback must succeed");
    bytemuck::cast_slice(&bytes).to_vec()
}

fn execute(
    validated: &Validated<MatMulSpec>,
    lhs: &TestStorage,
    rhs: &TestStorage,
) -> Result<TestStorage, BackendError> {
    let context = ExecutionContext::new(TestBackend::new());
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
fn rank2_descriptor_execution_matches_the_legacy_path() {
    let lhs = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let rhs = storage(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let validated = lower(&[2, 3], &[3, 2]);

    let legacy = <TestBackend as TensorOps<TestBackend>>::matmul::<f32>(&lhs, &rhs).unwrap();
    let descriptor = execute(&validated, &lhs, &rhs).unwrap();

    assert_eq!(descriptor.shape(), legacy.shape());
    assert_eq!(read(&descriptor), read(&legacy));
    assert_eq!(read(&descriptor), vec![58., 64., 139., 154.]);
}

#[test]
fn the_binder_requires_exactly_two_inputs() {
    let lhs = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let validated = lower(&[2, 3], &[3, 2]);
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(&lhs)];

    let Err(error) = context.backend().execute(ExecutionRequest {
        operation: &validated,
        inputs: &inputs,
        context: &context,
    }) else {
        panic!("a one-operand matmul request must not execute");
    };

    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::MatMulExact,
            reason: "matmul expects exactly two tensor inputs"
        }
    ));
}

#[test]
fn the_binder_rejects_operands_that_disagree_with_the_descriptor() {
    // Both operands are individually valid WGPU storage, and their product is a
    // legal matmul, but the descriptor was lowered for different extents. The
    // binder must reject rather than execute the shapes it was handed.
    let lhs = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let rhs = storage(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let mismatched = lower(&[3, 2], &[2, 3]);

    let Err(error) = execute(&mismatched, &lhs, &rhs) else {
        panic!("a descriptor lowered for other extents must not execute");
    };

    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::MatMulExact,
            reason: "matmul lhs metadata does not match the validated descriptor"
        }
    ));
}

#[test]
fn the_binder_rejects_storage_belonging_to_another_backend() {
    // A CPU allocation carries CPU metadata. The WGPU executor must refuse it
    // rather than downcast it into a GPU buffer.
    #[cfg(feature = "cpu")]
    {
        use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
        use incin_core::prelude::Cpu;

        let foreign = CpuStorage::try_from_contiguous(
            CpuBuffer::F32(vec![1., 2., 3., 4., 5., 6.]),
            vec![2, 3],
        )
        .unwrap();
        let rhs = storage(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
        let validated = lower(&[2, 3], &[3, 2]);
        let context = ExecutionContext::new(TestBackend::new());
        let inputs = [
            TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&foreign),
            TensorHandle::from_storage::<TestBackend, f32, Local>(&rhs),
        ];

        let Err(error) = context.backend().execute(ExecutionRequest {
            operation: &validated,
            inputs: &inputs,
            context: &context,
        }) else {
            panic!("CPU storage must not bind to the WGPU executor");
        };

        assert!(matches!(
            error,
            BackendError::InvalidInput {
                operation: OperationKind::MatMulExact,
                reason: "matmul input is not WGPU storage"
            }
        ));
    }
}

fn lower_reshape_2x6_to_3x4() -> Validated<ReshapeSpec> {
    <ReshapeRule as ShapeRule<(s![2, 6], s![3, 4])>>::lower(
        &(field::<s![2, 6]>(&[2, 6]), field::<s![3, 4]>(&[3, 4])),
        (),
    )
    .expect("12 elements either way")
}

fn execute_reshape(
    validated: &Validated<ReshapeSpec>,
    input: &TestStorage,
) -> Result<TestStorage, BackendError> {
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(input)];
    context.backend().execute(ExecutionRequest {
        operation: validated,
        inputs: &inputs,
        context: &context,
    })
}

#[test]
fn reshape_descriptor_execution_matches_the_legacy_path() {
    let input = storage(
        &[2, 6],
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
    );
    let validated = lower_reshape_2x6_to_3x4();

    let legacy = <TestBackend as TensorOps<TestBackend>>::reshape::<f32>(&input, &[3, 4]).unwrap();
    let descriptor = execute_reshape(&validated, &input).unwrap();

    assert_eq!(descriptor.shape(), legacy.shape());
    assert_eq!(descriptor.shape().dims(), &[3, 4]);
    assert_eq!(read(&descriptor), read(&legacy));
}

#[test]
fn the_reshape_binder_requires_exactly_one_input() {
    let input = storage(&[2, 6], &[1.; 12]);
    let validated = lower_reshape_2x6_to_3x4();
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&input),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&input),
    ];

    let Err(error) = context.backend().execute(ExecutionRequest {
        operation: &validated,
        inputs: &inputs,
        context: &context,
    }) else {
        panic!("a two-operand reshape request must not execute");
    };

    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::ReshapeExact,
            reason: "reshape expects exactly one tensor input"
        }
    ));
}

#[test]
fn the_reshape_binder_rejects_an_operand_that_disagrees_with_the_descriptor() {
    // Twelve elements either way, but the descriptor was proved against `[2, 6]`.
    let input = storage(&[4, 3], &[1.; 12]);

    let Err(error) = execute_reshape(&lower_reshape_2x6_to_3x4(), &input) else {
        panic!("a descriptor lowered for other extents must not execute");
    };

    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::ReshapeExact,
            reason: "reshape input metadata does not match the validated descriptor"
        }
    ));
}

type Conv3x3 = Conv2dRule<U3, U3, U1, U1, U1>;
type ConvInput = s![1, 2, 4, 4];

fn lower_conv2d() -> Validated<Conv2dSpec> {
    <Conv3x3 as ShapeRule<ConvInput>>::lower(
        &field::<ConvInput>(&[1, 2, 4, 4]),
        Conv2dArgs::dense(3),
    )
    .expect("a 3x3 window with padding 1 fits a 4x4 input")
}

#[test]
fn conv2d_descriptor_execution_matches_the_legacy_path() {
    let input: Vec<f32> = (0..32).map(|value| value as f32 * 0.25 - 4.0).collect();
    let weight: Vec<f32> = (0..54).map(|value| value as f32 * 0.1 - 2.5).collect();
    let input = storage(&[1, 2, 4, 4], &input);
    let weight = storage(&[3, 2, 3, 3], &weight);
    let bias = storage(&[3], &[0.5, -0.25, 1.0]);
    let validated = lower_conv2d();

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

    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&input),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&weight),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&bias),
    ];
    let descriptor = context
        .backend()
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &inputs,
            context: &context,
        })
        .expect("a valid conv2d descriptor must execute");

    assert_eq!(descriptor.shape().dims(), &[1, 3, 4, 4]);
    let (descriptor, legacy) = (read(&descriptor), read(&legacy));
    assert_eq!(descriptor.len(), legacy.len());
    for (index, (left, right)) in descriptor.into_iter().zip(legacy).enumerate() {
        assert!(
            (left - right).abs() <= 1e-4,
            "value {index} differs: {left} versus {right}"
        );
    }
}

#[test]
fn the_conv2d_binder_rejects_a_weight_that_disagrees_with_the_descriptor() {
    let input = storage(&[1, 2, 4, 4], &[1.; 32]);
    // A legal filter bank, but for four output channels rather than the three
    // the descriptor was proved for.
    let weight = storage(&[4, 2, 3, 3], &[1.; 72]);
    let validated = lower_conv2d();

    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&input),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&weight),
    ];
    let Err(error) = context.backend().execute(ExecutionRequest {
        operation: &validated,
        inputs: &inputs,
        context: &context,
    }) else {
        panic!("a weight of the wrong output width must not execute");
    };

    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::Conv2dExact,
            reason: "conv2d weight metadata does not match the validated descriptor"
        }
    ));
}

#[test]
fn a_conv2d_bias_actually_reaches_every_output_element() {
    // WGPU's elementwise kernels do not broadcast, so the bias has to be
    // stretched to the output shape before it is added. Comparing against the
    // unbiased convolution pins that down without trusting the other path:
    // every element of output channel `c` must differ by exactly `bias[c]`.
    let input: Vec<f32> = (0..32).map(|value| value as f32 * 0.25 - 4.0).collect();
    let weight: Vec<f32> = (0..54).map(|value| value as f32 * 0.1 - 2.5).collect();
    let input = storage(&[1, 2, 4, 4], &input);
    let weight = storage(&[3, 2, 3, 3], &weight);
    let bias_values = [0.5_f32, -0.25, 1.0];
    let bias = storage(&[3], &bias_values);
    let validated = lower_conv2d();

    let run = |bias: Option<&TestStorage>| {
        let context = ExecutionContext::new(TestBackend::new());
        let mut inputs = vec![
            TensorHandle::from_storage::<TestBackend, f32, Local>(&input),
            TensorHandle::from_storage::<TestBackend, f32, Local>(&weight),
        ];
        if let Some(bias) = bias {
            inputs.push(TensorHandle::from_storage::<TestBackend, f32, Local>(bias));
        }
        let output = context
            .backend()
            .execute(ExecutionRequest {
                operation: &validated,
                inputs: &inputs,
                context: &context,
            })
            .expect("a valid conv2d descriptor must execute");
        read(&output)
    };

    let unbiased = run(None);
    let biased = run(Some(&bias));
    assert_eq!(unbiased.len(), 3 * 4 * 4);

    for (index, (with, without)) in biased.into_iter().zip(unbiased).enumerate() {
        let channel = index / (4 * 4);
        let expected = bias_values[channel];
        assert!(
            (with - without - expected).abs() <= 1e-4,
            "element {index} in channel {channel} gained {} rather than {expected}",
            with - without
        );
    }
}

fn execute_one<O: incin_core::exec::OperationSpec>(
    validated: &Validated<O>,
    input: &TestStorage,
) -> Result<TestStorage, BackendError>
where
    TestBackend: Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(input)];
    context.backend().execute(ExecutionRequest {
        operation: validated,
        inputs: &inputs,
        context: &context,
    })
}

#[test]
fn a_reduction_descriptor_routes_to_the_accumulation_it_names_on_gpu() {
    let input = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);

    for (op, expected) in [
        (ReduceOp::Sum, [6.0, 15.0]),
        (ReduceOp::Mean, [2.0, 5.0]),
        (ReduceOp::Max, [3.0, 6.0]),
        (ReduceOp::Min, [1.0, 4.0]),
        (ReduceOp::Prod, [6.0, 120.0]),
    ] {
        let validated =
            <ReduceAtRule<Next<Here>> as ShapeRule<R2>>::lower(&field::<R2>(&[2, 3]), op)
                .expect("axis 1 is in range");
        let output = execute_one(&validated, &input)
            .unwrap_or_else(|error| panic!("{op} must execute on wgpu: {error:?}"));

        assert_eq!(read(&output), expected.to_vec(), "{op} over axis 1");
    }
}

#[test]
fn a_pool_descriptor_routes_to_the_accumulation_it_names_on_gpu() {
    let values: Vec<f32> = (1..=16).map(|value| value as f32).collect();
    let input = storage(&[1, 1, 4, 4], &values);

    for (op, expected) in [
        (PoolOp::Max, vec![6.0, 8.0, 14.0, 16.0]),
        (PoolOp::Average, vec![3.5, 5.5, 11.5, 13.5]),
    ] {
        let validated = <Pool2dRule<U2, U2, U0, U1> as ShapeRule<s![1, 1, 4, 4]>>::lower(
            &field::<s![1, 1, 4, 4]>(&[1, 1, 4, 4]),
            op,
        )
        .expect("a 2x2 window strided by 2 tiles a 4x4 input");
        let output = execute_one(&validated, &input)
            .unwrap_or_else(|error| panic!("{op} pooling must execute on wgpu: {error:?}"));

        assert_eq!(output.shape.to_vec(), vec![1, 1, 2, 2]);
        assert_eq!(read(&output), expected, "{op} pooling over a 2x2 window");
    }
}
