//! `EXE-008`: the descriptor execution contract on real WGPU hardware.
//!
//! `EXE-007` proved the contract for CPU. These cases prove the same sealed
//! `Validated<MatMulExact>` binds and executes against a second, genuinely
//! different backend - so the descriptor path is a shared contract rather than
//! a CPU-shaped one - and that the binder's rejections are enforced per backend
//! rather than inherited from the CPU implementation.
#![cfg(feature = "wgpu")]

extern crate incin_core as incin;

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::operations::ShapeAttributes;
use incin_core::backend_authoring::{Execute, ExecutionRequest, HostInterop};
use incin_core::exec::catalog::{
    AttributeContract, AxisAttributes, Conv2dAttributes, Pool2dAttributes,
};
use incin_core::exec::{
    CanonicalOperation, Descriptor, ExecutionContext, LogicalTensorMeta, TensorHandle, Validated,
    op,
};
use incin_core::prelude::{
    BackendError, DTypeId, DeviceId, Local, OperationKind, Shape, ShapeBuf, WgpuN, s,
};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as incin_core::backend_authoring::StorageBackend>::Storage<f32>;

fn field<S: Shape>(dims: &[usize]) -> ShapeBuf {
    S::try_from_dims(dims).expect("test dimensions must match the shape type")
}

fn lower(lhs: &[usize], rhs: &[usize]) -> Validated<Descriptor<op::MatMulExact>> {
    Descriptor::<op::MatMulExact>::infer_runtime(
        incin_core::backend_authoring::operations::NoAttributes,
        vec![
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(lhs)),
                dtype: None,
                device: None,
            },
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(rhs)),
                dtype: None,
                device: None,
            },
        ],
    )
    .expect("test operands must be valid matmul shapes")
}

fn has_wgpu() -> bool {
    TestBackend::from_bytes::<f32>(
        bytemuck::cast_slice(&[1.0f32]),
        &[1],
        DTypeId::F32.into(),
        &DeviceId::wgpu(0),
    )
    .is_ok()
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
    validated: &Validated<Descriptor<op::MatMulExact>>,
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
        payload: None,
    })
}

#[test]
fn rank2_descriptor_execution_produces_the_arithmetic_product() {
    if !has_wgpu() {
        return;
    }
    let lhs = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let rhs = storage(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let validated = lower(&[2, 3], &[3, 2]);

    let descriptor = execute(&validated, &lhs, &rhs).unwrap();

    assert_eq!(descriptor.shape().dims(), &[2, 2]);
    assert_eq!(read(&descriptor), vec![58., 64., 139., 154.]);
}

#[test]
fn the_binder_requires_exactly_two_inputs() {
    if !has_wgpu() {
        return;
    }
    let lhs = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let validated = lower(&[2, 3], &[3, 2]);
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(&lhs)];

    let Err(error) = context.backend().execute(ExecutionRequest {
        operation: &validated,
        inputs: &inputs,
        context: &context,
        payload: None,
    }) else {
        panic!("a one-operand matmul request must not execute");
    };

    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::MatMulExact,
            reason: "matmul expects 2 inputs"
        }
    ));
}

#[test]
fn the_binder_rejects_operands_that_disagree_with_the_descriptor() {
    if !has_wgpu() {
        return;
    }
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
    if !has_wgpu() {
        return;
    }
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
            payload: None,
        }) else {
            panic!("CPU storage must not bind to the WGPU executor");
        };

        assert!(matches!(
            error,
            BackendError::InvalidInput {
                operation: OperationKind::MatMulExact,
                reason: "lhs is not WGPU storage"
            }
        ));
    }
}

fn lower_reshape_2x6_to_3x4() -> Validated<Descriptor<op::ReshapeExact>> {
    Descriptor::<op::ReshapeExact>::infer_runtime(
        ShapeAttributes { shape: vec![3, 4] },
        vec![LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&[2, 6])),
            dtype: None,
            device: None,
        }],
    )
    .expect("12 elements either way")
}

fn execute_reshape(
    validated: &Validated<Descriptor<op::ReshapeExact>>,
    input: &TestStorage,
) -> Result<TestStorage, BackendError> {
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(input)];
    context.backend().execute(ExecutionRequest {
        operation: validated,
        inputs: &inputs,
        context: &context,
        payload: None,
    })
}

#[test]
fn reshape_descriptor_execution_rewrites_the_shape_and_keeps_row_major_order() {
    if !has_wgpu() {
        return;
    }
    let values = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let input = storage(&[2, 6], &values);
    let validated = lower_reshape_2x6_to_3x4();

    let descriptor = execute_reshape(&validated, &input).unwrap();

    assert_eq!(descriptor.shape().dims(), &[3, 4]);
    // Reshape is a reinterpretation, not a permutation: the row-major reading
    // order of the elements has to survive it untouched.
    assert_eq!(read(&descriptor), values.to_vec());
}

#[test]
fn the_reshape_binder_requires_exactly_one_input() {
    if !has_wgpu() {
        return;
    }
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
        payload: None,
    }) else {
        panic!("a two-operand reshape request must not execute");
    };

    assert!(matches!(
        error,
        BackendError::InvalidInput {
            operation: OperationKind::ReshapeExact,
            reason: "reshape expects 1 input"
        }
    ));
}

#[test]
fn the_reshape_binder_rejects_an_operand_that_disagree_with_the_descriptor() {
    if !has_wgpu() {
        return;
    }
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

type ConvInput = s![1, 2, 4, 4];

fn lower_conv2d() -> Validated<Descriptor<op::Conv2dExact>> {
    Descriptor::<op::Conv2dExact>::infer_runtime(
        Conv2dAttributes {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
            has_bias: true,
        },
        vec![
            LogicalTensorMeta {
                shape: Some(field::<ConvInput>(&[1, 2, 4, 4])),
                dtype: None,
                device: None,
            },
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[3, 2, 3, 3])),
                dtype: None,
                device: None,
            },
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[3])),
                dtype: None,
                device: None,
            },
        ],
    )
    .expect("a 3x3 window with padding 1 fits a 4x4 input")
}

/// A direct convolution on the host, written from the definition.
///
/// The point of this test is to check the GPU kernel against arithmetic, so
/// the reference deliberately shares nothing with the backend: it indexes the
/// operands itself and accumulates in the obvious order. Shapes are `NCHW`
/// input and `OIHW` weight, with a single group and unit stride and dilation.
#[allow(clippy::needless_range_loop)]
fn reference_conv2d(
    input: &[f32],
    input_shape: [usize; 4],
    weight: &[f32],
    weight_shape: [usize; 4],
    bias: &[f32],
    padding: usize,
) -> Vec<f32> {
    let [batch, in_channels, in_height, in_width] = input_shape;
    let [
        out_channels,
        weight_in_channels,
        kernel_height,
        kernel_width,
    ] = weight_shape;
    assert_eq!(in_channels, weight_in_channels);
    let out_height = in_height + 2 * padding - kernel_height + 1;
    let out_width = in_width + 2 * padding - kernel_width + 1;

    let mut output = Vec::with_capacity(batch * out_channels * out_height * out_width);
    for n in 0..batch {
        for co in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = bias[co];
                    for ci in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = (oh + kh) as isize - padding as isize;
                                let iw = (ow + kw) as isize - padding as isize;
                                if ih < 0
                                    || iw < 0
                                    || ih >= in_height as isize
                                    || iw >= in_width as isize
                                {
                                    continue;
                                }
                                let input_index =
                                    ((n * in_channels + ci) * in_height + ih as usize) * in_width
                                        + iw as usize;
                                let weight_index = ((co * in_channels + ci) * kernel_height + kh)
                                    * kernel_width
                                    + kw;
                                sum += input[input_index] * weight[weight_index];
                            }
                        }
                    }
                    output.push(sum);
                }
            }
        }
    }
    output
}

#[test]
fn conv2d_descriptor_execution_matches_a_direct_convolution() {
    if !has_wgpu() {
        return;
    }
    let input_values: Vec<f32> = (0..32).map(|value| value as f32 * 0.25 - 4.0).collect();
    let weight_values: Vec<f32> = (0..54).map(|value| value as f32 * 0.1 - 2.5).collect();
    let bias_values = [0.5_f32, -0.25, 1.0];
    let input = storage(&[1, 2, 4, 4], &input_values);
    let weight = storage(&[3, 2, 3, 3], &weight_values);
    let bias = storage(&[3], &bias_values);
    let validated = lower_conv2d();

    let expected = reference_conv2d(
        &input_values,
        [1, 2, 4, 4],
        &weight_values,
        [3, 2, 3, 3],
        &bias_values,
        1,
    );

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
            payload: None,
        })
        .expect("a valid conv2d descriptor must execute");

    assert_eq!(descriptor.shape().dims(), &[1, 3, 4, 4]);
    let produced = read(&descriptor);
    assert_eq!(produced.len(), expected.len());
    for (index, (left, right)) in produced.into_iter().zip(expected).enumerate() {
        assert!(
            (left - right).abs() <= 1e-4,
            "value {index} differs: {left} versus {right}"
        );
    }
}

#[test]
fn the_conv2d_binder_rejects_a_weight_that_disagrees_with_the_descriptor() {
    if !has_wgpu() {
        return;
    }
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
        payload: None,
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
    if !has_wgpu() {
        return;
    }
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
                payload: None,
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

fn execute_one<O>(
    validated: &Validated<Descriptor<O>>,
    input: &TestStorage,
) -> Result<TestStorage, BackendError>
where
    O: CanonicalOperation,
    O::Attributes: AttributeContract,
    TestBackend: Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(input)];
    context.backend().execute(ExecutionRequest {
        operation: validated,
        inputs: &inputs,
        context: &context,
        payload: None,
    })
}

#[test]
fn a_reduction_descriptor_routes_to_the_accumulation_it_names_on_gpu() {
    if !has_wgpu() {
        return;
    }
    let input = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);

    macro_rules! check {
        ($marker:ty, $expected:expr) => {{
            let validated = Descriptor::<$marker>::infer_runtime(
                AxisAttributes { axis: 1 },
                vec![LogicalTensorMeta {
                    shape: Some(ShapeBuf::from_slice(&[2, 3])),
                    dtype: None,
                    device: None,
                }],
            )
            .expect("axis 1 is in range");
            let output = execute_one(&validated, &input).expect("reduction must execute on wgpu");
            assert_eq!(read(&output), $expected);
        }};
    }
    check!(op::SumDim, vec![6., 15.]);
    check!(op::MeanDim, vec![2., 5.]);
    check!(op::MaxDim, vec![3., 6.]);
    check!(op::MinDim, vec![1., 4.]);
    check!(op::ProdDim, vec![6., 120.]);
}

#[test]
fn a_pool_descriptor_routes_to_the_accumulation_it_names_on_gpu() {
    if !has_wgpu() {
        return;
    }
    let values: Vec<f32> = (1..=16).map(|value| value as f32).collect();
    let input = storage(&[1, 1, 4, 4], &values);

    macro_rules! check {
        ($marker:ty, $expected:expr) => {{
            let validated = Descriptor::<$marker>::infer_runtime(
                Pool2dAttributes {
                    kernel: [2, 2],
                    stride: [2, 2],
                    padding: [0, 0],
                    dilation: [1, 1],
                },
                vec![LogicalTensorMeta {
                    shape: Some(ShapeBuf::from_slice(&[1, 1, 4, 4])),
                    dtype: None,
                    device: None,
                }],
            )
            .expect("a 2x2 window strided by 2 tiles a 4x4 input");
            let output = execute_one(&validated, &input).expect("pooling must execute on wgpu");
            assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
            assert_eq!(read(&output), $expected);
        }};
    }
    check!(op::MaxPool2d, vec![6., 8., 14., 16.]);
    let validated = Descriptor::<op::AvgPool2d>::infer_runtime(
        incin_core::exec::catalog::AvgPool2dAttributes {
            kernel: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
        },
        vec![LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&[1, 1, 4, 4])),
            dtype: None,
            device: None,
        }],
    )
    .unwrap();
    let output = execute_one(&validated, &input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
    assert_eq!(read(&output), vec![3.5, 5.5, 11.5, 13.5]);
}

/// Every unary activation with a WGSL kernel is reachable through canonical
/// dispatch, not merely through a direct `Execute` call.
///
/// This is the case that was missing when `wgpu/executor.rs` implemented
/// thirteen unary activations that `WGPU_CAPABILITIES` never advertised. A
/// direct `backend().execute(..)` - which is what every other case in this file
/// does - would have passed the whole time, because it never asks the
/// capability registry anything. `dispatch::execute` does ask, so it is the
/// only call that can tell an implemented operation from a reachable one.
#[test]
fn every_advertised_unary_activation_is_reachable_through_canonical_dispatch() {
    if !has_wgpu() {
        return;
    }
    use incin_core::backend_authoring::execute;
    use incin_core::backend_authoring::operations::NoAttributes;

    let context = ExecutionContext::new(TestBackend::new());

    // The operands are macro arguments rather than outer locals because
    // `macro_rules` hygiene binds an identifier in a macro body at the
    // definition site: a later `let values = ..` would not be the `values` this
    // expands to, and the second group would have been checked against the
    // first group's numbers.
    macro_rules! check {
        ($marker:ty, $values:expr, $reference:expr) => {{
            let values: [f32; 5] = $values;
            let input = storage(&[5], &values);
            let inputs = [TensorHandle::from_storage::<TestBackend, f32, Local>(
                &input,
            )];
            let output = execute::<$marker, TestBackend>(&context, NoAttributes, &inputs)
                .unwrap_or_else(|error| {
                    panic!(
                        "{} must be reachable through canonical dispatch: {error}",
                        <$marker as CanonicalOperation>::ID.name()
                    )
                });
            let reference: fn(f32) -> f32 = $reference;
            for (index, (actual, source)) in read(&output).into_iter().zip(values).enumerate() {
                let expected = reference(source);
                assert!(
                    (actual - expected).abs() <= 1e-4,
                    "{} element {index}: got {actual}, expected {expected}",
                    <$marker as CanonicalOperation>::ID.name()
                );
            }
        }};
    }

    const SIGNED: [f32; 5] = [-2.0, -0.5, 0.0, 0.5, 2.0];
    // `sqrt` and `log` are undefined on the negative half of `SIGNED`, so they
    // get a strictly positive operand rather than a tolerance wide enough to
    // hide a wrong answer.
    const POSITIVE: [f32; 5] = [0.25, 1.0, 2.0, 4.0, 9.0];

    check!(op::Relu, SIGNED, |x| x.max(0.0));
    check!(op::Step, SIGNED, |x| if x > 0.0 { 1.0 } else { 0.0 });
    check!(op::Abs, SIGNED, f32::abs);
    check!(op::Neg, SIGNED, |x| -x);
    check!(op::Tanh, SIGNED, f32::tanh);
    check!(op::Sigmoid, SIGNED, |x| 1.0 / (1.0 + (-x).exp()));
    check!(op::Exp, SIGNED, f32::exp);
    check!(op::Swish, SIGNED, |x| x / (1.0 + (-x).exp()));
    check!(op::Elu, SIGNED, |x| if x >= 0.0 {
        x
    } else {
        x.exp() - 1.0
    });
    check!(op::Mish, SIGNED, |x: f32| x * (1.0 + x.exp()).ln().tanh());
    check!(op::Gelu, SIGNED, |x: f32| {
        let inner = (2.0f32 / core::f32::consts::PI).sqrt() * (x + 0.044_715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    });
    check!(op::Sqrt, POSITIVE, f32::sqrt);
    check!(op::Log, POSITIVE, f32::ln);
}
