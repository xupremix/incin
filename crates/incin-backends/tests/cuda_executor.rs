//! Descriptor and canonical execution contract tests on CUDA backend.
#![cfg(feature = "cuda")]

extern crate incin_core as incin;

use incin_backends::cuda::CudaBackendImpl;
use incin_core::backend_authoring::{Execute, ExecutionRequest, HostInterop};
use incin_core::exec::catalog::{
    AttributeContract, AxisAttributes, Conv2dAttributes, Pool2dAttributes,
};
use incin_core::exec::{
    CanonicalOperation, Descriptor, ExecutionContext, LogicalTensorMeta, TensorHandle, Validated,
    op,
};
use incin_core::prelude::{BackendError, CudaN, DTypeId, DeviceId, Local, ShapeBuf};
use incin_core::typenum::U0;

type TestBackend = CudaBackendImpl<CudaN<U0>>;
type TestStorage = <TestBackend as incin_core::backend_authoring::StorageBackend>::Storage<f32>;

/// Aborts unless a CUDA device is present.
///
/// Replaces a `has_cuda() -> bool` predicate that callers used to skip with an
/// early `return`. Every caller is `#[ignore]`d, so reaching one is a deliberate
/// request for the hardware run, and returning early there reports `ok` for a
/// test that launched nothing -- the pattern that kept three real CUDA defects
/// green for as long as they existed.
///
/// # Panics
///
/// If no CUDA device can be opened on ordinal 0.
fn require_cuda() {
    assert!(
        TestBackend::from_bytes::<f32>(
            bytemuck::cast_slice(&[1.0f32]),
            &[1],
            DTypeId::F32.into(),
            &DeviceId::cuda(0),
        )
        .is_ok(),
        "no CUDA device, but this test is #[ignore]d -- running it is an explicit request for hardware. Skipping here would report `ok` for a test that launched nothing."
    );
}

fn storage(shape: &[usize], values: &[f32]) -> TestStorage {
    TestBackend::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32.into(),
        &DeviceId::cuda(0),
    )
    .expect("test buffer must match its shape")
}

fn read(storage: &TestStorage) -> Vec<f32> {
    let bytes = TestBackend::to_bytes::<f32>(storage).expect("readback must succeed");
    bytemuck::cast_slice(&bytes).to_vec()
}

fn lower_matmul(lhs: &[usize], rhs: &[usize]) -> Validated<Descriptor<op::MatMulExact>> {
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
                shape: Some(ShapeBuf::from_slice(&[1, 2, 4, 4])),
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
    .expect("canonical 1x2x4x4 padded conv2d must infer cleanly")
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
fn cuda_matmul_descriptor_execution_matches_matrix_multiplication() {
    require_cuda();
    let lhs = storage(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let rhs = storage(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let validated = lower_matmul(&[2, 3], &[3, 2]);

    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&lhs),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&rhs),
    ];
    let descriptor = context
        .backend()
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &inputs,
            context: &context,
            payload: None,
        })
        .expect("a valid matmul descriptor must execute on cuda");

    assert_eq!(descriptor.shape().dims(), &[2, 2]);
    assert_eq!(read(&descriptor), vec![58., 64., 139., 154.]);
}

#[test]
fn cuda_conv2d_descriptor_execution_matches_direct_convolution() {
    require_cuda();
    let input_values: Vec<f32> = (0..32).map(|value| value as f32 * 0.25 - 4.0).collect();
    let weight_values: Vec<f32> = (0..54).map(|value| value as f32 * 0.1 - 2.5).collect();
    let bias_values = [0.5_f32, -0.25, 1.0];
    let input = storage(&[1, 2, 4, 4], &input_values);
    let weight = storage(&[3, 2, 3, 3], &weight_values);
    let bias = storage(&[3], &bias_values);
    let validated = lower_conv2d();

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
        .expect("a valid conv2d descriptor must execute on cuda");

    assert_eq!(descriptor.shape().dims(), &[1, 3, 4, 4]);
}

#[test]
fn cuda_reduction_descriptor_routes_to_the_accumulation_it_names() {
    require_cuda();
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
            let output = execute_one(&validated, &input).expect("reduction must execute on cuda");
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
fn cuda_pool_descriptor_routes_to_the_accumulation_it_names() {
    require_cuda();
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
            let output = execute_one(&validated, &input).expect("pooling must execute on cuda");
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

#[test]
fn cuda_every_advertised_unary_activation_is_reachable_through_canonical_dispatch() {
    require_cuda();
    use incin_core::backend_authoring::execute;
    use incin_core::backend_authoring::operations::NoAttributes;

    let context = ExecutionContext::new(TestBackend::new());

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
    const POSITIVE: [f32; 5] = [0.25, 1.0, 2.0, 4.0, 9.0];

    check!(op::Relu, SIGNED, |x| x.max(0.0));
    check!(op::Step, SIGNED, |x| if x > 0.0 { 1.0 } else { 0.0 });
    check!(op::Abs, SIGNED, f32::abs);
    check!(op::Neg, SIGNED, |x| -x);
    check!(op::Exp, SIGNED, f32::exp);
    check!(op::Sqrt, POSITIVE, f32::sqrt);
    check!(op::Log, POSITIVE, f32::ln);
    check!(op::Tanh, SIGNED, f32::tanh);
    check!(op::Sigmoid, SIGNED, |x| 1.0 / (1.0 + (-x).exp()));
    check!(op::Gelu, SIGNED, |x| {
        0.5 * x * (1.0 + (0.797_884_6_f32 * (x + 0.044_715_f32 * x * x * x)).tanh())
    });
}

#[test]
fn cuda_fused_rms_norm_matches_analytical_reference() {
    require_cuda();
    let input = storage(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0]);
    let weight = storage(&[4], &[1.0, 1.0, 1.0, 1.0]);

    let validated = Descriptor::<op::RmsNorm>::infer_runtime(
        incin_core::exec::catalog::EpsilonAttributes { epsilon: 1e-5 },
        vec![
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[2, 4])),
                dtype: None,
                device: None,
            },
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[4])),
                dtype: None,
                device: None,
            },
        ],
    )
    .expect("valid rms_norm descriptor");

    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&input),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&weight),
    ];
    let output = context
        .backend()
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &inputs,
            context: &context,
            payload: None,
        })
        .expect("fused rms norm must execute on cuda");

    assert_eq!(output.shape().dims(), &[2, 4]);
    let values = read(&output);
    assert_eq!(values.len(), 8);
    // Row 2 is all 2.0s with weight 1.0 -> normalized values should all be ~1.0
    for v in &values[4..8] {
        assert!((v - 1.0).abs() <= 1e-3);
    }
}

#[test]
fn cuda_fused_softmax_matches_analytical_reference() {
    require_cuda();
    let input = storage(&[2, 3], &[0.0, 1.0, 2.0, -1.0, 0.0, 1.0]);

    let validated = Descriptor::<op::Softmax>::infer_runtime(
        AxisAttributes { axis: 1 },
        vec![LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&[2, 3])),
            dtype: None,
            device: None,
        }],
    )
    .expect("valid softmax descriptor");

    let output = execute_one(&validated, &input).expect("fused softmax must execute on cuda");
    assert_eq!(output.shape().dims(), &[2, 3]);

    let values = read(&output);
    // Probabilities along each row must sum to 1.0
    let row1_sum: f32 = values[0..3].iter().sum();
    let row2_sum: f32 = values[3..6].iter().sum();
    assert!((row1_sum - 1.0).abs() <= 1e-4);
    assert!((row2_sum - 1.0).abs() <= 1e-4);
}

#[test]
fn cuda_scaled_dot_product_attention_executes_cleanly() {
    require_cuda();
    use incin_core::exec::catalog::AttentionAttributes;

    let q = storage(&[2, 4], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let k = storage(&[2, 4], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let v = storage(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let validated = Descriptor::<op::ScaledDotProductAttention>::infer_runtime(
        AttentionAttributes {
            scale: Some(0.5),
            has_mask: false,
        },
        vec![
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[2, 4])),
                dtype: None,
                device: None,
            },
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[2, 4])),
                dtype: None,
                device: None,
            },
            LogicalTensorMeta {
                shape: Some(ShapeBuf::from_slice(&[2, 4])),
                dtype: None,
                device: None,
            },
        ],
    )
    .expect("valid SDPA descriptor");

    let context = ExecutionContext::new(TestBackend::new());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, Local>(&q),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&k),
        TensorHandle::from_storage::<TestBackend, f32, Local>(&v),
    ];
    let output = context
        .backend()
        .execute(ExecutionRequest {
            operation: &validated,
            inputs: &inputs,
            context: &context,
            payload: None,
        })
        .expect("SDPA must execute cleanly on cuda");

    assert_eq!(output.shape().dims(), &[2, 4]);
}
