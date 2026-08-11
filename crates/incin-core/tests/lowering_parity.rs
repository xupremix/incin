//! `EXE-003`: exact descriptors infer the same geometry the frontend lowers.
//!
//! The old rule and legacy-spec coverage lived here before the catalog became
//! the canonical descriptor construction seam. These tests exercise that seam
//! directly with storage-free logical metadata, so each exact operation still
//! proves its input geometry, typed attributes, output geometry, and identity.

use incin_core::exec::catalog::{
    AxisAttributes, Conv2dAttributes, NoAttributes, Pool2dAttributes, ShapeAttributes,
};
use incin_core::exec::{
    op, CanonicalOperation, Descriptor, DescriptorError, ExecutionDescriptor, LogicalTensorMeta,
    ProofLevel,
};
use incin_core::prelude::{OperationKind, ShapeBuf};

fn tensor(shape: &[usize]) -> LogicalTensorMeta {
    LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(shape)),
        dtype: None,
        device: None,
    }
}

fn output<O: incin_core::exec::Operation>(
    invocation: &incin_core::exec::Validated<Descriptor<O>>,
) -> &[usize] {
    invocation
        .descriptor()
        .output_shape()
        .expect("concrete inputs must infer a concrete output")
        .dims()
}

#[test]
fn binary_exact_descriptors_infer_broadcast_geometry() {
    let inputs = vec![tensor(&[2, 1, 4]), tensor(&[3, 4])];

    for output_shape in [
        output(&Descriptor::<op::Add>::infer_runtime(NoAttributes, inputs.clone()).unwrap()),
        output(&Descriptor::<op::Sub>::infer_runtime(NoAttributes, inputs.clone()).unwrap()),
        output(&Descriptor::<op::Mul>::infer_runtime(NoAttributes, inputs.clone()).unwrap()),
        output(&Descriptor::<op::Div>::infer_runtime(NoAttributes, inputs.clone()).unwrap()),
    ] {
        assert_eq!(output_shape, &[2, 3, 4]);
    }
}

#[test]
fn matmul_exact_descriptor_infers_batched_output_geometry() {
    let descriptor = Descriptor::<op::MatMulExact>::infer_runtime(
        NoAttributes,
        vec![tensor(&[5, 2, 3]), tensor(&[1, 3, 4])],
    )
    .expect("batched 2x3 times 3x4");

    assert_eq!(output(&descriptor), &[5, 2, 4]);
    assert_eq!(
        descriptor.descriptor().operation(),
        OperationKind::MatMulExact
    );
    assert_eq!(descriptor.proof_level(), ProofLevel::Dynamic);
}

#[test]
fn reshape_exact_uses_its_explicit_shape_attribute() {
    let descriptor = Descriptor::<op::ReshapeExact>::infer_runtime(
        ShapeAttributes { shape: vec![3, 4] },
        vec![tensor(&[2, 6])],
    )
    .expect("12 elements may be reshaped to 3x4");

    assert_eq!(output(&descriptor), &[3, 4]);
    assert_eq!(descriptor.descriptor().attributes().shape, vec![3, 4]);
}

#[test]
fn reshape_exact_rejects_a_target_with_a_different_element_count() {
    let error = Descriptor::<op::ReshapeExact>::infer_runtime(
        ShapeAttributes { shape: vec![5, 5] },
        vec![tensor(&[2, 6])],
    )
    .expect_err("12 elements cannot become 25");

    assert!(matches!(
        error,
        DescriptorError::InvalidAttribute {
            operation: OperationKind::ReshapeExact,
            ..
        } | DescriptorError::Shape(_)
    ));
}

#[test]
fn axis_exact_descriptors_distinguish_dropped_and_kept_reductions() {
    let input = vec![tensor(&[2, 3, 4])];
    let dropped =
        Descriptor::<op::SumDim>::infer_runtime(AxisAttributes { axis: 1 }, input.clone())
            .expect("axis 1 is in range");
    let kept = Descriptor::<op::SumKeepDim>::infer_runtime(AxisAttributes { axis: 1 }, input)
        .expect("axis 1 is in range");

    assert_eq!(output(&dropped), &[2, 4]);
    assert_eq!(output(&kept), &[2, 1, 4]);
    assert_eq!(dropped.descriptor().attributes().axis, 1);
    assert_eq!(kept.descriptor().attributes().axis, 1);
}

#[test]
fn axis_attributes_reject_an_axis_outside_the_input_rank() {
    let error =
        Descriptor::<op::Softmax>::infer_runtime(AxisAttributes { axis: 2 }, vec![tensor(&[2, 3])])
            .expect_err("rank-two input has no axis 2");

    assert!(matches!(
        error,
        DescriptorError::InvalidAttribute {
            operation: OperationKind::Softmax,
            attribute: "axis",
            ..
        }
    ));
}

#[test]
fn conv2d_exact_infers_spatial_and_channel_geometry() {
    let descriptor = Descriptor::<op::Conv2dExact>::infer_runtime(
        Conv2dAttributes {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
            has_bias: false,
        },
        vec![tensor(&[1, 3, 8, 8]), tensor(&[16, 3, 3, 3])],
    )
    .expect("padded 3x3 convolution fits an 8x8 activation");

    assert_eq!(output(&descriptor), &[1, 16, 8, 8]);
    assert_eq!(descriptor.descriptor().attributes().groups, 1);
}

#[test]
fn conv2d_exact_rejects_groups_that_do_not_divide_input_channels() {
    let error = Descriptor::<op::Conv2dExact>::infer_runtime(
        Conv2dAttributes {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 2,
            has_bias: false,
        },
        vec![tensor(&[1, 3, 8, 8]), tensor(&[16, 2, 3, 3])],
    )
    .expect_err("two groups cannot divide three input channels");

    assert!(matches!(
        error,
        DescriptorError::InvalidAttribute {
            operation: OperationKind::Conv2dExact,
            attribute: "channels",
            ..
        }
    ));
}

#[test]
fn max_pool2d_exact_infers_window_geometry() {
    let descriptor = Descriptor::<op::MaxPool2d>::infer_runtime(
        Pool2dAttributes {
            kernel: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
            dilation: [1, 1],
        },
        vec![tensor(&[1, 3, 8, 8])],
    )
    .expect("a 2x2 window strided by two tiles an 8x8 activation");

    assert_eq!(output(&descriptor), &[1, 3, 4, 4]);
}

#[test]
fn max_pool2d_exact_reports_a_window_larger_than_the_input() {
    let error = Descriptor::<op::MaxPool2d>::infer_runtime(
        Pool2dAttributes {
            kernel: [4, 4],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
        },
        vec![tensor(&[1, 3, 2, 2])],
    )
    .expect_err("a 4x4 window does not fit a 2x2 activation");

    assert!(matches!(
        error,
        DescriptorError::InvalidAttribute {
            operation: OperationKind::MaxPool2d,
            attribute: "spatial",
            ..
        }
    ));
}

#[test]
fn exact_descriptors_keep_their_catalog_identity_and_element_count() {
    assert_eq!(
        <op::MaxPool2d as CanonicalOperation>::ID,
        OperationKind::MaxPool2d
    );
    assert_eq!(
        <op::ReshapeExact as CanonicalOperation>::ID,
        OperationKind::ReshapeExact
    );

    let descriptor =
        Descriptor::<op::Add>::infer_runtime(NoAttributes, vec![tensor(&[3, 4]), tensor(&[1, 4])])
            .expect("3x4 against 1x4 broadcasts");

    assert_eq!(
        descriptor
            .descriptor()
            .output_shape()
            .unwrap()
            .dims()
            .iter()
            .product::<usize>(),
        12
    );
}
