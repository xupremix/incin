//! Canonical exact descriptor contract tests.
//!
//! These tests pin the catalog identity, metadata derivation, and high-rank
//! axis representation used by backend execution.

use incin_core::backend_authoring::operations::ShapeAttributes;
use incin_core::exec::catalog::{
    AxisAttributes, Conv2dAttributes, Descriptor, LogicalTensorMeta, NoAttributes,
    Pool2dAttributes, op,
};
use incin_core::exec::{AxisSet, DescriptorSchemaVersion, ExecutionDescriptor, OPERATION_CATALOG};
use incin_core::prelude::{OperationKind, ShapeBuf};

fn input(dims: &[usize]) -> LogicalTensorMeta {
    LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(dims)),
        dtype: None,
        device: None,
    }
}

#[test]
fn schema_version_is_explicit_and_compatible() {
    let current = DescriptorSchemaVersion::CURRENT;
    assert!(current.is_compatible_with(current));
    assert!(!current.is_compatible_with(DescriptorSchemaVersion::new(current.get() + 1)));
    assert_eq!(current.to_string(), "v3");
}

#[test]
fn catalog_contains_unique_exact_operation_identities() {
    for (index, left) in OPERATION_CATALOG.iter().enumerate() {
        assert!(!left.name.is_empty());
        for right in &OPERATION_CATALOG[index + 1..] {
            assert_ne!(left.operation, right.operation);
        }
    }
}

#[test]
fn exact_binary_descriptor_derives_broadcast_output() {
    let descriptor =
        Descriptor::<op::Add>::infer_runtime(NoAttributes, vec![input(&[4, 1, 3]), input(&[5, 3])])
            .unwrap()
            .into_descriptor();
    assert_eq!(descriptor.operation(), OperationKind::Add);
    assert_eq!(descriptor.output_shape().unwrap().dims(), &[4, 5, 3]);
}

#[test]
fn exact_matmul_descriptor_derives_batched_output() {
    let descriptor = Descriptor::<op::MatMulExact>::infer_runtime(
        NoAttributes,
        vec![input(&[2, 4, 3, 5]), input(&[2, 4, 5, 6])],
    )
    .unwrap()
    .into_descriptor();
    assert_eq!(descriptor.operation(), OperationKind::MatMulExact);
    assert_eq!(descriptor.output_shape().unwrap().dims(), &[2, 4, 3, 6]);
}

#[test]
fn exact_reduction_descriptor_handles_axis_seventy() {
    let descriptor =
        Descriptor::<op::SumDim>::infer_runtime(AxisAttributes { axis: 70 }, vec![input(&[1; 71])])
            .unwrap()
            .into_descriptor();
    assert_eq!(descriptor.output_shape().unwrap().rank(), 70);
}

#[test]
fn exact_keep_reduction_preserves_rank() {
    let descriptor = Descriptor::<op::SumKeepDim>::infer_runtime(
        AxisAttributes { axis: 1 },
        vec![input(&[2, 3, 4])],
    )
    .unwrap()
    .into_descriptor();
    assert_eq!(descriptor.output_shape().unwrap().dims(), &[2, 1, 4]);
}

#[test]
fn exact_reshape_descriptor_checks_element_count() {
    let descriptor = Descriptor::<op::ReshapeExact>::infer_runtime(
        ShapeAttributes { shape: vec![4] },
        vec![input(&[2, 2])],
    )
    .unwrap()
    .into_descriptor();
    assert_eq!(descriptor.output_shape().unwrap().dims(), &[4]);
}

#[test]
fn exact_conv_and_pool_descriptors_derive_spatial_geometry() {
    let conv = Descriptor::<op::Conv2dExact>::infer_runtime(
        Conv2dAttributes {
            stride: [1, 1],
            padding: [1, 1],
            dilation: [1, 1],
            groups: 1,
            has_bias: false,
        },
        vec![input(&[1, 3, 8, 8]), input(&[8, 3, 3, 3])],
    )
    .unwrap()
    .into_descriptor();
    assert_eq!(conv.output_shape().unwrap().dims(), &[1, 8, 8, 8]);

    let pool = Descriptor::<op::MaxPool2d>::infer_runtime(
        Pool2dAttributes {
            kernel: [2, 2],
            stride: [2, 2],
            padding: [0, 0],
            dilation: [1, 1],
        },
        vec![input(&[1, 3, 8, 8])],
    )
    .unwrap()
    .into_descriptor();
    assert_eq!(pool.output_shape().unwrap().dims(), &[1, 3, 4, 4]);
}

#[test]
fn axis_set_spills_without_a_semantic_rank_ceiling() {
    let set = AxisSet::EMPTY.insert(70);
    assert!(set.contains(70));
    assert_eq!(set.count(), 1);
}

#[test]
fn axis_set_mixes_inline_and_spilled_axes_without_losing_semantics() {
    let set = [0, 64, 65, 129, 65]
        .into_iter()
        .fold(AxisSet::EMPTY, AxisSet::insert);
    assert_eq!(set.axes().collect::<Vec<_>>(), vec![0, 64, 65, 129]);
    assert_eq!(set.count(), 4);
    assert!(set.contains(0));
    assert!(set.contains(64));
    assert!(set.contains(129));
    assert!(!set.contains(128));
    assert!(!set.is_contiguous_run());

    let contiguous = [64, 65, 66]
        .into_iter()
        .fold(AxisSet::EMPTY, AxisSet::insert);
    assert!(contiguous.is_contiguous_run());
}
