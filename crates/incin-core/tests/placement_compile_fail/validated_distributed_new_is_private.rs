//! Integration coverage for `main` on the documented public surface.
use incin_core::dist::{
    PlacementBuf, PlacementKind, PlacementTransition, ValidatedDistributed,
};
use incin_core::backend_authoring::operations::ShapeAttributes;
use incin_core::exec::{Descriptor, LogicalTensorMeta, op};
use incin_core::prelude::ShapeBuf;

fn main() {
    let shape = ShapeBuf::from_slice(&[4]);
    let operation = Descriptor::<op::ReshapeExact>::infer_runtime(
        ShapeAttributes { shape: vec![4] },
        vec![LogicalTensorMeta {
            shape: Some(shape.clone()),
            dtype: None,
            device: None,
        }],
    )
    .unwrap()
    .into_descriptor();

    // Only a checked distributed lowering rule may mint this proof.
    let _ = ValidatedDistributed::new(
        operation,
        shape.clone(),
        vec![shape],
        PlacementBuf::from([PlacementKind::Local]),
        PlacementKind::Local,
        PlacementTransition::Identity,
    );
}
