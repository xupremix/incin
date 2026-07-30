use incin_core::dist::{
    PlacementBuf, PlacementKind, PlacementTransition, ValidatedDistributed,
};
use incin_core::exec::ReshapeSpec;
use incin_core::prelude::ShapeBuf;

fn main() {
    let shape = ShapeBuf::from_slice(&[4]);
    let operation = ReshapeSpec::new(&shape, &shape).unwrap();

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
