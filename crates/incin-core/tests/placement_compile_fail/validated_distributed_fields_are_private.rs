use incin_core::dist::{
    PlacementBuf, PlacementKind, PlacementTransition, ValidatedDistributed,
};
use incin_core::exec::ReshapeSpec;
use incin_core::prelude::ShapeBuf;

fn main() {
    let shape = ShapeBuf::from_slice(&[4]);
    let operation = ReshapeSpec::new(&shape, &shape).unwrap();

    // Public fields would let a transport fabricate a proof without a rule.
    let _ = ValidatedDistributed {
        operation,
        global_shape: shape.clone(),
        local_shapes: vec![shape],
        input_placements: PlacementBuf::from([PlacementKind::Local]),
        output_placement: PlacementKind::Local,
        transition: PlacementTransition::Identity,
    };
}
