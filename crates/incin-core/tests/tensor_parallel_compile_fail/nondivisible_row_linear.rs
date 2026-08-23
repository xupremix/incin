//! Integration coverage for `three_input_features` on the documented public surface.
use incin_core::dist::{StreamId, TensorParallelId, TensorParallelPlanBuilder};
use incin_core::typenum::U3;

fn three_input_features(builder: &mut TensorParallelPlanBuilder<'_>) {
    builder
        .push_row_static::<f32, U3>(
            TensorParallelId::new(1).unwrap(),
            1,
            StreamId::default(),
        )
        .unwrap();
}

fn main() {}
