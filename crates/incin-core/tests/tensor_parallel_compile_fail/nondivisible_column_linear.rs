//! Integration coverage for `three_output_features` on the documented public surface.
use incin_core::dist::{StreamId, TensorParallelId, TensorParallelPlanBuilder};
use incin_core::typenum::{U1, U3};

fn three_output_features(builder: &mut TensorParallelPlanBuilder<'_>) {
    builder
        .push_column_static::<f32, U1, U3>(
            TensorParallelId::new(1).unwrap(),
            1,
            StreamId::default(),
        )
        .unwrap();
}

fn main() {}
