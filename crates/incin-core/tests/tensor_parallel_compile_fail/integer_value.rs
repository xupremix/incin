//! Integration coverage for `integer_tensor` on the documented public surface.
use incin_core::dist::{StreamId, TensorParallelId, TensorParallelPlanBuilder};
use incin_core::typenum::{U1, U4};

fn integer_tensor(builder: &mut TensorParallelPlanBuilder<'_>) {
    builder
        .push_column_static::<u32, U1, U4>(
            TensorParallelId::new(1).unwrap(),
            1,
            StreamId::default(),
        )
        .unwrap();
}

fn main() {}
