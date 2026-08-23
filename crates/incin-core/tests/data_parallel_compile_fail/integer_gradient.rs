//! Integration coverage for `integer_gradient` on the documented public surface.
use incin_core::dist::{DataParallelPlanBuilder, GradientId, StreamId};

fn integer_gradient(builder: &mut DataParallelPlanBuilder<'_>) {
    builder
        .push_static::<u32>(
            GradientId::new(1).unwrap(),
            4,
            StreamId::default(),
        )
        .unwrap();
}

fn main() {}
