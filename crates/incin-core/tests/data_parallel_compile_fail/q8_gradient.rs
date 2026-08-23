//! Integration coverage for `quantized_gradient` on the documented public surface.
use incin_core::dist::{DataParallelPlanBuilder, GradientId, StreamId};
use incin_core::prelude::Q8_0;

fn quantized_gradient(builder: &mut DataParallelPlanBuilder<'_>) {
    builder
        .push_static::<Q8_0>(
            GradientId::new(1).unwrap(),
            32,
            StreamId::default(),
        )
        .unwrap();
}

fn main() {}
