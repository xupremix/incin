//! Integration coverage for `too_many_microbatches` on the documented public surface.
use incin_core::dist::{
    ActivationCheckpoint, GPipe, PipelineBoundaryId, PipelinePlanBuilder, StreamId,
    TwoRankPipeline,
};
use incin_core::dist::mesh::DeviceMesh;
use incin_core::typenum::{U2, U4294967296};

fn too_many_microbatches(mesh: &DeviceMesh<TwoRankPipeline>) {
    PipelinePlanBuilder::build_static::<f32, (U2,), U4294967296, GPipe>(
        mesh,
        0,
        PipelineBoundaryId::new(1).unwrap(),
        ActivationCheckpoint::Keep,
        StreamId::default(),
    )
    .unwrap();
}

fn main() {}
