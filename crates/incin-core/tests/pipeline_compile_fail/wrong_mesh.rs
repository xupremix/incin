//! Integration coverage for `wrong_mesh` on the documented public surface.
use incin_core::dist::{
    ActivationCheckpoint, GPipe, PipelineBoundaryId, PipelinePlanBuilder, StreamId,
    TwoRankDataParallel,
};
use incin_core::dist::mesh::DeviceMesh;
use incin_core::typenum::{U1, U2};

fn wrong_mesh(mesh: &DeviceMesh<TwoRankDataParallel>) {
    PipelinePlanBuilder::build_static::<f32, (U2,), U1, GPipe>(
        mesh,
        0,
        PipelineBoundaryId::new(1).unwrap(),
        ActivationCheckpoint::Keep,
        StreamId::default(),
    )
    .unwrap();
}

fn main() {}
