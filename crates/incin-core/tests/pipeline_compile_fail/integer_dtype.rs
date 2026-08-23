use incin_core::dist::{
    ActivationCheckpoint, GPipe, PipelineBoundaryId, PipelinePlanBuilder, StreamId,
    TwoRankPipeline,
};
use incin_core::dist::mesh::DeviceMesh;
use incin_core::typenum::{U1, U2};

fn integer_pipeline(mesh: &DeviceMesh<TwoRankPipeline>) {
    PipelinePlanBuilder::build_static::<u32, (U2,), U1, GPipe>(
        mesh,
        0,
        PipelineBoundaryId::new(1).unwrap(),
        ActivationCheckpoint::Keep,
        StreamId::default(),
    )
    .unwrap();
}

fn main() {}
