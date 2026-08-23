use incin_core::dist::{
    ActivationCheckpoint, GPipe, PipelineBoundaryId, PipelinePlanBuilder, StreamId,
    TwoRankPipeline,
};
use incin_core::dist::mesh::DeviceMesh;
use incin_core::typenum::{U0, U2};

fn zero_microbatches(mesh: &DeviceMesh<TwoRankPipeline>) {
    PipelinePlanBuilder::build_static::<f32, (U2,), U0, GPipe>(
        mesh,
        0,
        PipelineBoundaryId::new(1).unwrap(),
        ActivationCheckpoint::Keep,
        StreamId::default(),
    )
    .unwrap();
}

fn main() {}
