#![cfg(feature = "distributed")]

use incin::prelude::*;
use incin_core::dist::{ConstPlacement, PlacementKind};
use incin_macros::{mesh, placement};

type ClusterMesh = mesh![dp = 4, tp = 2, pp = 2];

type ClusterRepl = placement![Replicated on ClusterMesh];
type ClusterShard = placement![Sharded(1) on ClusterMesh];
type ClusterPartial = placement![Partial(Sum) on ClusterMesh];
type ClusterPipe = placement![PipelineStage(1) on ClusterMesh];

#[module]
#[allow(dead_code)]
pub struct DistributedTransformer<B: Backend> {
    #[parallel(mesh = ClusterMesh, stage = 0)]
    embed: Linear<s![512, 1024], B>,

    #[shard(mesh = ClusterMesh, axis = 1)]
    proj: Linear<s![1024, 2048], B>,
}

#[test]
fn test_distributed_macro_suite_projections() {
    assert_eq!(ClusterRepl::PLACEMENT, PlacementKind::Replicated);
    assert_eq!(ClusterShard::PLACEMENT, PlacementKind::Sharded { axis: 1 });
    assert_eq!(
        ClusterPartial::PLACEMENT,
        PlacementKind::Partial {
            reduction: incin_core::exec::ReduceOp::Sum
        }
    );
    assert_eq!(
        ClusterPipe::PLACEMENT,
        PlacementKind::PipelineStage { index: 1 }
    );

    let model = DistributedTransformer::<DefaultBackend> {
        embed: Linear::build(()).unwrap(),
        proj: Linear::build(()).unwrap(),
    };
    assert_eq!(model.parameters().len(), 4);
}
