//! Integration coverage for `DistributedTransformer` on the documented public surface.
#![cfg(feature = "distributed")]

use incin::prelude::*;
use incin_core::dist::{ConstPlacement, PlacementKind};
use incin_macros::{mesh, placement};

type ClusterMesh = mesh![dp = 4, tp = 2, pp = 2];

type ClusterRepl = placement![Replicated on ClusterMesh];
type ClusterShard = placement![Sharded(1) on ClusterMesh];
type ClusterPartial = placement![Partial(Sum) on ClusterMesh];
type ClusterPipe = placement![PipelineStage(1) on ClusterMesh];

/// The embedding projection, named so the field below reads as one type rather
/// than as the expanded typenum tree `s!` produces for these sizes.
type Embed<B> = Linear<s![512, 1024], B>;
type Proj<B> = Linear<s![1024, 2048], B>;

#[module]
#[allow(dead_code)]
pub struct DistributedTransformer<
    B: incin_core::backend_authoring::Backend + incin_core::backend_authoring::VariableBackend,
> {
    #[parallel(mesh = ClusterMesh, stage = 0)]
    embed: Embed<B>,

    #[shard(mesh = ClusterMesh, axis = 1)]
    proj: Proj<B>,
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
    let parameters = ParameterGroup::<DefaultBackend, f32>::from_module(&model).unwrap();
    assert_eq!(parameters.len(), 4);
}
