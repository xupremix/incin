//! Integration coverage for `placement_expansion_and_projections` on the documented public surface.
use incin_core::dist::{ConstPlacement, PlacementKind};
use incin_macros::{mesh, placement};

type MyMesh = mesh![dp = 2, tp = 4];

#[test]
fn placement_expansion_and_projections() {
    type PLocal = placement![Local];
    assert_eq!(PLocal::PLACEMENT, PlacementKind::Local);

    type PRepl = placement![Replicated on MyMesh];
    assert_eq!(PRepl::PLACEMENT, PlacementKind::Replicated);

    type PShard0 = placement![Sharded(0) on MyMesh];
    assert_eq!(PShard0::PLACEMENT, PlacementKind::Sharded { axis: 0 });

    type PShard1 = placement![Sharded(1) on MyMesh];
    assert_eq!(PShard1::PLACEMENT, PlacementKind::Sharded { axis: 1 });

    type PPartialSum = placement![Partial(Sum) on MyMesh];
    assert_eq!(
        PPartialSum::PLACEMENT,
        PlacementKind::Partial {
            reduction: incin_core::exec::ReduceOp::Sum
        }
    );

    type PPipeline0 = placement![PipelineStage(0) on MyMesh];
    assert_eq!(
        PPipeline0::PLACEMENT,
        PlacementKind::PipelineStage { index: 0 }
    );
}

#[test]
fn placement_compile_fail_diagnostics() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/placement_compile_fail/*.rs");
}
