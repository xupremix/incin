//! Integration coverage for `main` on the documented public surface.
use incin::experimental::distributed::mesh::{Data, MeshSpec, Pipeline, TensorParallel};
use incin::experimental::distributed::{RunId, TwoRankLaunchPlan};
use incin::typenum::{U1, U3};

type ThreeRanks = MeshSpec<Data<U3>, TensorParallel<U1>, Pipeline<U1>>;

fn main() {
    let _ = TwoRankLaunchPlan::<ThreeRanks>::new_static(
        RunId::new("wrong-world").unwrap(),
        "127.0.0.1:12345".parse().unwrap(),
        [0, 0],
        std::time::Duration::from_secs(1),
    );
}
