use incin_core::dist::mesh::{Data, DeviceMesh, MeshSpec, Pipeline, TensorParallel};
use incin_core::dist::TwoRankPlanningTopology;
use incin_core::typenum::U1;

type OneRank = MeshSpec<Data<U1>, TensorParallel<U1>, Pipeline<U1>>;

fn wrong_world(mesh: &DeviceMesh<OneRank>) {
    let _ = TwoRankPlanningTopology::from_static_mesh(mesh);
}

fn main() {}
