use incin_core::dist::mesh::{Data, DeviceMesh, MeshSpec, TensorParallel};
use incin_core::dist::DataParallelPlanBuilder;
use incin_core::typenum::{U1, U2};

type TensorParallelTwo = MeshSpec<Data<U1>, TensorParallel<U2>>;

fn wrong_mesh(mesh: &DeviceMesh<TensorParallelTwo>) {
    let _ = DataParallelPlanBuilder::new(mesh, 0);
}

fn main() {}
