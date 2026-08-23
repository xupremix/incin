//! Integration coverage for `wrong_mesh` on the documented public surface.
use incin_core::dist::mesh::{Data, DeviceMesh, MeshSpec};
use incin_core::dist::TensorParallelPlanBuilder;
use incin_core::typenum::U2;

type DataParallelTwo = MeshSpec<Data<U2>>;

fn wrong_mesh(mesh: &DeviceMesh<DataParallelTwo>) {
    let _ = TensorParallelPlanBuilder::new(mesh, 0);
}

fn main() {}
