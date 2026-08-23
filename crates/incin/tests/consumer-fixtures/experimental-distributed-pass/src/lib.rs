//! Integration coverage for `experimental_contract` on the documented public surface.
use incin::experimental::distributed::mesh::ValidMesh;
use incin::experimental::{mesh, placement};

type Mesh = mesh![dp = 2];
type Placement = placement!(Replicated on Mesh);

pub fn experimental_contract() -> usize {
    let _: core::marker::PhantomData<Placement> = core::marker::PhantomData;
    Mesh::WORLD
}
