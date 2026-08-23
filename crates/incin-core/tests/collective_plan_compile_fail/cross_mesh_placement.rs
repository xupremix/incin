//! Integration coverage for `cross_mesh` on the documented public surface.
use incin_core::dist::mesh::{Data, MeshAxis, MeshSpec, TensorParallel};
use incin_core::dist::{CollectivePlanBuilder, Replicated, Sharded, StreamId};
use incin_core::typenum::{U1, U2};

type BoundMesh = MeshSpec<Data<U1>, TensorParallel<U2>>;
type OtherMesh = MeshSpec<Data<U2>>;

fn cross_mesh(builder: &mut CollectivePlanBuilder<'_, BoundMesh>) {
    // A legal transition on another mesh is not legal in this bound plan.
    let _ = builder.push_static::<
        f32,
        Sharded<OtherMesh, U1>,
        Replicated<OtherMesh>,
    >(MeshAxis::Tensor, 0, 8, StreamId::default(), None);
}

fn main() {}
