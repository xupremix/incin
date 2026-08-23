//! Integration coverage for `local_selection_is_not_a_collective` on the documented public surface.
use incin_core::dist::mesh::{Data, DeviceMesh, MeshAxis, MeshSpec, TensorParallel};
use incin_core::dist::{CollectivePlanBuilder, Replicated, Sharded, StreamId};
use incin_core::typenum::{U1, U2};

type Mesh = MeshSpec<Data<U1>, TensorParallel<U2>>;
type Replica = Replicated<Mesh>;
type Shard = Sharded<Mesh, U1>;

fn local_selection_is_not_a_collective(mesh: &DeviceMesh<Mesh>) {
    let mut builder = CollectivePlanBuilder::new(mesh);
    builder
        .push_static::<f32, Replica, Shard>(
            MeshAxis::Tensor,
            0,
            4,
            StreamId::default(),
            None,
        )
        .unwrap();
}

fn main() {}
