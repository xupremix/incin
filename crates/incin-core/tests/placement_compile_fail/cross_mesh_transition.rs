use incin_core::dist::{LegalTransition, Placement, Replicated, Sharded};
use incin_core::dist::mesh::{Data, MeshSpec, TensorParallel};
use incin_core::typenum::{U0, U1, U2};

type FirstMesh = MeshSpec<Data<U2>>;
type SecondMesh = MeshSpec<Data<U1>, TensorParallel<U2>>;

fn requires_direct_transition<From, To>()
where
    From: LegalTransition<To>,
    To: Placement,
{
}

fn main() {
    // Moving between meshes requires an explicit boundary, not an in-mesh shard.
    requires_direct_transition::<Replicated<FirstMesh>, Sharded<SecondMesh, U0>>();
}
