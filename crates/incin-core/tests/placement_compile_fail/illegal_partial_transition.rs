use incin_core::dist::{LegalTransition, Partial, Placement, Sharded, Sum};
use incin_core::dist::mesh::{Data, MeshSpec};
use incin_core::typenum::{U0, U1};

type Mesh = MeshSpec<Data<U1>>;

fn requires_direct_transition<From, To>()
where
    From: LegalTransition<To>,
    To: Placement,
{
}

fn main() {
    // Producing a partial is an operation rule, not a reshard transition.
    requires_direct_transition::<Sharded<Mesh, U0>, Partial<Mesh, Sum>>();
}
