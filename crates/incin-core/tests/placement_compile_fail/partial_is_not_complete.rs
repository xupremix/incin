use incin_core::dist::{CompletePlacement, Partial, Sum};
use incin_core::dist::mesh::{Data, MeshSpec};
use incin_core::typenum::U1;

type Mesh = MeshSpec<Data<U1>>;

fn requires_complete<P: CompletePlacement>() {}

fn main() {
    // A partial sum must be reduced before an ordinary consumer can read it.
    requires_complete::<Partial<Mesh, Sum>>();
}
