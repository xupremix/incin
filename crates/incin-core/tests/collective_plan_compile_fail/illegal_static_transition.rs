//! Integration coverage for `illegal` on the documented public surface.
use incin_core::dist::mesh::{Data, MeshAxis, MeshSpec, TensorParallel};
use incin_core::dist::{CollectivePlanBuilder, Sharded, StreamId};
use incin_core::typenum::{U0, U1, U2};

type Mesh = MeshSpec<Data<U1>, TensorParallel<U2>>;

fn illegal(builder: &mut CollectivePlanBuilder<'_, Mesh>) {
    // Changing the sharded tensor axis has no direct LegalTransition. Static
    // plan construction must reject it before a descriptor can exist.
    let _ = builder.push_static::<
        f32,
        Sharded<Mesh, U0>,
        Sharded<Mesh, U1>,
    >(MeshAxis::Tensor, 0, 8, StreamId::default(), None);
}

fn main() {}
