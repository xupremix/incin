//! Integration coverage for `unsupported` on the documented public surface.
use incin_core::dist::mesh::{Data, MeshAxis, MeshSpec, TensorParallel};
use incin_core::dist::{CollectivePlanBuilder, Replicated, Sharded, StreamId};
use incin_core::prelude::Q8_0;
use incin_core::typenum::{U1, U2};

type Mesh = MeshSpec<Data<U1>, TensorParallel<U2>>;

fn unsupported(builder: &mut CollectivePlanBuilder<'_, Mesh>) {
    // Q8_0 has no scalar CollectiveDType implementation. Its blocks require a
    // separate layout-aware contract rather than inferred scalar byte counts.
    let _ = builder.push_static::<
        Q8_0,
        Sharded<Mesh, U1>,
        Replicated<Mesh>,
    >(MeshAxis::Tensor, 0, 32, StreamId::default(), None);
}

fn main() {}
