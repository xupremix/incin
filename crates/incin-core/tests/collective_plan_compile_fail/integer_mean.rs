use incin_core::dist::mesh::{Data, DeviceMesh, MeshAxis, MeshSpec, TensorParallel};
use incin_core::dist::{CollectivePlanBuilder, Mean, Partial, Replicated, StreamId};
use incin_core::typenum::{U1, U2};

type Mesh = MeshSpec<Data<U1>, TensorParallel<U2>>;
type PartialMean = Partial<Mesh, Mean>;
type Replica = Replicated<Mesh>;

fn integer_mean(mesh: &DeviceMesh<Mesh>) {
    let mut builder = CollectivePlanBuilder::new(mesh);
    builder
        .push_static::<u32, PartialMean, Replica>(
            MeshAxis::Tensor,
            0,
            4,
            StreamId::default(),
            None,
        )
        .unwrap();
}

fn main() {}
