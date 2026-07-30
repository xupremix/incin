use incin_backends::cpu::{CpuBackendImpl, CpuStorage};
use incin_core::dist::mesh::{Data, MeshSpec};
use incin_core::dist::{PipelineStage, Replicated, ValidatedDistributed};
use incin_core::exec::ReshapeSpec;
use incin_core::prelude::{Grad, Tensor};
use incin_core::typenum::U2;

type Mesh = MeshSpec<Data<U2>>;
type Source = Tensor<(U2, U2), CpuBackendImpl, f32, Grad, Replicated<Mesh>>;

fn illegal(
    tensor: Source,
    storage: CpuStorage,
    proof: &ValidatedDistributed<ReshapeSpec>,
) {
    // Pipeline-stage changes require an explicit send/receive plan. There is
    // no `LegalTransition` implementation from replicated storage.
    let _ = tensor.try_reshard::<PipelineStage<Mesh, 0>, _>(storage, 0, proof);
}

fn main() {}
