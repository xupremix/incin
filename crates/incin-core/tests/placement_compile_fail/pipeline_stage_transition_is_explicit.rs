use incin_core::dist::{LegalTransition, PipelineStage, Placement};
use incin_core::dist::mesh::{Data, MeshSpec, Pipeline, TensorParallel};
use incin_core::typenum::{U1, U2};

type Mesh = MeshSpec<Data<U1>, TensorParallel<U1>, Pipeline<U2>>;

fn requires_direct_transition<From, To>()
where
    From: LegalTransition<To>,
    To: Placement,
{
}

fn main() {
    // Sending between stages belongs to the pipeline scheduler.
    requires_direct_transition::<PipelineStage<Mesh, 0>, PipelineStage<Mesh, 1>>();
}
