use incin_backends::cuda::CudaBackendImpl;
use incin_backends::dist::NcclTransport;
use incin_core::dist::{PipelineBoundaryId, PipelineTransfer};
use incin_core::prelude::{CudaN, Dyn, Tensor};
use incin_core::typenum::U0;

type Cuda = CudaBackendImpl<CudaN<U0>>;

fn integer_pipeline(
    transport: &mut NcclTransport,
    tensor: &Tensor<Dyn, Cuda, u32>,
) {
    transport
        .execute_pipeline(
            PipelineBoundaryId::new(1).unwrap(),
            PipelineTransfer::ForwardActivation,
            0,
            tensor,
        )
        .unwrap();
}

fn main() {}
