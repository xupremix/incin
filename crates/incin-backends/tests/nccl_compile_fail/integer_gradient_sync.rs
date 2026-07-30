use incin_backends::cuda::{CudaBackendImpl, CudaGrads};
use incin_backends::dist::{NcclTransport};
use incin_core::dist::GradientId;
use incin_core::prelude::{CudaN, Dyn, Tensor};
use incin_core::typenum::U0;

type Backend = CudaBackendImpl<f32, CudaN<U0>>;
type IntegerParameter = Tensor<Dyn, Backend, u32>;

fn synchronize(
    transport: &mut NcclTransport,
    parameter: &IntegerParameter,
    gradients: &mut CudaGrads,
) {
    let _ = transport.synchronize_gradient(
        GradientId::new(1).unwrap(),
        parameter,
        gradients,
    );
}

fn main() {}
