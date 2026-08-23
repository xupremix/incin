//! Integration coverage for `integer_tensor_parallel` on the documented public surface.
use incin_backends::cuda::CudaBackendImpl;
use incin_backends::dist::NcclTransport;
use incin_core::dist::{TensorParallelCollective, TensorParallelId};
use incin_core::prelude::{CudaN, Dyn, Tensor};
use incin_core::typenum::U0;

type Cuda = CudaBackendImpl<CudaN<U0>>;

fn integer_tensor_parallel(
    transport: &mut NcclTransport,
    tensor: &Tensor<Dyn, Cuda, u32>,
) {
    transport
        .execute_tensor_parallel_flat(
            TensorParallelId::new(1).unwrap(),
            TensorParallelCollective::RowOutputSum,
            tensor,
        )
        .unwrap();
}

fn main() {}
