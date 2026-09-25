//! A handler returning the wrong output must fail as a mismatched-types error
//! naming the operation and the declared output.
use ::incin::backend_authoring::{ExecutionRequest, StorageBackend, TensorMeta, operations::op};
use ::incin::prelude::{Cpu, DType};
use ::incin_macros::define_backend_operations;

struct Backend;

impl StorageBackend for Backend {
    const BACKEND_NAME: &'static str = "output-fixture";
    type Storage<K: DType> = TensorMeta;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &TensorMeta) -> &TensorMeta {
        storage
    }
}

fn support(
    _: &Backend,
    _: &::incin::backend_authoring::CapabilityQuery,
) -> ::incin::backend_authoring::SupportLevel {
    ::incin::backend_authoring::SupportLevel::Native
}

fn scalar(_: &Backend, _: ExecutionRequest<'_, op::Zeros, Backend>) -> Result<f64, ::incin::BackendError> {
    Ok(0.0)
}

define_backend_operations! {
    for Backend {
        op::Zeros => Vec<f64> = scalar;
    }
    capabilities for Backend {
        op::Zeros => support;
    }
}

fn main() {}
