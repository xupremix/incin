//! A capability entry without an executor entry must fail at expansion time,
//! naming the operation, rather than as a bare trait-bound error.
use ::incin::backend_authoring::{StorageBackend, TensorMeta};
use ::incin::prelude::{Cpu, DType};
use ::incin_macros::define_backend_operations;

struct Backend;

impl StorageBackend for Backend {
    const BACKEND_NAME: &'static str = "missing-executor-fixture";
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

fn zeros(
    _: &Backend,
    _: ::incin::backend_authoring::ExecutionRequest<
        '_,
        ::incin::backend_authoring::operations::op::Zeros,
        Backend,
    >,
) -> Result<f64, ::incin::BackendError> {
    Ok(0.0)
}

define_backend_operations! {
    for Backend {
        ::incin::backend_authoring::operations::op::Zeros => f64 = zeros;
    }
    capabilities for Backend {
        ::incin::backend_authoring::operations::op::Zeros => support;
        ::incin::backend_authoring::operations::op::Ones => support;
    }
}

fn main() {}
