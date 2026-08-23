//! Integration coverage for `Probe` on the documented public surface.
extern crate incin_core as incin;

use incin::backend_authoring::{Execute, ExecutionRequest, StorageBackend};
use incin::exec::op;
use incin::prelude::BackendError;

struct Probe;

impl StorageBackend for Probe {
    const BACKEND_NAME: &'static str = "probe";
    type Storage<K: incin::prelude::DType> = ();
    type Device = incin::prelude::Cpu;

    fn metadata<K: incin::prelude::DType>(
        _storage: &Self::Storage<K>,
    ) -> &incin::backend_authoring::TensorMeta {
        unimplemented!()
    }
}

impl Execute<op::Add> for Probe {
    type Output = usize;

    fn execute(
        &self,
        _request: ExecutionRequest<'_, op::Add, Self>,
    ) -> Result<Self::Output, BackendError> {
        Ok(0)
    }
}

fn main() {}
