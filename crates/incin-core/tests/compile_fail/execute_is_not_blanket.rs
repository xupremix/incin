//! Merely owning storage does not grant an operation. Concrete descriptor
//! execution exists only when the backend implements `Execute<O>`.

use incin_core::exec::{TensorMeta, op};
use incin_core::backend_authoring::{Execute, StorageBackend};
use incin_core::prelude::{Cpu, DType};

struct StorageOnly;

impl StorageBackend for StorageOnly {
    const BACKEND_NAME: &'static str = "StorageOnly";
    type Storage<K: DType> = TensorMeta;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage
    }
}

fn requires_broadcast<B: Execute<op::Add>>() {}

fn main() {
    requires_broadcast::<StorageOnly>();
}
