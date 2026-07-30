//! `StorageBackend` and `Capabilities` implementation for the Metal backend.
//!
//! The descriptor-execution contract requires `StorageBackend` to be a
//! separate impl from the main `Backend` impl (mirroring the CPU/WGPU pattern
//! of separating `backend.rs` and `executor.rs`).

use incin_core::exec::{Capabilities, CapabilityQuery, SupportLevel, TensorMeta};
use incin_core::prelude::{DType, Device, DeviceKind, StorageBackend};

use super::backend::MetalBackendImpl;
use super::storage::MetalStorage;

impl<T: DType, D: Device> StorageBackend for MetalBackendImpl<T, D> {
    type Storage<K: DType> = MetalStorage;
    type Device = D;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage.metadata()
    }
}

impl<T: DType, D: Device> Capabilities for MetalBackendImpl<T, D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(DeviceKind::Metal, query)
    }
}
