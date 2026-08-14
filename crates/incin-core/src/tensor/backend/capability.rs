//! Optional backend capability views.
//!
//! These traits are the migration seam away from the historical all-in-one
//! [`Backend`](super::Backend) contract. They deliberately do not introduce a
//! runtime/session abstraction: ownership remains with the backend type.

use super::{Backend, StorageBackend, TensorOps};
use crate::err::Result;
use crate::tensor::device::DeviceId;
use crate::tensor::dtype::{DType, DTypeDescriptor};
use crate::shapes::ShapeBuf;

/// Host-visible tensor metadata and formatting capabilities.
pub trait HostInterop: StorageBackend {
    /// Returns the logical shape of a storage handle.
    fn host_shape<K: DType>(storage: &Self::Storage<K>) -> ShapeBuf;
    /// Returns the physical dtype when the storage exposes it.
    fn host_storage_dtype<K: DType>(storage: &Self::Storage<K>) -> Option<DTypeDescriptor>;
    /// Returns the physical device when the storage exposes it.
    fn host_storage_device<K: DType>(storage: &Self::Storage<K>) -> Option<DeviceId>;
    /// Formats a storage value for human-facing display.
    fn host_format_display<K: DType>(storage: &Self::Storage<K>) -> alloc::string::String;
    /// Formats a storage value for diagnostic output.
    fn host_format_debug<K: DType>(storage: &Self::Storage<K>) -> alloc::string::String;
}

/// Blanket host capability view for the compatibility `Backend` contract.
impl<B: Backend + TensorOps<B>> HostInterop for B {
    fn host_shape<K: DType>(storage: &Self::Storage<K>) -> ShapeBuf {
        <B as StorageBackend>::shape(storage)
    }
    fn host_storage_dtype<K: DType>(storage: &Self::Storage<K>) -> Option<DTypeDescriptor> {
        <B as StorageBackend>::storage_dtype(storage)
    }
    fn host_storage_device<K: DType>(storage: &Self::Storage<K>) -> Option<DeviceId> {
        <B as StorageBackend>::storage_device(storage)
    }
    fn host_format_display<K: DType>(storage: &Self::Storage<K>) -> alloc::string::String {
        <B as Backend>::format_tensor_display(storage)
    }
    fn host_format_debug<K: DType>(storage: &Self::Storage<K>) -> alloc::string::String {
        <B as Backend>::format_tensor_debug(storage)
    }
}

/// Explicit capability marker for backends that can move storage or
/// variables to `NewD`. `TransferTo` remains the method-bearing compatibility
/// contract used by existing implementations.
pub trait TransferBackend<NewD: crate::tensor::device::Device>: super::TransferTo<NewD> {}

impl<B, NewD> TransferBackend<NewD> for B
where
    B: super::TransferTo<NewD>,
    NewD: crate::tensor::device::Device,
{
}
