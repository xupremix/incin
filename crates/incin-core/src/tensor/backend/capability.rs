//! Optional backend capability views.
//!
//! These traits are the migration seam away from the historical all-in-one
//! [`Backend`](super::Backend) contract. They deliberately do not introduce a
//! runtime/session abstraction: ownership remains with the backend type.

use super::StorageBackend;
use crate::err::Result;
use crate::tensor::device::DeviceId;
use crate::tensor::dtype::{DType, DTypeDescriptor};
use crate::shapes::ShapeBuf;

/// Reads tensor values into host-owned vectors for inspection and formatting.
///
/// This is kept separate from the metadata/serialization capability so host
/// formatting does not expose the historical all-in-one operation family.
pub trait HostReadback: StorageBackend {
    /// Reads floating-point values in logical row-major order.
    fn float_to_vec1<K: DType>(storage: &Self::Storage<K>) -> Result<alloc::vec::Vec<f64>>;
    /// Reads integer values in logical row-major order.
    fn int_to_vec1<K: DType>(storage: &Self::Storage<K>) -> Result<alloc::vec::Vec<i64>>;
}

/// Host-visible tensor metadata and formatting capabilities.
pub trait HostInterop: StorageBackend + HostReadback {
    /// Returns the logical shape of a storage handle.
    fn host_shape<K: DType>(storage: &Self::Storage<K>) -> ShapeBuf {
        <Self as StorageBackend>::shape(storage)
    }
    /// Returns the physical dtype when the storage exposes it.
    fn host_storage_dtype<K: DType>(storage: &Self::Storage<K>) -> Option<DTypeDescriptor> {
        <Self as StorageBackend>::storage_dtype(storage)
    }
    /// Returns the physical device when the storage exposes it.
    fn host_storage_device<K: DType>(storage: &Self::Storage<K>) -> Option<DeviceId> {
        <Self as StorageBackend>::storage_device(storage)
    }
    /// Formats a storage value for human-facing display.
    fn host_format_display<K: DType>(
        storage: &Self::Storage<K>,
    ) -> alloc::string::String
    {
        use crate::tensor::display::{render, Values};
        let shape = Self::shape(storage);
        match Self::storage_dtype(storage) {
            None => alloc::format!("<tensor: shape={shape:?}, dtype unknown to this backend>"),
            Some(dtype) if dtype.is_quantized() => alloc::format!(
                "<{} tensor: shape={shape:?}, not printable without dequantizing>",
                dtype.name()
            ),
            Some(dtype) if dtype.is_integer() => match Self::int_to_vec1(storage) {
                Ok(values) => render(&shape, &Values::Int(values)),
                Err(err) => alloc::format!("<tensor: shape={shape:?}, values unavailable: {err}>"),
            },
            Some(_) => match Self::float_to_vec1(storage) {
                Ok(values) => render(&shape, &Values::Float(values)),
                Err(err) => alloc::format!("<tensor: shape={shape:?}, values unavailable: {err}>"),
            },
        }
    }
    /// Formats a storage value for diagnostic output.
    fn host_format_debug<K: DType>(storage: &Self::Storage<K>) -> alloc::string::String
    {
        Self::host_format_display(storage)
    }

    /// Serializes storage to a flat, dtype-native byte buffer.
    fn to_bytes<K: DType>(storage: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>>;

    /// Reconstructs storage from bytes produced by [`Self::to_bytes`].
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>>;
}

/// Explicit capability marker for backends that can move tensor storage to
/// `NewD`. This marker deliberately does not require variable ownership or
/// training capabilities; inference-only backends can implement it through
/// [`super::StorageTransfer`].
pub trait TransferBackend<NewD: crate::tensor::device::Device>: super::StorageTransfer<NewD> {}

impl<B, NewD> TransferBackend<NewD> for B
where
    B: super::StorageTransfer<NewD>,
    NewD: crate::tensor::device::Device,
{
}
