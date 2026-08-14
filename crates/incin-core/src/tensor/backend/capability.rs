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

/// Trainable-variable storage capabilities.
pub trait VariableBackend: StorageBackend {
    /// Backend-native variable handle.
    type RawVar: Clone;
    /// Views a variable as ordinary tensor storage.
    fn variable_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>>;
    /// Promotes tensor storage to a trainable variable.
    fn variable_from_tensor<K: DType>(storage: &Self::Storage<K>) -> Result<Self::RawVar>;
    /// Failure-atomic variable assignment.
    fn variable_assign<K: DType>(var: &mut Self::RawVar, storage: &Self::Storage<K>) -> Result<()>;
}

/// Reverse-mode automatic differentiation capabilities.
pub trait AutogradBackend: StorageBackend {
    /// Backend-owned gradient collection.
    type Grads;
    /// Runs reverse-mode differentiation from `storage`.
    fn autograd_backward<K: DType>(storage: &Self::Storage<K>) -> Result<Self::Grads>;
    /// Runs reverse-mode differentiation with an explicit seed.
    fn autograd_backward_with<K: DType>(
        storage: &Self::Storage<K>,
        seed: &Self::Storage<K>,
    ) -> Result<Self::Grads>;
    /// Looks up a gradient for a storage handle.
    fn autograd_get_grad<K: DType>(
        storage: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>>;
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

impl<B: Backend> VariableBackend for B {
    type RawVar = B::RawVar;

    fn variable_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        <B as Backend>::var_as_tensor(var)
    }
    fn variable_from_tensor<K: DType>(storage: &Self::Storage<K>) -> Result<Self::RawVar> {
        <B as Backend>::var_from_tensor(storage)
    }
    fn variable_assign<K: DType>(var: &mut Self::RawVar, storage: &Self::Storage<K>) -> Result<()> {
        <B as Backend>::assign_var(var, storage)
    }
}

impl<B: Backend> AutogradBackend for B {
    type Grads = B::Grads;

    fn autograd_backward<K: DType>(storage: &Self::Storage<K>) -> Result<Self::Grads> {
        <B as Backend>::backward(storage)
    }
    fn autograd_backward_with<K: DType>(
        storage: &Self::Storage<K>,
        seed: &Self::Storage<K>,
    ) -> Result<Self::Grads> {
        <B as Backend>::backward_with(storage, seed)
    }
    fn autograd_get_grad<K: DType>(
        storage: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        <B as Backend>::get_grad(storage, grads)
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
