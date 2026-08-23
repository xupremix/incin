//! Checked, type-erased tensor inputs for descriptor execution.

use core::any::Any;
use core::fmt;

use crate::dist::placement::Placement;
use crate::exec::TensorMeta;
use crate::tensor::backend::StorageBackend;
use crate::tensor::dtype::DType;

/// A borrowed backend storage handle coupled to its checked physical metadata.
///
/// Construction goes through `StorageBackend::metadata`, so callers cannot pair
/// one allocation with another tensor's metadata. Executors recover their
/// concrete storage type with `downcast_ref` after the backend has been chosen.
pub struct TensorHandle<'a> {
    storage: &'a dyn Any,
    metadata: &'a TensorMeta,
    execution_storage: &'a dyn Any,
    tracing_value: Option<usize>,
}

/// Optional borrowed payload owned by a creation or host-side execution call.
pub type ExecutionPayload<'a> = &'a [u8];

impl<'a> TensorHandle<'a> {
    #[must_use]
    /// Wraps storage borrowed from a backend.
    pub fn from_storage<B, K, P>(storage: &'a <B as StorageBackend<P>>::Storage<K>) -> Self
    where
        B: StorageBackend<P>,
        K: DType,
        P: Placement,
        <B as StorageBackend<P>>::Storage<K>: Any,
    {
        let (execution_storage, tracing_value) = B::execution_storage(storage);
        Self {
            storage,
            metadata: B::metadata(storage),
            execution_storage,
            tracing_value,
        }
    }

    #[must_use]
    /// Validated metadata of the wrapped storage.
    pub const fn metadata(&self) -> &TensorMeta {
        self.metadata
    }

    /// Downcasts to a concrete handle type when possible.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.storage.downcast_ref()
    }

    pub(crate) fn execution_view(&self) -> Self {
        Self {
            storage: self.execution_storage,
            metadata: self.metadata,
            execution_storage: self.execution_storage,
            tracing_value: None,
        }
    }

    pub(crate) const fn tracing_value(&self) -> Option<usize> {
        self.tracing_value
    }
}

impl fmt::Debug for TensorHandle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorHandle")
            .field("metadata", self.metadata)
            .finish_non_exhaustive()
    }
}
