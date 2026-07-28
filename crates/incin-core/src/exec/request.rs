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
}

impl<'a> TensorHandle<'a> {
    #[must_use]
    pub fn from_storage<B, K, P>(storage: &'a <B as StorageBackend<P>>::Storage<K>) -> Self
    where
        B: StorageBackend<P>,
        K: DType,
        P: Placement,
        <B as StorageBackend<P>>::Storage<K>: Any,
    {
        Self {
            storage,
            metadata: <B as StorageBackend<P>>::metadata(storage),
        }
    }

    #[must_use]
    pub const fn metadata(&self) -> &TensorMeta {
        self.metadata
    }

    #[must_use]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.storage.downcast_ref()
    }
}

impl fmt::Debug for TensorHandle<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorHandle")
            .field("metadata", self.metadata)
            .finish_non_exhaustive()
    }
}
