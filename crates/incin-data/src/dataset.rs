use alloc::sync::Arc;

use crate::loader::DataError;

/// Dataset.
pub trait Dataset: Send + Sync {
    /// Item.
    type Item: Send + 'static;

    /// Len.
    fn len(&self) -> usize;

    /// Is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get.
    fn get(&self, index: usize) -> Result<Option<Self::Item>, DataError>;
}

// Implement for Arc<T>
impl<T: Dataset + ?Sized> Dataset for Arc<T> {
    /// Item.
    type Item = T::Item;

    /// Len.
    fn len(&self) -> usize {
        (**self).len()
    }

    /// Get.
    fn get(&self, index: usize) -> Result<Option<Self::Item>, DataError> {
        (**self).get(index)
    }
}
