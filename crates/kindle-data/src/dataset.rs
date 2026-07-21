use alloc::sync::Arc;

/// Auto-generated documentation for Dataset.
pub trait Dataset: Send + Sync {
    /// Auto-generated documentation for Item.
    type Item: Send + 'static;

    /// Auto-generated documentation for len.
    fn len(&self) -> usize;

    /// Auto-generated documentation for is_empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Auto-generated documentation for get.
    fn get(&self, index: usize) -> Option<Self::Item>;
}

// Implement for Arc<T>
impl<T: Dataset + ?Sized> Dataset for Arc<T> {
    /// Auto-generated documentation for Item.
    type Item = T::Item;

    /// Auto-generated documentation for len.
    fn len(&self) -> usize {
        (**self).len()
    }

    /// Auto-generated documentation for get.
    fn get(&self, index: usize) -> Option<Self::Item> {
        (**self).get(index)
    }
}
