use alloc::sync::Arc;

/// Core abstraction for `Dataset` within the Kindle framework.
pub trait Dataset: Send + Sync {
    /// Core abstraction for `Item` within the Kindle framework.
    type Item: Send + 'static;

    /// Core abstraction for `len` within the Kindle framework.
    fn len(&self) -> usize;

    /// Core abstraction for `is_empty` within the Kindle framework.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Core abstraction for `get` within the Kindle framework.
    fn get(&self, index: usize) -> Option<Self::Item>;
}

// Implement for Arc<T>
impl<T: Dataset + ?Sized> Dataset for Arc<T> {
    /// Core abstraction for `Item` within the Kindle framework.
    type Item = T::Item;

    /// Core abstraction for `len` within the Kindle framework.
    fn len(&self) -> usize {
        (**self).len()
    }

    /// Core abstraction for `get` within the Kindle framework.
    fn get(&self, index: usize) -> Option<Self::Item> {
        (**self).get(index)
    }
}
