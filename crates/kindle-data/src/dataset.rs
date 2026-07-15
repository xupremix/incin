use alloc::sync::Arc;

pub trait Dataset: Send + Sync {
    type Item: Send + 'static;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, index: usize) -> Option<Self::Item>;
}

// Implement for Arc<T>
impl<T: Dataset + ?Sized> Dataset for Arc<T> {
    type Item = T::Item;

    fn len(&self) -> usize {
        (**self).len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        (**self).get(index)
    }
}
