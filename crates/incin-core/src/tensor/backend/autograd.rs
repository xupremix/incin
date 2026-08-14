use super::{Backend, StorageBackend};
use crate::err::Result;
use crate::tensor::dtype::DType;

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
