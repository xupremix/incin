use super::StorageBackend;
use crate::err::{BackendError, Result};
use crate::tensor::dtype::DType;

/// Reverse-mode automatic differentiation capabilities.
pub trait AutogradBackend: StorageBackend {
    /// Backend-owned gradient collection.
    type Grads;
    /// Runs reverse-mode differentiation from `storage`.
    fn backward<K: DType>(storage: &Self::Storage<K>) -> Result<Self::Grads>;
    /// Runs reverse-mode differentiation with an explicit seed.
    fn backward_with<K: DType>(
        storage: &Self::Storage<K>,
        seed: &Self::Storage<K>,
    ) -> Result<Self::Grads> {
        let _ = (storage, seed);
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "seeded backward",
            },
        )))
    }
    /// Looks up a gradient for a storage handle.
    fn get_grad<K: DType>(
        storage: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>>;
}
