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

    /// Replaces the gradient recorded for a storage handle.
    ///
    /// This is a replacement, not an accumulation: the reverse walk's own
    /// accumulation has finished by the time anything calls this, and a second
    /// summing spelling here is how a gradient gets counted twice. It exists so
    /// that post-backward transforms which rescale a whole gradient set —
    /// clipping is the one in tree — can be written once against the trait
    /// instead of once per backend.
    ///
    /// Required rather than defaulted. A backend that silently dropped the new
    /// value would turn clipping into a no-op, and a caller cannot tell a
    /// no-op rescale from a rescale by one.
    fn set_grad<K: DType>(
        storage: &Self::Storage<K>,
        grads: &mut Self::Grads,
        value: Self::Storage<K>,
    ) -> Result<()>;
}
