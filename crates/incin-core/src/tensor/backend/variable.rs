use super::Backend;
use crate::err::{BackendError, Result};
use crate::tensor::dtype::DType;

/// Trainable-variable storage capabilities.
pub trait VariableBackend: Backend {
    /// Backend-native variable handle.
    type RawVar: Clone;
    /// Views a variable as ordinary tensor storage.
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "trainable variables",
            },
        )))
    }
    /// Promotes tensor storage to a trainable variable.
    fn var_from_tensor<K: DType>(storage: &Self::Storage<K>) -> Result<Self::RawVar> {
        let _ = storage;
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "trainable variables",
            },
        )))
    }
    /// Failure-atomic variable assignment.
    fn assign_var<K: DType>(var: &mut Self::RawVar, storage: &Self::Storage<K>) -> Result<()> {
        let _ = (var, storage);
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "trainable variables",
            },
        )))
    }
}
