use super::Backend;
use crate::err::{BackendError, Result};
use crate::tensor::dtype::DType;

/// Trainable-variable storage capabilities.
pub trait VariableBackend: Backend {
    /// Backend-native variable handle.
    type Var<K: DType>: Clone + 'static;
    /// Returns a stable identity for a variable slot when cloned handles share
    /// one mutable destination. Backends that cannot make that guarantee must
    /// return `None`; state loading will then treat every parameter as distinct.
    fn var_slot_identity<K: DType>(_var: &Self::Var<K>) -> Option<usize> {
        None
    }
    /// Views a variable as ordinary tensor storage.
    fn var_as_tensor<K: DType>(_var: &Self::Var<K>) -> Result<Self::Storage<K>> {
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "trainable variables",
            },
        )))
    }
    /// Promotes tensor storage to a trainable variable.
    fn var_from_tensor<K: DType>(storage: &Self::Storage<K>) -> Result<Self::Var<K>> {
        let _ = storage;
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "trainable variables",
            },
        )))
    }
    /// Failure-atomic variable assignment.
    fn assign_var<K: DType>(var: &mut Self::Var<K>, storage: &Self::Storage<K>) -> Result<()> {
        let _ = (var, storage);
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "trainable variables",
            },
        )))
    }
}
