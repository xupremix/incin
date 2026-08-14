use super::{Backend, StorageBackend};
use crate::err::Result;
use crate::tensor::dtype::DType;

/// Trainable-variable storage capabilities.
pub trait VariableBackend: StorageBackend {
    /// Backend-native variable handle.
    type RawVar: Clone;
    /// Views a variable as ordinary tensor storage.
    fn variable_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>>;
    /// Promotes tensor storage to a trainable variable.
    fn variable_from_tensor<K: DType>(storage: &Self::Storage<K>) -> Result<Self::RawVar>;
    /// Failure-atomic variable assignment.
    fn variable_assign<K: DType>(var: &mut Self::RawVar, storage: &Self::Storage<K>) -> Result<()>;
}

impl<B: Backend> VariableBackend for B {
    type RawVar = B::RawVar;

    fn variable_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        <B as Backend>::var_as_tensor(var)
    }
    fn variable_from_tensor<K: DType>(storage: &Self::Storage<K>) -> Result<Self::RawVar> {
        <B as Backend>::var_from_tensor(storage)
    }
    fn variable_assign<K: DType>(var: &mut Self::RawVar, storage: &Self::Storage<K>) -> Result<()> {
        <B as Backend>::assign_var(var, storage)
    }
}
