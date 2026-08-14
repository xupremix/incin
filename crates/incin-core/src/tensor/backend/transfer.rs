use super::{Backend, StorageBackend, SupportsDType, VariableBackend};
use crate::err::Result;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;

/// Transfers storage and variables from one backend to a backend on `NewD`.
///
/// Implementations must not assume that storage handles or raw variable types
/// are compatible across backend families.
pub trait TransferTo<NewD: Device>: VariableBackend {
    /// Backend selected for the destination device.
    type Output: VariableBackend<Device = NewD>;

    /// Transfers tensor storage while preserving shape and dtype.
    fn transfer_storage<K: DType>(
        storage: &<Self as StorageBackend>::Storage<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as StorageBackend>::Storage<K>>
    where
        Self::Output: SupportsDType<K>;

    /// Transfers a variable into destination-native variable storage.
    /// Generic over K so that non-f32 parameter dtypes are preserved.
    fn transfer_var<K: DType>(
        variable: &Self::RawVar,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as crate::tensor::backend::VariableBackend>::RawVar>
    where
        Self::Output: SupportsDType<K>;
}
