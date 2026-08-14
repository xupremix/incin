use super::{Backend, StorageBackend, SupportsDType, VariableBackend};
use crate::err::Result;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;

/// Transfers storage and variables from one backend to a backend on `NewD`.
///
/// Implementations must not assume that storage handles or raw variable types
/// are compatible across backend families.
pub trait TransferTo<NewD: Device>: StorageTransfer<NewD> + VariableBackend
where
    <Self as StorageTransfer<NewD>>::Output: VariableBackend<Device = NewD>,
{
    /// Transfers a variable into destination-native variable storage.
    /// Generic over K so that non-f32 parameter dtypes are preserved.
    fn transfer_var<K: DType>(
        variable: &Self::Var<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<<Self as StorageTransfer<NewD>>::Output as VariableBackend>::Var<K>>
    where
        <Self as StorageTransfer<NewD>>::Output: SupportsDType<K>;
}

/// Transfers backend-owned variables without requiring callers to use the
/// combined storage-and-variable compatibility contract.
pub trait VariableTransfer<NewD: Device>: VariableBackend {
    /// Backend selected for destination variables.
    type VariableOutput: VariableBackend<Device = NewD>;

    /// Transfers a variable into destination-native variable storage.
    fn transfer_var<K: DType>(
        variable: &Self::Var<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::VariableOutput as VariableBackend>::Var<K>>
    where
        Self::VariableOutput: SupportsDType<K>;
}

impl<B, NewD> VariableTransfer<NewD> for B
where
    B: TransferTo<NewD>,
    NewD: Device,
    <B as StorageTransfer<NewD>>::Output: VariableBackend<Device = NewD>,
{
    type VariableOutput = <B as StorageTransfer<NewD>>::Output;

    fn transfer_var<K: DType>(
        variable: &Self::Var<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::VariableOutput as VariableBackend>::Var<K>>
    where
        Self::VariableOutput: SupportsDType<K>,
    {
        <B as TransferTo<NewD>>::transfer_var(variable, dtype, device)
    }
}

/// Storage-only transfer capability for inference and tensor movement.
///
/// This capability deliberately has no `VariableBackend` bound. Inference-only
/// backends can implement it directly, while variable-capable backends may
/// additionally implement [`TransferTo`].
pub trait StorageTransfer<NewD: Device>: Backend {
    /// Backend selected for the destination device.
    type Output: Backend<Device = NewD> + StorageBackend<Device = NewD>;

    /// Transfers tensor storage while preserving shape and dtype.
    fn transfer_storage<K: DType>(
        storage: &<Self as StorageBackend>::Storage<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as StorageBackend>::Storage<K>>
    where
        Self::Output: SupportsDType<K>;
}
