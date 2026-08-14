use crate::dist::placement::{Local, Placement};
use crate::err::BackendError;
use crate::exec::context::ExecutionContext;
use crate::exec::request::TensorHandle;
use crate::exec::{TensorMeta, Validated};
use crate::tensor::device::{Device, DeviceId};
use crate::tensor::dtype::{DType, DTypeDescriptor};
use crate::shapes::ShapeBuf;

/// Physical storage ownership, independent of operation execution.
pub trait StorageBackend<P: Placement = Local>: Sized {
    /// How this backend names itself when it refuses work.
    const BACKEND_NAME: &'static str;

    type Storage<K: DType>: Clone;
    type Device: Device;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta;

    /// Returns the authoritative logical shape of a storage handle.
    fn shape<K: DType>(storage: &Self::Storage<K>) -> ShapeBuf {
        Self::metadata(storage).shape().clone()
    }

    /// Returns the physical storage dtype when the backend can inspect it.
    fn storage_dtype<K: DType>(storage: &Self::Storage<K>) -> Option<DTypeDescriptor> {
        Some(Self::metadata(storage).dtype())
    }

    /// Returns the physical storage device when the backend can inspect it.
    fn storage_device<K: DType>(storage: &Self::Storage<K>) -> Option<DeviceId> {
        Some(Self::metadata(storage).device)
    }

    /// Returns `storage` with a fresh autograd identity after it crosses a
    /// gradient-tracking boundary.
    fn fresh_autograd_identity<K: DType>(storage: Self::Storage<K>) -> Self::Storage<K> {
        storage
    }

    #[doc(hidden)]
    fn execution_storage<K: DType>(
        storage: &Self::Storage<K>,
    ) -> (&dyn core::any::Any, Option<usize>)
    where
        Self::Storage<K>: core::any::Any,
    {
        (storage, None)
    }
}

/// One validated descriptor invocation against checked tensor handles.
pub struct ExecutionRequest<'a, O, B>
where
    O: crate::exec::catalog::Operation,
    B: StorageBackend,
{
    pub operation: &'a Validated<crate::exec::catalog::Descriptor<O>>,
    pub inputs: &'a [TensorHandle<'a>],
    pub context: &'a ExecutionContext<B>,
    /// Borrowed execution data kept outside the semantic descriptor.
    pub payload: Option<crate::exec::request::ExecutionPayload<'a>>,
}

/// A value returned by a backend executor.
pub trait ExecuteOutput {}

pub trait StorageOutput {}

impl<T: StorageOutput> ExecuteOutput for T {}
impl ExecuteOutput for crate::shapes::ShapeBuf {}
impl ExecuteOutput for crate::exec::ProofLevel {}
impl ExecuteOutput for f64 {}
impl ExecuteOutput for i64 {}
impl ExecuteOutput for () {}
impl ExecuteOutput for f32 {}
impl ExecuteOutput for i32 {}
impl ExecuteOutput for alloc::vec::Vec<u8> {}
impl ExecuteOutput for alloc::vec::Vec<usize> {}
impl<T: ExecuteOutput> ExecuteOutput for alloc::vec::Vec<T> {}
impl<L: ExecuteOutput, R: ExecuteOutput> ExecuteOutput for (L, R) {}

/// Executes one descriptor type. Absence of an implementation is a compile-time fact.
pub trait Execute<O>: StorageBackend + Sized
where
    O: crate::exec::catalog::Operation,
{
    type Output: ExecuteOutput;

    fn supports_custom(&self, _query: &crate::exec::CapabilityQuery) -> crate::exec::SupportLevel {
        crate::exec::SupportLevel::Native
    }

    fn supports_custom_operation(
        &self,
        _operation: &crate::exec::OperationIdentity,
        _training: bool,
        _math_mode: crate::exec::MathMode,
    ) -> crate::exec::SupportLevel {
        crate::exec::SupportLevel::Native
    }

    /// Run a validated invocation using lowered runtime semantic evidence.
    fn execute(
        &self,
        request: ExecutionRequest<'_, O, Self>,
    ) -> core::result::Result<Self::Output, BackendError>;
}
