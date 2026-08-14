use crate::err::BackendError;
use crate::exec::capability::Capabilities;
use crate::prelude::{DTypeId, DeviceId, FloatToIntPolicy, Result, ShapeBuf, convert_f64_to_i64};
use crate::exec::TensorMeta;
use crate::tensor::device::Device;
use crate::tensor::dtype::{DType, DTypeDescriptor, FloatDType, QuantDType};

mod execute;
pub use execute::{Execute, ExecuteOutput, ExecutionRequest, StorageBackend, StorageOutput};
mod transfer;
pub use transfer::TransferTo;
mod autograd;
pub use autograd::AutogradBackend;
mod variable;
pub use variable::VariableBackend;
mod capability;
pub use capability::{HostInterop, TransferBackend};
mod optimizer;
pub use optimizer::{OptimizerOps, adamw_step_composed};
mod legacy;
pub use legacy::{CreationOps, FloatOps, LossOps, ModuleOps, NumericOps, QuantizedOps, ReductionOps, TensorOps};
pub mod dummy {
    pub use super::legacy::DummyBackend;
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// A backend-agnostic scalar, tagged by whether it originated as a float or
/// an integer literal so callers can round-trip the value without losing
/// that distinction (e.g. when building a `ScalarValue` from a Rust literal
/// via `From` and later reading it back as whichever type the op needs).
pub enum ScalarValue {
    /// A floating-point scalar, stored at full `f64` precision.
    Float(f64),
    /// An integer scalar, stored as `i64`.
    Int(i64),
}

impl ScalarValue {
    /// Reads the value as `f64`, casting from `Int` if needed.
    pub fn to_f64(self) -> f64 {
        match self {
            ScalarValue::Float(f) => f,
            ScalarValue::Int(i) => i as f64,
        }
    }

    /// Reads the value as `i64` under an explicitly selected conversion
    /// policy. Callers that require a lossless conversion use
    /// [`FloatToIntPolicy::Exact`].
    pub fn to_i64(self, policy: FloatToIntPolicy) -> Result<i64> {
        match self {
            ScalarValue::Float(f) => {
                convert_f64_to_i64("scalar_value_to_i64", DTypeId::F64.descriptor(), f, policy)
            }
            ScalarValue::Int(i) => Ok(i),
        }
    }
}

impl From<f32> for ScalarValue {
    /// Widens an `f32` literal into a `Float` scalar.
    fn from(v: f32) -> Self {
        ScalarValue::Float(v as f64)
    }
}
impl From<f64> for ScalarValue {
    /// Wraps an `f64` literal as a `Float` scalar.
    fn from(v: f64) -> Self {
        ScalarValue::Float(v)
    }
}
impl From<i32> for ScalarValue {
    /// Widens an `i32` literal into an `Int` scalar.
    fn from(v: i32) -> Self {
        ScalarValue::Int(v as i64)
    }
}
impl From<i64> for ScalarValue {
    /// Wraps an `i64` literal as an `Int` scalar.
    fn from(v: i64) -> Self {
        ScalarValue::Int(v)
    }
}

#[cfg(test)]
mod scalar_value_tests {
    use super::*;
    use crate::prelude::{ConversionFailure, Error};

    #[test]
    fn float_to_integer_requires_an_explicit_checked_policy() {
        assert_eq!(
            ScalarValue::Float(12.0)
                .to_i64(FloatToIntPolicy::Exact)
                .unwrap(),
            12
        );
        assert!(matches!(
            ScalarValue::Float(12.5).to_i64(FloatToIntPolicy::Exact),
            Err(Error::InvalidConversion {
                reason: ConversionFailure::Fractional,
                ..
            })
        ));
        assert_eq!(
            ScalarValue::Float(12.5)
                .to_i64(FloatToIntPolicy::Truncate)
                .unwrap(),
            12
        );
    }
}

/// Resolves the dtype represented by `K` for a concrete runtime device.
pub trait SupportsDType<K: DType> {
    /// Resolve and validate dtype metadata before storage is created.
    fn resolve_dtype(field: &K::Field, device: &DeviceId) -> Result<DTypeDescriptor>;
}

/// The broad backend profile used by generic tensor modules.
///
/// This profile contains backend identity, storage, dtype resolution, and
/// capability admission. Individual operation implementations remain
/// expressed by exact `Execute<O>` bounds at the call site, so a module can
/// add only the operations it actually invokes without inheriting the old
/// operation-family hierarchy.
pub trait TensorBackend<K: DType>: Backend + SupportsDType<K> {}

impl<B, K> TensorBackend<K> for B
where
    B: Backend + SupportsDType<K>,
    K: DType,
{
}

/// The framework's single extension point: implement this to add a new compute backend.
///
/// `Tensor<S, B, K, G>` is generic over `B: Backend` and stores exactly one
/// `B::Storage<K>` handle.
pub trait Backend:
    StorageBackend
        + Capabilities
        + HostInterop
        + AutogradBackend
        + Default
        + Sized
        + Clone
        + Send
        + Sync
        + 'static
{
    /// The backend actually doing the compute once any runtime-dispatch
    /// wrapper (see `Dyn`'s `DispatchBackend`) has been resolved. Equal to
    /// `Self` for every concrete (non-dispatching) backend.
    type InnerBackend: Backend;

    /// Compatibility formatting hook; new capability-aware code should use
    /// [`HostInterop::host_format_display`] directly.
    fn format_tensor_display<K: DType>(
        storage: &<Self as StorageBackend>::Storage<K>,
    ) -> alloc::string::String
    where
        Self: TensorOps<Self>,
    {
        <Self as HostInterop>::host_format_display(storage)
    }

    /// Compatibility formatting hook for diagnostic output.
    fn format_tensor_debug<K: DType>(
        storage: &<Self as StorageBackend>::Storage<K>,
    ) -> alloc::string::String
    where
        Self: TensorOps<Self>,
    {
        <Self as HostInterop>::host_format_debug(storage)
    }

}
