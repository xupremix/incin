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
    StorageBackend + Capabilities + Default + Sized + Clone + Send + Sync + 'static
{
    /// Backend-native handle for a trainable variable (as opposed to a
    /// plain, non-owning tensor view).
    type RawVar: Clone;
    /// The gradient collection returned by `backward`, indexed however the
    /// backend's own tape implementation chooses (usually by tensor id).
    type Grads;

    /// The backend actually doing the compute once any runtime-dispatch
    /// wrapper (see `Dyn`'s `DispatchBackend`) has been resolved. Equal to
    /// `Self` for every concrete (non-dispatching) backend.
    type InnerBackend: Backend;

    /// Renders a tensor's values for `Display` (concise, human-facing), the
    /// PyTorch-style bracketed grid `Tensor`'s own `Display` wraps in
    /// `tensor(...)` (`tensor/base.rs`).
    ///
    /// The default reads every element back through [`TensorOps::float_to_vec1`]
    /// or [`TensorOps::int_to_vec1`] (chosen by [`storage_dtype`](Self::storage_dtype))
    /// and hands them to the crate's own renderer, which stays private because
    /// the grid format is not something a backend author gets to depend on. A
    /// backend only needs to override this if reading every element back to the
    /// host is not how it wants to support printing.
    fn format_tensor_display<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> alloc::string::String
    where
        Self: TensorOps<Self>,
    {
        use crate::tensor::display::{Values, render};
        let shape = Self::shape(t);
        match Self::storage_dtype(t) {
            None => alloc::format!("<tensor: shape={shape:?}, dtype unknown to this backend>"),
            Some(dtype) if dtype.is_quantized() => alloc::format!(
                "<{} tensor: shape={shape:?}, not printable without dequantizing>",
                dtype.name()
            ),
            Some(dtype) if dtype.is_integer() => match Self::int_to_vec1(t) {
                Ok(values) => render(&shape, &Values::Int(values)),
                Err(err) => alloc::format!("<tensor: shape={shape:?}, values unavailable: {err}>"),
            },
            Some(_) => match Self::float_to_vec1(t) {
                Ok(values) => render(&shape, &Values::Float(values)),
                Err(err) => alloc::format!("<tensor: shape={shape:?}, values unavailable: {err}>"),
            },
        }
    }
    /// Renders a tensor's values and metadata for `Debug` (verbose,
    /// diagnostic-facing --- shape/dtype/device alongside the data).
    ///
    /// `Tensor`'s own `Debug` (`tensor/base.rs`) already prints shape,
    /// placement and rank on its own line before calling this, so the
    /// default here is exactly [`format_tensor_display`](Self::format_tensor_display) ---
    /// the same value grid, not a second copy of the metadata.
    fn format_tensor_debug<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> alloc::string::String
    where
        Self: TensorOps<Self>,
    {
        Self::format_tensor_display::<K>(t)
    }

    /// Runs backpropagation from `t` through the backend's recorded tape,
    /// returning the resulting per-tensor gradients.
    ///
    /// There is one of these since `GRD-005`. Whether the pass checks its
    /// gradients for a non-finite value is
    /// [`NanPolicy`](crate::exec::NanPolicy), an ambient execution-policy axis
    /// every backend's walk reads, rather than a second method that also
    /// decided to abort the process on failure.
    fn backward<K: DType>(_t: &<Self as StorageBackend>::Storage<K>) -> Result<Self::Grads> {
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "backward",
            },
        )))
    }

    /// Runs backpropagation with an explicit output cotangent.
    fn backward_with<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _seed: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<Self::Grads> {
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "seeded backward",
            },
        )))
    }
    /// Looks up the gradient computed for `t` in a `Grads` collection
    /// returned by `backward`. `None` if `t` received no gradient (e.g. it
    /// wasn't reachable from the tensor `backward` was called on).
    fn get_grad<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _grads: &Self::Grads,
    ) -> Result<Option<<Self as StorageBackend>::Storage<K>>> {
        Ok(None)
    }

    /// Serializes storage to a flat, dtype-native byte buffer (row-major,
    /// no header) --- the inverse of `from_bytes`.
    fn to_bytes<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<u8>> {
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "tensor readback",
            },
        )))
    }
    /// Reconstructs storage from raw bytes produced by `to_bytes`,
    /// validating that `bytes.len()` matches `shape`/`dtype`'s expected size.
    fn from_bytes<K: DType>(
        _bytes: &[u8],
        _shape: &[usize],
        _dtype: DTypeDescriptor,
        _device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "tensor creation from bytes",
            },
        )))
    }

    /// Views a trainable variable as a plain tensor storage handle.
    fn var_as_tensor<K: DType>(
        _var: &Self::RawVar,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "trainable variables",
            },
        )))
    }
    /// Promotes a plain tensor storage handle into a trainable variable.
    fn var_from_tensor<K: DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<Self::RawVar> {
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "trainable variables",
            },
        )))
    }
    /// Overwrites a variable's value in place (e.g. an optimizer step),
    /// without changing its identity for gradient-tracking purposes.
    ///
    /// An implementation must be failure-atomic for this individual variable:
    /// returning `Err` guarantees that `var` still contains its exact prior
    /// bytes. Optimizers rely on that contract to roll back a multi-parameter
    /// commit when a later assignment fails.
    fn assign_var<K: DType>(
        _var: &mut Self::RawVar,
        _tensor: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<()> {
        Err(crate::err::Error::Backend(BackendError::unsupported(
            Self::BACKEND_NAME,
            crate::exec::UnsupportedReason::MissingDeviceFeature {
                feature: "trainable variables",
            },
        )))
    }
}
