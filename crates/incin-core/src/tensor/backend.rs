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

// FloatOps only requires Backend, operates on FloatTensorPrimitive
/// Elementwise floating-point operations: activation functions and
/// scalar-broadcast arithmetic. Every method is required; a backend without a
/// kernel for one declares it, rather than inheriting a refusal.
pub trait FloatOps<B: Backend> {
    /// Rectified linear unit: `max(0, x)`.
    fn relu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Heaviside step function: `1` where `x > 0`, else `0`.
    fn step<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Mish activation: `x * tanh(softplus(x))`.
    fn mish<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Exponential Linear Unit: `x` where `x > 0`, else `exp(x) - 1`.
    fn elu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Gaussian Error Linear Unit (exact, erf-based):
    /// `x * 0.5 * (1 + erf(x / sqrt(2)))`.
    fn gelu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise absolute value.
    fn abs<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise natural exponential.
    fn exp<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise negation: `-x`.
    fn neg<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise square root.
    fn sqrt<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise natural logarithm.
    fn log<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise hyperbolic tangent.
    fn tanh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise logistic sigmoid: `1 / (1 + exp(-x))`.
    fn sigmoid<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Swish/SiLU activation: `x * sigmoid(x)`.
    fn swish<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Softmax along `dim`, numerically stabilized by subtracting the
    /// per-slice max before exponentiating.
    fn softmax<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Adds scalar `scalar` to every element.
    fn add_scalar_float<K: DType>(_t: &B::Storage<K>, _scalar: f64) -> Result<B::Storage<K>>;
    /// Multiplies every element by scalar `scalar`.
    fn mul_scalar_float<K: DType>(_t: &B::Storage<K>, _scalar: f64) -> Result<B::Storage<K>>;
    /// Elementwise power by float exponent `exponent`.
    fn powf<K: DType>(_t: &B::Storage<K>, _exponent: f64) -> Result<B::Storage<K>>;
    /// Elementwise clamp to `[min, max]`.
    fn clamp<K: DType>(_t: &B::Storage<K>, _min: f64, _max: f64) -> Result<B::Storage<K>>;
    /// Elementwise sign (-1.0 for negative, 0.0 for zero, +1.0 for positive).
    fn sign<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise floor.
    fn floor<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise ceil.
    fn ceil<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise round.
    fn round<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise base-2 logarithm.
    fn log2<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise base-10 logarithm.
    fn log10<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise sine.
    fn sin<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise cosine.
    fn cos<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise tangent.
    fn tan<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise arcsine.
    fn asin<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise arccosine.
    fn acos<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise arctangent.
    fn atan<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise 2-argument arctangent.
    fn atan2<K: DType>(_y: &B::Storage<K>, _x: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise hyperbolic sine.
    fn sinh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise hyperbolic cosine.
    fn cosh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise inverse hyperbolic sine.
    fn asinh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise inverse hyperbolic cosine.
    fn acosh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise inverse hyperbolic tangent.
    fn atanh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise error function.
    fn erf<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise reciprocal square root: 1 / sqrt(x).
    fn rsqrt<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise truncation toward zero.
    fn trunc<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise fractional part: x - trunc(x).
    fn frac<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise floating point remainder `x % y`.
    fn fmod<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise IEEE remainder.
    fn remainder<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
}

// NumericOps operates generically over any TensorKind!
/// Elementwise binary arithmetic with NumPy-style broadcasting (any
/// mismatched dimension must be size 1 on one side).
pub trait NumericOps<B: Backend> {
    /// Elementwise addition: `lhs + rhs`.
    fn add<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise subtraction: `lhs - rhs`.
    fn sub<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise multiplication: `lhs * rhs`.
    fn mul<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise division: `lhs / rhs`.
    fn div<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
}

/// Shape, layout, and dtype manipulation that doesn't change element
/// values (aside from `tensor_to_dtype`'s cast) --- reshapes, views,
/// concatenation, and host-readback conversions.
pub trait TensorOps<B: Backend> {
    /// Reinterprets storage under a new `shape` with the same element count
    /// and row-major ordering (no data movement on backends with
    /// contiguous storage).
    fn reshape<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>>;
    /// Swaps dimensions `dim1` and `dim2` in the logical shape (a view, not
    /// a copy, on backends with strided storage).
    fn transpose<K: DType>(_t: &B::Storage<K>, _dim1: usize, _dim2: usize)
    -> Result<B::Storage<K>>;
    /// Batched matrix multiplication over the trailing two dimensions of
    /// `lhs`/`rhs`, broadcasting any leading batch dimensions.
    fn matmul<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Broadcasts `t` to `shape` per NumPy rules (each dimension where the
    /// source size differs from the target must be exactly 1).
    fn broadcast_as<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>>;
    /// Takes the `len`-element window `[start, start + len)` along `dim`,
    /// keeping every other dimension unchanged.
    fn narrow<K: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _start: usize,
        _len: usize,
    ) -> Result<B::Storage<K>>;
    /// Removes dimension `dim`, which must have size 1.
    fn squeeze<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Stacks same-shaped tensors along a brand-new dimension inserted at
    /// `dim` (output has one more dimension than each input).
    fn stack<K: DType>(_t: &[&B::Storage<K>], _dim: usize) -> Result<B::Storage<K>>;
    /// Concatenates tensors along an existing dimension `dim` (every other
    /// dimension must already match across inputs).
    fn concat<K: DType>(_t: &[&B::Storage<K>], _dim: usize) -> Result<B::Storage<K>>;
    /// Takes a `[start, end)` window per dimension, one `(start, end)` pair
    /// in `ranges` for each dimension of `t`, in order.
    fn slice<K: DType>(_t: &B::Storage<K>, _ranges: &[(usize, usize)]) -> Result<B::Storage<K>>;
    /// Collapses dimensions `[start_dim, end_dim]` (inclusive) into a
    /// single dimension, preserving element order.
    fn flatten<K: DType>(
        _t: &B::Storage<K>,
        _start_dim: usize,
        _end_dim: usize,
    ) -> Result<B::Storage<K>>;
    /// Selects elements from `on_true` where `mask` is true, and `on_false` elsewhere.
    fn where_cond<K: DType>(
        _mask: &B::Storage<bool>,
        _on_true: &B::Storage<K>,
        _on_false: &B::Storage<K>,
    ) -> Result<B::Storage<K>>;
    /// Gathers values along `dim` using `index` tensor.
    fn gather<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _index: &B::Storage<KInt>,
    ) -> Result<B::Storage<K>>;
    /// Scatters `src` values along `dim` into `t` using `index` tensor.
    fn scatter<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _index: &B::Storage<KInt>,
        _src: &B::Storage<K>,
    ) -> Result<B::Storage<K>>;
    /// Selects slice along `dim` according to 1D `index` tensor.
    fn index_select<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _index: &B::Storage<KInt>,
    ) -> Result<B::Storage<K>>;
    /// Fills elements of `t` where `mask` is true with scalar `value`.
    fn masked_fill<K: DType>(
        _t: &B::Storage<K>,
        _mask: &B::Storage<bool>,
        _value: f64,
    ) -> Result<B::Storage<K>>;
    /// Inserts a 1-sized dimension at `dim`.
    fn unsqueeze<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Repeats tensor data along each dimension according to `repeats`.
    fn repeat<K: DType>(_t: &B::Storage<K>, _repeats: &[usize]) -> Result<B::Storage<K>>;
    /// Pads tensor with `val` according to `padding` (before, after) pairs.
    fn pad<K: DType>(
        _t: &B::Storage<K>,
        _padding: &[(usize, usize)],
        _val: f64,
    ) -> Result<B::Storage<K>>;
    /// Retains upper triangular part of matrix, zeroing the rest.
    fn triu<K: DType>(_t: &B::Storage<K>, _k: i64) -> Result<B::Storage<K>>;
    /// Retains lower triangular part of matrix, zeroing the rest.
    fn tril<K: DType>(_t: &B::Storage<K>, _k: i64) -> Result<B::Storage<K>>;
    /// Extracts diagonal or constructs diagonal matrix.
    fn diag<K: DType>(_t: &B::Storage<K>, _k: i64) -> Result<B::Storage<K>>;
    /// Reads a single-element floating-point tensor's value as `f64`.
    /// Errors if `t` has more than one element.
    fn float_to_scalar<K: DType>(_t: &B::Storage<K>) -> Result<f64>;

    /// Element-wise equality (`self == rhs`).
    fn cmp_eq<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;
    /// Element-wise inequality (`self != rhs`).
    fn cmp_ne<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;
    /// Element-wise less-than (`self < rhs`).
    fn cmp_lt<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;
    /// Element-wise less-than-or-equal (`self <= rhs`).
    fn cmp_le<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;
    /// Element-wise greater-than (`self > rhs`).
    fn cmp_gt<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;
    /// Element-wise greater-than-or-equal (`self >= rhs`).
    fn cmp_ge<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;

    /// Logical AND.
    fn logical_and(_lhs: &B::Storage<bool>, _rhs: &B::Storage<bool>) -> Result<B::Storage<bool>>;
    /// Logical OR.
    fn logical_or(_lhs: &B::Storage<bool>, _rhs: &B::Storage<bool>) -> Result<B::Storage<bool>>;
    /// Logical NOT.
    fn logical_not(_t: &B::Storage<bool>) -> Result<B::Storage<bool>>;

    /// Subtract scalar (`self - scalar`).
    fn sub_scalar<K: DType>(_t: &B::Storage<K>, _val: f64) -> Result<B::Storage<K>>;
    /// Divide scalar (`self / scalar`).
    fn div_scalar<K: DType>(_t: &B::Storage<K>, _val: f64) -> Result<B::Storage<K>>;

    /// Element-wise maximum of two tensors.
    fn maximum<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Element-wise minimum of two tensors.
    fn minimum<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Element-wise absolute difference `|lhs - rhs|`.
    fn abs_diff<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Linear interpolation `start + weight * (end - start)`.
    fn lerp<K: DType>(
        _start: &B::Storage<K>,
        _end: &B::Storage<K>,
        _weight: f64,
    ) -> Result<B::Storage<K>>;

    /// Fused add-matmul: `beta * mat + alpha * (mat1 x mat2)`.
    fn addmm<K: DType>(
        _mat: &B::Storage<K>,
        _mat1: &B::Storage<K>,
        _mat2: &B::Storage<K>,
        _beta: f64,
        _alpha: f64,
    ) -> Result<B::Storage<K>>;
    /// Batched matrix multiplication for 3D tensors.
    fn bmm<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Scaled Dot-Product Attention: `softmax(q * k^T / scale) * v`.
    fn scaled_dot_product_attention<K: DType>(
        _q: &B::Storage<K>,
        _k: &B::Storage<K>,
        _v: &B::Storage<K>,
        _mask: Option<&B::Storage<K>>,
        _scale: Option<f64>,
    ) -> Result<B::Storage<K>>;

    /// Sliding window extraction along `dim`.
    fn unfold<K: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> Result<B::Storage<K>>;
    /// Pixel shuffle upscaling for 4D (N, C, H, W) tensors.
    fn pixel_shuffle<K: DType>(_t: &B::Storage<K>, _upscale_factor: usize)
    -> Result<B::Storage<K>>;
    /// Group normalization across `groups`.
    fn group_norm<K: DType>(_t: &B::Storage<K>, _groups: usize, _eps: f64)
    -> Result<B::Storage<K>>;
    /// Instance normalization for 4D (N, C, H, W) tensors.
    fn instance_norm<K: DType>(_t: &B::Storage<K>, _eps: f64) -> Result<B::Storage<K>>;

    /// Prepends size-1 dimensions on the left until `t` has as many
    /// dimensions as `shape`, then broadcasts to `shape` (the NumPy
    /// "align on the right" convention for broadcasting mismatched ranks).
    fn broadcast_left<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>>;
    /// Reads a 1-D floating-point tensor's values into a host `Vec<f64>`.
    fn float_to_vec1<K: DType>(_t: &B::Storage<K>) -> Result<alloc::vec::Vec<f64>>;

    /// Reads a single-element integer tensor's value as `i64`. Errors if
    /// `t` has more than one element.
    fn int_to_scalar<K: DType>(_t: &B::Storage<K>) -> Result<i64>;
    /// Reads a 1-D integer tensor's values into a host `Vec<i64>`.
    fn int_to_vec1<K: DType>(_t: &B::Storage<K>) -> Result<alloc::vec::Vec<i64>>;

    /// Casts storage from dtype `K` to dtype `K2`, converting element
    /// values (not a bit-reinterpret --- see `dtype` for the target's
    /// `DTypeId`).
    fn tensor_to_dtype<K: DType, K2: DType>(
        _t: &B::Storage<K>,
        _dtype: DTypeDescriptor,
    ) -> Result<B::Storage<K2>>;
}

/// Allocates fresh storage and trainable variables --- the only place new
/// tensor data can originate from (every other op transforms existing
/// storage).
pub trait CreationOps<B: Backend> {
    /// Allocates a `shape`-sized tensor of `dtype`, filled with zero.
    fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::Storage<K>>;
    /// Allocates a `shape`-sized tensor of `dtype`, filled with one.
    fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::Storage<K>>;
    /// Allocates a `shape`-sized tensor of `dtype`, filled with samples
    /// from `Uniform(0, 1)`.
    fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::Storage<K>>;
    /// Allocates a `shape`-sized tensor of `dtype`, filled with samples
    /// from the standard normal distribution `N(0, 1)`.
    fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::Storage<K>>;

    /// Same as `zeros`, but returns a trainable `RawVar` directly.
    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::RawVar>;
    /// Same as `ones`, but returns a trainable `RawVar` directly.
    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::RawVar>;
    /// Same as `rand`, but returns a trainable `RawVar` directly.
    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::RawVar>;
    /// Same as `randn`, but returns a trainable `RawVar` directly.
    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::RawVar>;
    /// Allocates a `shape`-sized tensor filled with `val`.
    fn full<K: DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::Storage<K>>;
    /// Allocates a 1D tensor with values from `start` with step `step`.
    fn arange<K: DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::Storage<K>>;
    /// Allocates a 1D tensor of `shape` with linearly spaced values between `start` and `end`.
    fn linspace<K: DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<B::Storage<K>>;
}

/// Reductions that collapse a tensor along one or all dimensions ---
/// aggregate statistics (`sum`/`mean`/`max`/`min`) and index-producing
/// selections (`argmax`/`argmin`/`topk`/`argsort`).
pub trait ReductionOps<B: Backend> {
    /// Sums every element into a single-element tensor.
    fn sum_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Averages every element into a single-element tensor.
    fn mean_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Reduces to the single largest element.
    fn max_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Reduces to the single smallest element.
    fn min_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Sums along `dim`, removing that dimension from the output shape.
    fn sum_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Sums along `dim`, keeping it in the output shape as size 1.
    fn sum_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Averages along `dim`, removing that dimension from the output shape.
    fn mean_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Averages along `dim`, keeping it in the output shape as size 1.
    fn mean_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Reduces along `dim` to its max, removing that dimension.
    fn max_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Reduces along `dim` to its max, keeping it in the output shape as
    /// size 1.
    fn max_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Reduces along `dim` to its min, removing that dimension.
    fn min_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Reduces along `dim` to its min, keeping it in the output shape as
    /// size 1.
    fn min_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Index of the maximum element, either flattened (`dim: None`) or
    /// along a single `dim`.
    fn argmax<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: Option<usize>,
    ) -> Result<B::Storage<KInt>>;
    /// Index of the minimum element, either flattened (`dim: None`) or
    /// along a single `dim`.
    fn argmin<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: Option<usize>,
    ) -> Result<B::Storage<KInt>>;
    /// Product of all elements in tensor.
    fn prod_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Product of elements along `dim`.
    fn prod_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Cumulative sum along `dim`.
    fn cumsum<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// The `k` largest (`largest: true`) or smallest (`largest: false`)
    /// elements along `dim`, returned as `(values, indices)`.
    fn topk<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(B::Storage<K>, B::Storage<KInt>)>;
    /// Indices that would sort `t` along `dim`, ascending or `descending`.
    fn argsort<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<B::Storage<KInt>>;
}

/// Neural-network layer primitives: normalization, embedding lookup,
/// convolution, and pooling. Each takes plain storage (not `Param`/`Module`
/// wrappers) --- the `nn` layer types call through to these.
pub trait ModuleOps<B: Backend> {
    /// Layer normalization over the last dimension: normalizes `t` to zero
    /// mean/unit variance (with `eps` added for numerical stability), then
    /// applies an affine `weight` scale and optional `bias` shift.
    fn layer_norm<K: DType>(
        t: &B::Storage<K>,
        weight: &B::Storage<K>,
        bias: Option<&B::Storage<K>>,
        eps: f32,
    ) -> Result<B::Storage<K>>;
    /// Batch normalization over the channel dimension: normalizes using
    /// batch statistics (training) or `rm`/`rv` running mean/variance
    /// (inference), with `momentum` controlling running-stat updates and
    /// optional affine `w`/`b`.
    fn batch_norm<K: DType>(
        t: &B::Storage<K>,
        w: Option<&B::Storage<K>>,
        b: Option<&B::Storage<K>>,
        rm: Option<&B::Storage<K>>,
        rv: Option<&B::Storage<K>>,
        e: f32,
        momentum: f64,
    ) -> Result<B::Storage<K>>;
    /// Embedding table lookup: gathers rows of the weight matrix `w` at
    /// the integer indices in `t`.
    fn embedding<K: DType, KInt: DType>(
        t: &B::Storage<KInt>,
        w: &B::Storage<K>,
    ) -> Result<B::Storage<K>>;
    /// 1-D convolution of `t` with kernel `w` (and optional bias `b`),
    /// with the given `stride`/`padding`/`dilation`/`groups`.
    fn conv1d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    /// 2-D convolution of `t` with kernel `w` (and optional bias `b`),
    /// with the given `stride`/`padding`/`dilation`/`groups`.
    fn conv2d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    /// Transposed ("deconvolution") 2-D convolution --- the gradient
    /// operation of `conv2d` used as a forward op for upsampling, with an
    /// extra `output_padding` to resolve the otherwise-ambiguous output size.
    fn conv_transpose2d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    /// 2-D max pooling: for each output position, the max over its
    /// `kernel_size` window (given `stride`/`padding`/`dilation`).
    fn max_pool2d<K: DType>(
        t: &B::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<B::Storage<K>>;
    /// 2-D average pooling: for each output position, the mean over its
    /// `kernel_size` window (given `stride`/`padding`).
    fn avg_pool2d<K: DType>(
        t: &B::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<B::Storage<K>>;
    /// Average pooling that derives its own window size per output
    /// position so the output spatial size is exactly `output_size`,
    /// regardless of the input size (PyTorch's `AdaptiveAvgPool2d`).
    fn adaptive_avg_pool2d<K: DType>(
        t: &B::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<B::Storage<K>>;
}

/// Loss functions, given a default implementation in terms of
/// `NumericOps`/`FloatOps`/`ReductionOps` so a backend implementing those
/// gets working losses for free (override individually only for a
/// backend-specific fused kernel).
pub trait LossOps<B: Backend + NumericOps<B> + FloatOps<B> + ReductionOps<B>>:
    NumericOps<B> + FloatOps<B> + ReductionOps<B>
{
    /// Mean/sum/none-reduced squared error: `(pred - target)^2`.
    fn mse_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: crate::tensor::reduction::Reduction,
    ) -> Result<B::Storage<K>> {
        let diff = <B as NumericOps<B>>::sub::<K>(pred, target)?;
        let sq = <B as NumericOps<B>>::mul::<K>(&diff, &diff)?;
        match reduction {
            crate::tensor::reduction::Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&sq),
            crate::tensor::reduction::Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&sq),
            crate::tensor::reduction::Reduction::None => Ok(sq),
        }
    }

    /// Mean/sum/none-reduced absolute error: `|pred - target|`.
    fn l1_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: crate::tensor::reduction::Reduction,
    ) -> Result<B::Storage<K>> {
        let diff = <B as NumericOps<B>>::sub::<K>(pred, target)?;
        let abs_diff = <B as FloatOps<B>>::abs::<K>(&diff)?;
        match reduction {
            crate::tensor::reduction::Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&abs_diff),
            crate::tensor::reduction::Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&abs_diff),
            crate::tensor::reduction::Reduction::None => Ok(abs_diff),
        }
    }

    /// Binary cross-entropy computed from raw logits (`pred`, pre-sigmoid),
    /// using the numerically stable formulation
    /// `max(x,0) - x*z + log(1 + exp(-|x|))` so it never evaluates
    /// `sigmoid`/`log` on the raw logit directly.
    fn bce_with_logits_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: crate::tensor::reduction::Reduction,
    ) -> Result<B::Storage<K>> {
        let max_x_0 = <B as FloatOps<B>>::relu::<K>(pred)?;
        let x_times_z = <B as NumericOps<B>>::mul::<K>(pred, target)?;
        let term1 = <B as NumericOps<B>>::sub::<K>(&max_x_0, &x_times_z)?;

        let abs_x = <B as FloatOps<B>>::abs::<K>(pred)?;
        let neg_abs_x = <B as FloatOps<B>>::neg::<K>(&abs_x)?;
        let exp_neg_abs_x = <B as FloatOps<B>>::exp::<K>(&neg_abs_x)?;
        let one_plus = <B as FloatOps<B>>::add_scalar_float::<K>(&exp_neg_abs_x, 1.0)?;
        let term2 = <B as FloatOps<B>>::log::<K>(&one_plus)?;

        let loss_elem = <B as NumericOps<B>>::add::<K>(&term1, &term2)?;

        match reduction {
            crate::tensor::reduction::Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&loss_elem),
            crate::tensor::reduction::Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&loss_elem),
            crate::tensor::reduction::Reduction::None => Ok(loss_elem),
        }
    }

    /// Cross-entropy loss between raw `pred` logits (softmax applied
    /// internally) and integer class-index `target`s --- no default
    /// implementation, since it needs a numerically-stable fused
    /// log-softmax rather than composing `softmax` + `log` naively.
    fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<KInt>,
        reduction: crate::tensor::reduction::Reduction,
    ) -> Result<B::Storage<K>>;
}

/// Block quantization: compresses `FloatDType` storage into a `QuantDType`
/// representation for reduced memory footprint, and the reverse.
pub trait QuantizedOps<B: Backend> {
    /// Compresses `t` from a float dtype into quantized storage `Q`.
    fn quantize<K: FloatDType, Q: QuantDType>(_t: &B::Storage<K>) -> Result<B::Storage<Q>>;
    /// Expands quantized storage `Q` back into a float dtype `K`
    /// (lossy --- the inverse of `quantize` only up to quantization error).
    fn dequantize<Q: QuantDType, K: FloatDType>(_t: &B::Storage<Q>) -> Result<B::Storage<K>>;
    /// Matrix multiplication of two quantized-storage operands, producing
    /// `f32` output without needing to fully dequantize both operands first.
    fn quantized_matmul<Q: QuantDType>(
        lhs: &B::Storage<Q>,
        rhs: &B::Storage<Q>,
    ) -> Result<B::Storage<f32>>;
}

/// A minimal, allocation-free `Backend` implementation used only by unit
/// tests elsewhere in this crate that need a concrete `Backend` type
/// without depending on `incin-backends`. See `DummyBackend`.
pub mod dummy {
    use super::*;
    use crate::exec::spec::ExecutionDescriptor;
    use crate::tensor::reduction::Reduction;
    use crate::prelude::Result;
    use crate::tensor::device::Device;
    use crate::tensor::device::DeviceId;
    use crate::tensor::dtype::DType;

    /// Test-only stand-in `Backend` used by `tensor/base.rs`'s unit tests to
    /// exercise `Tensor`'s generic-over-`Backend` machinery without pulling
    /// in a real compute backend. Its `Storage<K>` is literally the shape
    /// (`Vec<usize>`) --- every op below tracks how an operation would
    /// transform the *shape*, using the same arithmetic real backends use,
    /// but performs no actual data computation and holds no element values.
    ///
    /// Dtype is not part of the backend's identity --- a single `DummyBackend<D>`
    /// can hold f32, f64, i64, etc. tensors, all represented as shape-only `Vec<usize>`.
    pub struct DummyBackend<D> {
        _marker: core::marker::PhantomData<D>,
    }

    impl<D: Device + Clone + 'static> Default for DummyBackend<D> {
        fn default() -> Self {
            DummyBackend {
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<D: Device + Clone + 'static> Capabilities for DummyBackend<D> {
        fn support(&self, _query: &crate::exec::CapabilityQuery) -> crate::exec::SupportLevel {
            crate::exec::SupportLevel::Native
        }
    }

    impl<D: Device + Clone + 'static> Clone for DummyBackend<D> {
        /// Cheap: the type carries no state beyond its `PhantomData` markers.
        fn clone(&self) -> Self {
            DummyBackend {
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<D: Device + Clone + 'static> StorageBackend for DummyBackend<D> {
        const BACKEND_NAME: &'static str = "dummy";
        /// The device type this stand-in is parameterized over.
        type Device = D;
        /// Shape-only storage: `Storage<K>` is the tensor's shape, not its
        /// values, regardless of `K`.
        type Storage<K: DType> = alloc::vec::Vec<usize>;

        fn metadata<K: DType>(storage: &<Self as StorageBackend>::Storage<K>) -> &TensorMeta {
            let shape_buf = crate::shapes::ShapeBuf::from_slice(storage);
            let numel = shape_buf.numel().unwrap_or(0);
            let dtype = K::descriptor(&K::Field::default());
            alloc::boxed::Box::leak(alloc::boxed::Box::new(
                TensorMeta::contiguous(
                    shape_buf,
                    dtype,
                    DeviceId::cpu(),
                    crate::exec::meta::Alignment::of::<f32>(),
                    numel,
                )
                .expect("valid dummy metadata"),
            ))
        }
    }

    impl<D: Device + Clone + 'static, O: crate::exec::catalog::Operation> Execute<O>
        for DummyBackend<D>
    {
        type Output = alloc::vec::Vec<usize>;

        fn execute(
            &self,
            request: ExecutionRequest<'_, O, Self>,
        ) -> core::result::Result<Self::Output, BackendError> {
            if let Some(output) = request.operation.descriptor().output_shape() {
                return Ok(output.dims().to_vec());
            }
            if let Some(input) = request.inputs.first() {
                Ok(input.metadata().shape.dims().to_vec())
            } else {
                Ok(alloc::vec![])
            }
        }
    }

    impl<D: Device + Clone + 'static> Backend for DummyBackend<D> {
        /// A trainable variable is just its shape, like `Storage`.
        type RawVar = alloc::vec::Vec<usize>;
        /// No real gradients are tracked, so this carries no data.
        type Grads = ();
        /// No dispatch wrapper --- this stand-in is always its own inner backend.
        type InnerBackend = Self;

        /// Always `"dummy"` --- there are no real values to render.
        fn format_tensor_display<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }
        /// Always `"dummy"` --- there are no real values to render.
        fn format_tensor_debug<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }
        /// No-op: there is no tape to run backward through.
        fn backward<K: DType>(_t: &<Self as StorageBackend>::Storage<K>) -> Result<Self::Grads> {
            Ok(())
        }
        /// Always `None`: `Grads` carries no data to look a gradient up in.
        fn get_grad<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _grads: &Self::Grads,
        ) -> Result<Option<<Self as StorageBackend>::Storage<K>>> {
            Ok(None)
        }
        /// Always empty: there are no element values to serialize.
        fn to_bytes<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<u8>> {
            Ok(alloc::vec::Vec::new())
        }
        /// Ignores `bytes` entirely and reconstructs storage from `shape`
        /// alone, since `Storage<K>` only ever tracks shape.
        fn from_bytes<K: DType>(
            _bytes: &[u8],
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// `RawVar` and `Storage<K>` are the same representation, so this
        /// is a plain clone.
        fn var_as_tensor<K: DType>(
            var: &Self::RawVar,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(var.clone())
        }
        /// `RawVar` and `Storage<K>` are the same representation, so this
        /// is a plain clone.
        fn var_from_tensor<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<Self::RawVar> {
            Ok(t.clone())
        }
        /// Overwrites `var`'s shape with `tensor`'s.
        fn assign_var<K: DType>(
            var: &mut Self::RawVar,
            tensor: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<()> {
            *var = tensor.clone();
            Ok(())
        }
    }

    impl<D: Device + Clone + 'static, K: DType> SupportsDType<K> for DummyBackend<D> {
        fn resolve_dtype(field: &K::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
            Ok(K::descriptor(field))
        }
    }

    /// Output spatial size for conv/pool shape math:
    /// `(in + 2*pad - dilation*(kernel-1) - 1) / stride + 1`. Uses saturating
    /// arithmetic throughout (never panics/wraps on pathological inputs ---
    /// small `in` with a large `kernel`/`dilation`/`padding` would otherwise
    /// underflow the `usize` subtraction), matching the CPU backend's own
    /// `out_size` (`cpu/ops/pool.rs`), which already uses the same
    /// saturate-rather-than-error convention for this exact case. This is
    /// shape-only bookkeeping for `DummyBackend` (a test-only stand-in with
    /// no real storage), so a saturated/degenerate size is the appropriate
    /// "can't compute a real answer" response, not an error.
    fn conv_out_size(
        len: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> usize {
        let padded = len.saturating_add(2 * padding);
        let effective_kernel = dilation
            .saturating_mul(kernel_size.saturating_sub(1))
            .saturating_add(1);
        padded.saturating_sub(effective_kernel) / stride.max(1) + 1
    }

    /// Output spatial size for `conv_transpose2d` shape math:
    /// `(in - 1) * stride - 2*pad + dilation*(kernel-1) + output_padding + 1`.
    /// Same saturating-arithmetic rationale as `conv_out_size`.
    fn conv_transpose_out_size(
        len: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> usize {
        let strided = len.saturating_sub(1).saturating_mul(stride);
        let effective_kernel = dilation.saturating_mul(kernel_size.saturating_sub(1));
        strided
            .saturating_sub(2 * padding)
            .saturating_add(effective_kernel)
            .saturating_add(output_padding)
            .saturating_add(1)
    }

    impl<D: Device + Clone + 'static, NewD: Device + Clone + 'static> TransferTo<NewD>
        for DummyBackend<D>
    {
        type Output = DummyBackend<NewD>;

        fn transfer_storage<K: DType>(
            storage: &<Self as StorageBackend>::Storage<K>,
            _dtype: &K::Field,
            _device: &NewD::Field,
        ) -> Result<<Self::Output as StorageBackend>::Storage<K>>
        where
            Self::Output: SupportsDType<K>,
        {
            Ok(storage.clone())
        }

        fn transfer_var<K: DType>(
            variable: &Self::RawVar,
            _dtype: &K::Field,
            _device: &NewD::Field,
        ) -> Result<<Self::Output as Backend>::RawVar>
        where
            Self::Output: SupportsDType<K>,
        {
            Ok(variable.clone())
        }
    }

    /// Shape is preserved by every allocation, since it's the only thing
    /// `Storage`/`RawVar` track --- no real fill value is ever written.
    impl<D: Device + Clone + 'static> CreationOps<Self> for DummyBackend<D> {
        /// Returns `shape` verbatim as the storage handle.
        fn zeros<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn full<K: DType>(
            _val: f64,
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn arange<K: DType>(
            _start: f64,
            _step: f64,
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn linspace<K: DType>(
            _start: f64,
            _end: f64,
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn ones<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn rand<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn randn<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_zeros<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_ones<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_rand<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_randn<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
    }

    /// Every binary op broadcasts its two shapes the way a real backend does.
    ///
    /// These returned `lhs`'s shape unchanged until `UX-013`, which is wrong for
    /// the same reason it was invisible: `add` is reached with differently
    /// shaped operands only through `broadcast_add` and friends, which then
    /// hand the result to `Tensor::from_parts` against the *broadcast* type. A
    /// stand-in whose shape arithmetic disagrees with every real backend's is
    /// not a stand-in, and this crate's own documented examples of
    /// `broadcast_add` were the first thing to run into it.
    impl<D: Device + Clone + 'static> NumericOps<Self> for DummyBackend<D> {
        /// Returns the two operands' broadcast shape.
        fn add<K: DType>(
            lhs: &<Self as StorageBackend>::Storage<K>,
            rhs: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(crate::shapes::broadcast::broadcast_dim_slices(lhs, rhs)?)
        }
        /// Returns the two operands' broadcast shape.
        fn sub<K: DType>(
            lhs: &<Self as StorageBackend>::Storage<K>,
            rhs: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(crate::shapes::broadcast::broadcast_dim_slices(lhs, rhs)?)
        }
        /// Returns the two operands' broadcast shape.
        fn mul<K: DType>(
            lhs: &<Self as StorageBackend>::Storage<K>,
            rhs: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(crate::shapes::broadcast::broadcast_dim_slices(lhs, rhs)?)
        }
        /// Returns the two operands' broadcast shape.
        fn div<K: DType>(
            lhs: &<Self as StorageBackend>::Storage<K>,
            rhs: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(crate::shapes::broadcast::broadcast_dim_slices(lhs, rhs)?)
        }
    }

    /// The remaining float operations, all of which preserve their input's
    /// shape and so are the same clone as the ones written out above.
    ///
    /// `DummyBackend` exists to exercise shape behavior, so covering these by
    /// hand would add a hundred lines that all say `Ok(t.clone())`. They are
    /// listed rather than inherited because `FloatOps` no longer supplies a
    /// default body: an operation this backend does not model has to be
    /// visible here.
    macro_rules! shape_preserving_float_ops {
        (
            unary: $($unary:ident),* $(,)?;
            exponent: $($exponent:ident),* $(,)?;
            bounds: $($bounds:ident),* $(,)?;
            binary: $($binary:ident),* $(,)?;
        ) => {
            $(
                fn $unary<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $exponent<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                    _exponent: f64,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $bounds<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                    _min: f64,
                    _max: f64,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $binary<K: DType>(
                    lhs: &<Self as StorageBackend>::Storage<K>,
                    _rhs: &<Self as StorageBackend>::Storage<K>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(lhs.clone())
                }
            )*
        };
    }

    /// Every activation and scalar op is shape-preserving, so each is a
    /// plain clone of the input shape.
    impl<D: Device + Clone + 'static> FloatOps<Self> for DummyBackend<D> {
        /// Returns `t`'s shape unchanged.
        fn add_scalar_float<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn mul_scalar_float<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn relu<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn step<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn mish<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn elu<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn gelu<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn abs<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn exp<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn neg<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn sqrt<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn log<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn tanh<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn sigmoid<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn swish<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged (`dim` is not validated).
        fn softmax<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }

        shape_preserving_float_ops! {
            unary: sign, floor, ceil, round, log2, log10, sin, cos, tan, asin,
                   acos, atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt,
                   trunc, frac;
            exponent: powf;
            bounds: clamp;
            binary: atan2, fmod, remainder;
        }
    }

    /// `_all` reductions collapse to an (empty) scalar shape; `_dim`
    /// reductions either remove `dim` or clamp it to size 1 (`_keepdim`),
    /// exactly like a real reduction's shape effect --- again with no real
    /// values behind either result.
    impl<D: Device + Clone + 'static> ReductionOps<Self> for DummyBackend<D> {
        /// Collapses to an empty (scalar) shape.
        fn sum_all<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn mean_all<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn max_all<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn min_all<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn prod_all<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Removes `dim` from the shape.
        fn prod_dim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// A running sum along `dim` leaves the shape unchanged.
        fn cumsum<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Removes `dim` from the shape.
        fn sum_dim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn sum_keepdim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Removes `dim` from the shape.
        fn mean_dim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn mean_keepdim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Removes `dim` from the shape.
        fn max_dim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn max_keepdim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Removes `dim` from the shape.
        fn min_dim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn min_keepdim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Always an empty shape --- no indices are actually computed.
        fn argmax<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        /// Always an empty shape --- no indices are actually computed.
        fn argmin<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        /// Always an empty `(values, indices)` pair.
        fn topk<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _k: usize,
            _dim: usize,
            _largest: bool,
        ) -> Result<(
            <Self as StorageBackend>::Storage<K>,
            <Self as StorageBackend>::Storage<KInt>,
        )> {
            Ok((alloc::vec![], alloc::vec![]))
        }
        /// Always an empty shape --- no indices are actually computed.
        fn argsort<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _dim: usize,
            _descending: bool,
        ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
    }

    /// The `TensorOps` members whose output shape equals an input's, split by
    /// which operand supplies it. These mirror `NumericOps`' convention above,
    /// where a binary op reports `lhs`'s shape.
    macro_rules! shape_preserving_tensor_ops {
        (
            unary: $($unary:ident),* $(,)?;
            scalar: $($scalar:ident),* $(,)?;
            diagonal: $($diagonal:ident),* $(,)?;
            binary: $($binary:ident),* $(,)?;
        ) => {
            $(
                fn $unary<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $scalar<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                    _val: f64,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $diagonal<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                    _k: i64,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $binary<K: DType>(
                    lhs: &<Self as StorageBackend>::Storage<K>,
                    _rhs: &<Self as StorageBackend>::Storage<K>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(lhs.clone())
                }
            )*
        };
    }

    /// The `TensorOps` members whose output shape this stand-in does not
    /// model. Returning a plausible-looking wrong shape would be worse than
    /// refusing: shape is the only thing `DummyBackend` asserts, and a test
    /// reading a fabricated one would pass for the wrong reason.
    macro_rules! unmodeled_tensor_ops {
        (
            indexed: $($indexed:ident),* $(,)?;
            dim: $($dim:ident),* $(,)?;
            binary: $($binary:ident),* $(,)?;
        ) => {
            $(
                fn $indexed<K: DType, KInt: DType>(
                    _t: &<Self as StorageBackend>::Storage<K>,
                    _dim: usize,
                    _index: &<Self as StorageBackend>::Storage<KInt>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Err(crate::err::Error::UnsupportedBackendOperation {
                        op: stringify!($indexed),
                        backend: core::any::type_name::<Self>(),
                    })
                }
            )*
            $(
                fn $dim<K: DType>(
                    _t: &<Self as StorageBackend>::Storage<K>,
                    _dim: usize,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Err(crate::err::Error::UnsupportedBackendOperation {
                        op: stringify!($dim),
                        backend: core::any::type_name::<Self>(),
                    })
                }
            )*
            $(
                fn $binary<K: DType>(
                    _lhs: &<Self as StorageBackend>::Storage<K>,
                    _rhs: &<Self as StorageBackend>::Storage<K>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Err(crate::err::Error::UnsupportedBackendOperation {
                        op: stringify!($binary),
                        backend: core::any::type_name::<Self>(),
                    })
                }
            )*
        };
    }

    /// Each op tracks its real shape-transformation logic (matmul's last
    /// dim, transpose's swap, flatten's dimension collapse, etc.) since
    /// shape *is* everything this stand-in's storage represents --- but
    /// still no element values exist behind any of it.
    impl<D: Device + Clone + 'static> TensorOps<Self> for DummyBackend<D> {
        shape_preserving_tensor_ops! {
            unary: ;
            scalar: sub_scalar, div_scalar, instance_norm;
            diagonal: triu, tril;
            binary: maximum, minimum, abs_diff;
        }

        fn cmp_eq<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn cmp_ne<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn cmp_lt<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn cmp_le<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn cmp_gt<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn cmp_ge<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn logical_and(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn logical_or(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn logical_not(t: &alloc::vec::Vec<usize>) -> Result<alloc::vec::Vec<usize>> {
            Ok(t.clone())
        }

        unmodeled_tensor_ops! {
            indexed: gather, index_select;
            dim: unsqueeze, pixel_shuffle;
            binary: bmm;
        }

        /// Returns `on_true`'s shape, which is the branch the output takes.
        fn where_cond<K: DType>(
            _mask: &<Self as StorageBackend>::Storage<bool>,
            on_true: &<Self as StorageBackend>::Storage<K>,
            _on_false: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(on_true.clone())
        }

        /// Filling masked positions leaves the shape untouched.
        fn masked_fill<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _mask: &<Self as StorageBackend>::Storage<bool>,
            _value: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }

        /// Interpolating between two tensors keeps `start`'s shape.
        fn lerp<K: DType>(
            start: &<Self as StorageBackend>::Storage<K>,
            _end: &<Self as StorageBackend>::Storage<K>,
            _weight: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(start.clone())
        }

        /// Normalizing over groups leaves the shape untouched.
        fn group_norm<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _groups: usize,
            _eps: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }

        /// Not modeled: the output tiles each axis by its own factor.
        fn repeat<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _repeats: &[usize],
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "repeat",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: the output grows by the padding on each axis.
        fn pad<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _padding: &[(usize, usize)],
            _val: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "pad",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: `diag` both extracts and constructs, changing rank
        /// in opposite directions depending on the input.
        fn diag<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _k: i64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "diag",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: writes into a copy of the target, whose shape this
        /// stand-in would have to reconcile against the index and source.
        fn scatter<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _dim: usize,
            _index: &<Self as StorageBackend>::Storage<KInt>,
            _src: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "scatter",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: the fused product's shape follows `mat1 @ mat2`
        /// broadcast against `mat`.
        fn addmm<K: DType>(
            _mat: &<Self as StorageBackend>::Storage<K>,
            _mat1: &<Self as StorageBackend>::Storage<K>,
            _mat2: &<Self as StorageBackend>::Storage<K>,
            _beta: f64,
            _alpha: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "addmm",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: the output takes its trailing axis from `v`, not `q`.
        fn scaled_dot_product_attention<K: DType>(
            _q: &<Self as StorageBackend>::Storage<K>,
            _k: &<Self as StorageBackend>::Storage<K>,
            _v: &<Self as StorageBackend>::Storage<K>,
            _mask: Option<&<Self as StorageBackend>::Storage<K>>,
            _scale: Option<f64>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "scaled_dot_product_attention",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: sliding windows replace one axis with two.
        fn unfold<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _dim: usize,
            _size: usize,
            _step: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "unfold",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Broadcasts leading batch axes and applies the trailing matrix
        /// contraction, mirroring real matmul's output shape.
        fn matmul<K: DType>(
            lhs: &<Self as StorageBackend>::Storage<K>,
            rhs: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            if lhs.len() < 2 || rhs.len() < 2 || lhs[lhs.len() - 1] != rhs[rhs.len() - 2] {
                return Err(crate::err::Error::ShapeMismatch {
                    op: "matmul",
                    expected: lhs.clone(),
                    got: rhs.clone(),
                    msg: "matmul requires rank >= 2 and equal contraction dimensions".into(),
                });
            }

            let lhs_batch = &lhs[..lhs.len() - 2];
            let rhs_batch = &rhs[..rhs.len() - 2];
            let rank = lhs_batch.len().max(rhs_batch.len());
            let mut out = alloc::vec::Vec::with_capacity(rank + 2);
            for axis in 0..rank {
                let from_end = rank - axis;
                let l = lhs_batch
                    .len()
                    .checked_sub(from_end)
                    .map_or(1, |index| lhs_batch[index]);
                let r = rhs_batch
                    .len()
                    .checked_sub(from_end)
                    .map_or(1, |index| rhs_batch[index]);
                if l != r && l != 1 && r != 1 {
                    return Err(crate::err::Error::ShapeMismatch {
                        op: "matmul",
                        expected: lhs.clone(),
                        got: rhs.clone(),
                        msg: "matmul batch dimensions are not broadcast-compatible".into(),
                    });
                }
                out.push(if l == 1 { r } else { l });
            }
            out.extend_from_slice(&[lhs[lhs.len() - 2], rhs[rhs.len() - 1]]);
            Ok(out)
        }
        /// Always `0.0` --- there is no real element value to read.
        fn float_to_scalar<K: DType>(_t: &<Self as StorageBackend>::Storage<K>) -> Result<f64> {
            Ok(0.0)
        }
        /// Always a single `0.0` --- there are no real element values to read.
        fn float_to_vec1<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<f64>> {
            Ok(alloc::vec![0.0])
        }
        /// Always `0` --- there is no real element value to read.
        fn int_to_scalar<K: DType>(_t: &<Self as StorageBackend>::Storage<K>) -> Result<i64> {
            Ok(0)
        }
        /// Always a single `0` --- there are no real element values to read.
        fn int_to_vec1<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<i64>> {
            Ok(alloc::vec![0])
        }

        /// Returns the target `shape` verbatim (broadcast compatibility is
        /// not validated).
        fn broadcast_as<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(s.to_vec())
        }
        /// Prepends the target `shape`'s dimensions ahead of `t`'s own.
        fn broadcast_left<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = s.to_vec();
            out.extend_from_slice(t);
            Ok(out)
        }
        /// Returns the target `shape` verbatim (numel compatibility is not
        /// validated).
        fn reshape<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(s.to_vec())
        }
        /// Swaps dimensions `d1`/`d2` in the shape.
        fn transpose<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            d1: usize,
            d2: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            if d1 < out.len() && d2 < out.len() {
                out.swap(d1, d2);
            }
            Ok(out)
        }
        /// Collapses dimensions `[s, e]` into their product.
        fn flatten<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            s: usize,
            e: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            if s > e || e >= t.len() {
                return Ok(t.clone());
            }
            let mut out = t[..s].to_vec();
            out.push(
                crate::shapes::ShapeBuf::from_slice(&t[s..=e])
                    .checked_numel(crate::shapes::error::OperationKind::Flatten)?,
            );
            out.extend_from_slice(&t[e + 1..]);
            Ok(out)
        }
        /// Sets each dimension's size to its `(start, end)` window length.
        fn slice<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            ranges: &[(usize, usize)],
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            for (dim, &(start, end)) in ranges.iter().enumerate() {
                if dim < out.len() {
                    out[dim] = end.saturating_sub(start);
                }
            }
            Ok(out)
        }
        /// Sets dimension `d`'s size to the requested window length `l`.
        fn narrow<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            d: usize,
            _s: usize,
            l: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            if d < out.len() {
                out[d] = l;
            }
            Ok(out)
        }
        /// Removes dimension `d` from the shape.
        fn squeeze<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            d: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            if d < out.len() {
                out.remove(d);
            }
            Ok(out)
        }
        /// Inserts a new dimension at `d`, sized to the number of stacked
        /// tensors.
        fn stack<K: DType>(
            t: &[&<Self as StorageBackend>::Storage<K>],
            d: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            if t.is_empty() {
                return Ok(alloc::vec![]);
            }
            let mut out = t[0].clone();
            if d <= out.len() {
                out.insert(d, t.len());
            }
            Ok(out)
        }
        /// Sets dimension `d`'s size to the sum of every input's size
        /// along `d`.
        fn concat<K: DType>(
            t: &[&<Self as StorageBackend>::Storage<K>],
            d: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            if t.is_empty() {
                return Ok(alloc::vec![]);
            }
            let mut out = t[0].clone();
            if d < out.len() {
                out[d] = t.iter().map(|s| s.get(d).copied().unwrap_or(0)).sum();
            }
            Ok(out)
        }
        /// Returns `t`'s shape unchanged --- no element values exist to cast.
        fn tensor_to_dtype<K: DType, K2: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _dtype: DTypeDescriptor,
        ) -> Result<<Self as StorageBackend>::Storage<K2>> {
            Ok(t.clone())
        }
    }

    /// Normalization ops are shape-preserving no-ops; conv/pool ops
    /// compute their real output spatial size via `conv_out_size`/
    /// `conv_transpose_out_size` (the saturating helpers above) so tests
    /// can assert on shape correctness even though no data is computed.
    impl<D: Device + Clone + 'static> ModuleOps<Self> for DummyBackend<D> {
        /// Returns `t`'s shape unchanged.
        fn layer_norm<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _w: &<Self as StorageBackend>::Storage<K>,
            _b: Option<&<Self as StorageBackend>::Storage<K>>,
            _e: f32,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn batch_norm<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _w: Option<&<Self as StorageBackend>::Storage<K>>,
            _b: Option<&<Self as StorageBackend>::Storage<K>>,
            _rm: Option<&<Self as StorageBackend>::Storage<K>>,
            _rv: Option<&<Self as StorageBackend>::Storage<K>>,
            _e: f32,
            _m: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Always an empty shape --- no real gather is performed.
        fn embedding<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<KInt>,
            _w: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Computes the real output shape: channel dim from `w[0]`, spatial
        /// dim via `conv_out_size`.
        fn conv1d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            w: &<Self as StorageBackend>::Storage<K>,
            _b: Option<&<Self as StorageBackend>::Storage<K>>,
            s: usize,
            p: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 3 && w.len() >= 3 {
                let l_in = out[len - 1];
                let k = w[w.len() - 1];
                let c_out = w[0]; // Assuming [C_out, C_in / groups, K]
                out[len - 2] = c_out;
                out[len - 1] = conv_out_size(l_in, k, s, p, d);
            }
            Ok(out)
        }
        /// Computes the real output shape: channel dim from `w[0]`, spatial
        /// dims via `conv_out_size`.
        fn conv2d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            w: &<Self as StorageBackend>::Storage<K>,
            _b: Option<&<Self as StorageBackend>::Storage<K>>,
            s: usize,
            p: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 4 && w.len() >= 4 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                let k_h = w[w.len() - 2];
                let k_w = w[w.len() - 1];
                let c_out = w[0]; // [C_out, C_in / groups, K_H, K_W]
                out[len - 3] = c_out;
                out[len - 2] = conv_out_size(h_in, k_h, s, p, d);
                out[len - 1] = conv_out_size(w_in, k_w, s, p, d);
            }
            Ok(out)
        }
        /// Computes the real output shape: channel dim from `w[1]`
        /// (transposed conv's weight layout), spatial dims via
        /// `conv_transpose_out_size`.
        fn conv_transpose2d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            w: &<Self as StorageBackend>::Storage<K>,
            _b: Option<&<Self as StorageBackend>::Storage<K>>,
            s: usize,
            p: usize,
            op: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 4 && w.len() >= 4 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                let k_h = w[w.len() - 2];
                let k_w = w[w.len() - 1];
                let c_out = w[1]; // [C_in, C_out / groups, K_H, K_W]
                out[len - 3] = c_out;
                out[len - 2] = conv_transpose_out_size(h_in, k_h, s, p, op, d);
                out[len - 1] = conv_transpose_out_size(w_in, k_w, s, p, op, d);
            }
            Ok(out)
        }
        /// Computes the real output spatial shape via `conv_out_size`.
        fn max_pool2d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            k: (usize, usize),
            s: (usize, usize),
            p: (usize, usize),
            d: (usize, usize),
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 2 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                out[len - 2] = conv_out_size(h_in, k.0, s.0, p.0, d.0);
                out[len - 1] = conv_out_size(w_in, k.1, s.1, p.1, d.1);
            }
            Ok(out)
        }
        /// Computes the real output spatial shape via `conv_out_size`
        /// (dilation fixed to 1, matching `avg_pool2d` having no dilation
        /// parameter).
        fn avg_pool2d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            k: (usize, usize),
            s: (usize, usize),
            p: (usize, usize),
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 2 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                out[len - 2] = conv_out_size(h_in, k.0, s.0, p.0, 1);
                out[len - 1] = conv_out_size(w_in, k.1, s.1, p.1, 1);
            }
            Ok(out)
        }
        /// Sets the trailing two dimensions directly to `out` (adaptive
        /// pooling's whole point is that the output size is exact,
        /// regardless of input size).
        fn adaptive_avg_pool2d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            out: (usize, usize),
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut shape = t.clone();
            let len = shape.len();
            if len >= 2 {
                shape[len - 2] = out.0;
                shape[len - 1] = out.1;
            }
            Ok(shape)
        }
    }

    /// All four losses reduce to an empty (scalar) shape, ignoring
    /// `reduction`'s actual `None`/`Sum`/`Mean` distinction since there are
    /// no real values to reduce.
    impl<D: Device + Clone + 'static> LossOps<Self> for DummyBackend<D> {
        /// Always an empty (scalar) shape.
        fn mse_loss<K: DType>(
            _pred: &<Self as StorageBackend>::Storage<K>,
            _target: &<Self as StorageBackend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Always an empty (scalar) shape.
        fn l1_loss<K: DType>(
            _pred: &<Self as StorageBackend>::Storage<K>,
            _target: &<Self as StorageBackend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Always an empty (scalar) shape.
        fn bce_with_logits_loss<K: DType>(
            _pred: &<Self as StorageBackend>::Storage<K>,
            _target: &<Self as StorageBackend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Always an empty (scalar) shape.
        fn cross_entropy_loss<K: DType, KInt: DType>(
            _pred: &<Self as StorageBackend>::Storage<K>,
            _target: &<Self as StorageBackend>::Storage<KInt>,
            _r: Reduction,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
    }

    /// All quantization ops are no-ops returning an empty shape --- there is
    /// no real data to (de)quantize.
    impl<D: Device + Clone + 'static> QuantizedOps<Self> for DummyBackend<D> {
        /// Always an empty shape.
        fn quantize<K: FloatDType, Q: QuantDType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<Q>> {
            Ok(alloc::vec![])
        }
        /// Always an empty shape.
        fn dequantize<Q: QuantDType, K: FloatDType>(
            _t: &<Self as StorageBackend>::Storage<Q>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Always an empty shape.
        fn quantized_matmul<Q: QuantDType>(
            _lhs: &<Self as StorageBackend>::Storage<Q>,
            _rhs: &<Self as StorageBackend>::Storage<Q>,
        ) -> Result<<Self as StorageBackend>::Storage<f32>> {
            Ok(alloc::vec![])
        }
    }
    impl<D: Device + Clone + 'static> OptimizerOps<Self> for DummyBackend<D> {}
}
