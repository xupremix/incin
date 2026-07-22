use crate::prelude::{DTypeId, DeviceId, Result};
use crate::tensor::device::Device;
use crate::tensor::dtype::{DType, FloatDType, QuantDType};

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
    pub fn to_f64(&self) -> f64 {
        match self {
            ScalarValue::Float(f) => *f,
            ScalarValue::Int(i) => *i as f64,
        }
    }

    /// Reads the value as `i64`, truncating from `Float` if needed.
    pub fn to_i64(&self) -> i64 {
        match self {
            ScalarValue::Float(f) => *f as i64,
            ScalarValue::Int(i) => *i,
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

/// Resolves the dtype represented by `K` for a concrete runtime device.
pub trait SupportsDType<K: DType> {
    /// Resolve and validate dtype metadata before storage is created.
    fn resolve_dtype(field: &K::Field, _device: &DeviceId) -> Result<DTypeId> {
        Ok(K::to_kindle(field))
    }
}

/// The framework's single extension point: implement this (plus its
/// operation sub-traits) to add a new compute backend.
///
/// `Tensor<S, B, K, G>` is generic over `B: Backend` and stores exactly one
/// `B::Storage<K>` handle — every tensor operation ultimately dispatches to
/// a method on this trait or one of the op sub-traits it requires
/// (`NumericOps`, `FloatOps`, `CreationOps`, `ReductionOps`, `ModuleOps`,
/// `LossOps`, `QuantizedOps`, `OptimizerOps`, `TensorOps`). A method with no
/// override on a sub-trait returns `Err(UnsupportedBackendOperation)` by
/// default, so a backend only needs to implement the operations it actually
/// supports.
pub trait Backend:
    Sized
    + Clone
    + Send
    + Sync
    + 'static
    + TensorOps<Self>
    + NumericOps<Self>
    + FloatOps<Self>
    + CreationOps<Self>
    + ReductionOps<Self>
    + QuantizedOps<Self>
    + OptimizerOps<Self>
    + crate::tensor::backend::ModuleOps<Self>
    + crate::tensor::backend::LossOps<Self>
{
    /// The type-level device this backend runs on (`Cpu`, `Wgpu<N>`, `Cuda<N>`, `Dyn`, ...).
    type Device: Device;
    /// The floating-point element type this backend instance computes in.
    type FloatElem: DType;
    /// The integer element type this backend instance uses for indices
    /// (embedding lookups, `argmax`/`argmin`, etc.).
    type IntElem: DType;

    /// The concrete, backend-native handle a `Tensor<_, Self, K>` wraps.
    /// Every op takes and returns this type directly — there is no shared
    /// storage representation across backends.
    type Storage<K: DType>: Clone;
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

    /// Returns the logical shape (dimension sizes) of a storage handle.
    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize>;
    /// Returns the physical storage dtype when the backend can inspect it.
    fn storage_dtype<K: DType>(_t: &Self::Storage<K>) -> Option<DTypeId> {
        None
    }
    /// Returns the physical storage device when the backend can inspect it.
    fn storage_device<K: DType>(_t: &Self::Storage<K>) -> Option<DeviceId> {
        None
    }
    /// Renders a tensor's values for `Display` (concise, human-facing).
    fn format_tensor_display<K: DType>(t: &Self::Storage<K>) -> alloc::string::String;
    /// Renders a tensor's values and metadata for `Debug` (verbose,
    /// diagnostic-facing — shape/dtype/device alongside the data).
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String;

    /// Runs backpropagation from `t` through the backend's recorded tape,
    /// returning the resulting per-tensor gradients.
    fn backward<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads>;
    /// Same as `backward`, but additionally checks intermediate gradients
    /// for `NaN`/`Inf` and reports them as an error instead of silently
    /// propagating corrupted values.
    fn backward_with_nan_check<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads>;
    /// Looks up the gradient computed for `t` in a `Grads` collection
    /// returned by `backward`. `None` if `t` received no gradient (e.g. it
    /// wasn't reachable from the tensor `backward` was called on).
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>>;

    /// Serializes storage to a flat, dtype-native byte buffer (row-major,
    /// no header) — the inverse of `from_bytes`.
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>>;
    /// Reconstructs storage from raw bytes produced by `to_bytes`,
    /// validating that `bytes.len()` matches `shape`/`dtype`'s expected size.
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>>;

    /// Views a trainable variable as a plain tensor storage handle.
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>>;
    /// Promotes a plain tensor storage handle into a trainable variable.
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar>;
    /// Overwrites a variable's value in place (e.g. an optimizer step),
    /// without changing its identity for gradient-tracking purposes.
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()>;
}

/// Transfers storage and variables from one backend to a backend on `NewD`.
///
/// Implementations must not assume that storage handles or raw variable types
/// are compatible across backend families.
pub trait TransferTo<NewD: Device>: Backend {
    /// Backend selected for the destination device.
    type Output: Backend<Device = NewD, FloatElem = Self::FloatElem, IntElem = Self::IntElem>;

    /// Transfers tensor storage while preserving shape and dtype.
    fn transfer_storage<K: DType>(
        storage: &Self::Storage<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as Backend>::Storage<K>>
    where
        Self::Output: SupportsDType<K>;

    /// Transfers a variable into destination-native variable storage.
    fn transfer_var(
        variable: &Self::RawVar,
        dtype: &<Self::FloatElem as DType>::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as Backend>::RawVar>
    where
        Self::Output: SupportsDType<Self::FloatElem>;
}

// FloatOps only requires Backend, operates on FloatTensorPrimitive
/// Elementwise floating-point operations: activation functions and
/// scalar-broadcast arithmetic. Every method defaults to
/// `Err(UnsupportedBackendOperation)`, so a backend only overrides what it
/// actually implements.
pub trait FloatOps<B: Backend> {
    /// Rectified linear unit: `max(0, x)`.
    fn relu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "relu",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Heaviside step function: `1` where `x > 0`, else `0`.
    fn step<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "step",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Mish activation: `x * tanh(softplus(x))`.
    fn mish<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mish",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Exponential Linear Unit: `x` where `x > 0`, else `exp(x) - 1`.
    fn elu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "elu",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Gaussian Error Linear Unit (exact, erf-based):
    /// `x * 0.5 * (1 + erf(x / sqrt(2)))`.
    fn gelu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "gelu",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Elementwise absolute value.
    fn abs<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "abs",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Elementwise natural exponential.
    fn exp<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "exp",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Elementwise negation: `-x`.
    fn neg<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "neg",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Elementwise square root.
    fn sqrt<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sqrt",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Elementwise natural logarithm.
    fn log<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "log",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Elementwise hyperbolic tangent.
    fn tanh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "tanh",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Elementwise logistic sigmoid: `1 / (1 + exp(-x))`.
    fn sigmoid<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sigmoid",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Swish/SiLU activation: `x * sigmoid(x)`.
    fn swish<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "swish",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Softmax along `dim`, numerically stabilized by subtracting the
    /// per-slice max before exponentiating.
    fn softmax<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "softmax",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Adds scalar `scalar` to every element.
    fn add_scalar_float<K: DType>(_t: &B::Storage<K>, _scalar: f64) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "add_scalar_float",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Multiplies every element by scalar `scalar`.
    fn mul_scalar_float<K: DType>(_t: &B::Storage<K>, _scalar: f64) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mul_scalar_float",
            backend: core::any::type_name::<Self>(),
        })
    }
}

// NumericOps operates generically over any TensorKind!
/// Elementwise binary arithmetic with NumPy-style broadcasting (any
/// mismatched dimension must be size 1 on one side).
pub trait NumericOps<B: Backend> {
    /// Elementwise addition: `lhs + rhs`.
    fn add<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "add",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Elementwise subtraction: `lhs - rhs`.
    fn sub<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sub",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Elementwise multiplication: `lhs * rhs`.
    fn mul<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mul",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Elementwise division: `lhs / rhs`.
    fn div<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "div",
            backend: core::any::type_name::<Self>(),
        })
    }
}

/// Shape, layout, and dtype manipulation that doesn't change element
/// values (aside from `tensor_to_dtype`'s cast) — reshapes, views,
/// concatenation, and host-readback conversions.
pub trait TensorOps<B: Backend> {
    /// Reinterprets storage under a new `shape` with the same element count
    /// and row-major ordering (no data movement on backends with
    /// contiguous storage).
    fn reshape<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "reshape",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Swaps dimensions `dim1` and `dim2` in the logical shape (a view, not
    /// a copy, on backends with strided storage).
    fn transpose<K: DType>(
        _t: &B::Storage<K>,
        _dim1: usize,
        _dim2: usize,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "transpose",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Batched matrix multiplication over the trailing two dimensions of
    /// `lhs`/`rhs`, broadcasting any leading batch dimensions.
    fn matmul<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "matmul",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Broadcasts `t` to `shape` per NumPy rules (each dimension where the
    /// source size differs from the target must be exactly 1).
    fn broadcast_as<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "broadcast_as",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Takes the `len`-element window `[start, start + len)` along `dim`,
    /// keeping every other dimension unchanged.
    fn narrow<K: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _start: usize,
        _len: usize,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "narrow",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Removes dimension `dim`, which must have size 1.
    fn squeeze<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "squeeze",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Stacks same-shaped tensors along a brand-new dimension inserted at
    /// `dim` (output has one more dimension than each input).
    fn stack<K: DType>(_t: &[&B::Storage<K>], _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "stack",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Concatenates tensors along an existing dimension `dim` (every other
    /// dimension must already match across inputs).
    fn concat<K: DType>(_t: &[&B::Storage<K>], _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "concat",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Takes a `[start, end)` window per dimension, one `(start, end)` pair
    /// in `ranges` for each dimension of `t`, in order.
    fn slice<K: DType>(_t: &B::Storage<K>, _ranges: &[(usize, usize)]) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "slice",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Collapses dimensions `[start_dim, end_dim]` (inclusive) into a
    /// single dimension, preserving element order.
    fn flatten<K: DType>(
        _t: &B::Storage<K>,
        _start_dim: usize,
        _end_dim: usize,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "flatten",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Prepends size-1 dimensions on the left until `t` has as many
    /// dimensions as `shape`, then broadcasts to `shape` (the NumPy
    /// "align on the right" convention for broadcasting mismatched ranks).
    fn broadcast_left<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "broadcast_left",
            backend: core::any::type_name::<Self>(),
        })
    }

    /// Reads a single-element floating-point tensor's value as `f64`.
    /// Errors if `t` has more than one element.
    fn float_to_scalar<K: DType>(_t: &B::Storage<K>) -> Result<f64> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "float_to_scalar",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Reads a 1-D floating-point tensor's values into a host `Vec<f64>`.
    fn float_to_vec1<K: DType>(_t: &B::Storage<K>) -> Result<alloc::vec::Vec<f64>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "float_to_vec1",
            backend: core::any::type_name::<Self>(),
        })
    }

    /// Reads a single-element integer tensor's value as `i64`. Errors if
    /// `t` has more than one element.
    fn int_to_scalar<K: DType>(_t: &B::Storage<K>) -> Result<i64> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "int_to_scalar",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Reads a 1-D integer tensor's values into a host `Vec<i64>`.
    fn int_to_vec1<K: DType>(_t: &B::Storage<K>) -> Result<alloc::vec::Vec<i64>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "int_to_vec1",
            backend: core::any::type_name::<Self>(),
        })
    }

    /// Casts storage from dtype `K` to dtype `K2`, converting element
    /// values (not a bit-reinterpret — see `dtype` for the target's
    /// `DTypeId`).
    fn tensor_to_dtype<K: DType, K2: DType>(
        _t: &B::Storage<K>,
        _dtype: DTypeId,
    ) -> Result<B::Storage<K2>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "tensor_to_dtype",
            backend: core::any::type_name::<Self>(),
        })
    }
}

/// Allocates fresh storage and trainable variables — the only place new
/// tensor data can originate from (every other op transforms existing
/// storage).
pub trait CreationOps<B: Backend> {
    /// Allocates a `shape`-sized tensor of `dtype`, filled with zero.
    fn zeros<K: DType>(
        _shape: &[usize],
        _dtype: DTypeId,
        _device: &DeviceId,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "zeros",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Allocates a `shape`-sized tensor of `dtype`, filled with one.
    fn ones<K: DType>(
        _shape: &[usize],
        _dtype: DTypeId,
        _device: &DeviceId,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "ones",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Allocates a `shape`-sized tensor of `dtype`, filled with samples
    /// from `Uniform(0, 1)`.
    fn rand<K: DType>(
        _shape: &[usize],
        _dtype: DTypeId,
        _device: &DeviceId,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "rand",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Allocates a `shape`-sized tensor of `dtype`, filled with samples
    /// from the standard normal distribution `N(0, 1)`.
    fn randn<K: DType>(
        _shape: &[usize],
        _dtype: DTypeId,
        _device: &DeviceId,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "randn",
            backend: core::any::type_name::<Self>(),
        })
    }

    /// Same as `zeros`, but returns a trainable `RawVar` directly.
    fn var_zeros<K: DType>(
        _shape: &[usize],
        _dtype: DTypeId,
        _device: &DeviceId,
    ) -> Result<B::RawVar> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "var_zeros",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Same as `ones`, but returns a trainable `RawVar` directly.
    fn var_ones<K: DType>(
        _shape: &[usize],
        _dtype: DTypeId,
        _device: &DeviceId,
    ) -> Result<B::RawVar> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "var_ones",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Same as `rand`, but returns a trainable `RawVar` directly.
    fn var_rand<K: DType>(
        _shape: &[usize],
        _dtype: DTypeId,
        _device: &DeviceId,
    ) -> Result<B::RawVar> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "var_rand",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Same as `randn`, but returns a trainable `RawVar` directly.
    fn var_randn<K: DType>(
        _shape: &[usize],
        _dtype: DTypeId,
        _device: &DeviceId,
    ) -> Result<B::RawVar> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "var_randn",
            backend: core::any::type_name::<Self>(),
        })
    }
}

/// Reductions that collapse a tensor along one or all dimensions —
/// aggregate statistics (`sum`/`mean`/`max`/`min`) and index-producing
/// selections (`argmax`/`argmin`/`topk`/`argsort`).
pub trait ReductionOps<B: Backend> {
    /// Sums every element into a single-element tensor.
    fn sum_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sum_all",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Averages every element into a single-element tensor.
    fn mean_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mean_all",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Reduces to the single largest element.
    fn max_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "max_all",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Reduces to the single smallest element.
    fn min_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "min_all",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Sums along `dim`, removing that dimension from the output shape.
    fn sum_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sum_dim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Sums along `dim`, keeping it in the output shape as size 1.
    fn sum_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sum_keepdim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Averages along `dim`, removing that dimension from the output shape.
    fn mean_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mean_dim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Averages along `dim`, keeping it in the output shape as size 1.
    fn mean_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mean_keepdim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Reduces along `dim` to its max, removing that dimension.
    fn max_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "max_dim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Reduces along `dim` to its max, keeping it in the output shape as
    /// size 1.
    fn max_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "max_keepdim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Reduces along `dim` to its min, removing that dimension.
    fn min_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "min_dim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Reduces along `dim` to its min, keeping it in the output shape as
    /// size 1.
    fn min_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "min_keepdim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Index of the maximum element, either flattened (`dim: None`) or
    /// along a single `dim`.
    fn argmax<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: Option<usize>,
    ) -> Result<B::Storage<KInt>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "argmax",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Index of the minimum element, either flattened (`dim: None`) or
    /// along a single `dim`.
    fn argmin<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: Option<usize>,
    ) -> Result<B::Storage<KInt>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "argmin",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// The `k` largest (`largest: true`) or smallest (`largest: false`)
    /// elements along `dim`, returned as `(values, indices)`.
    fn topk<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _k: usize,
        _dim: usize,
        _largest: bool,
    ) -> Result<(B::Storage<K>, B::Storage<KInt>)> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "topk",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Indices that would sort `t` along `dim`, ascending or `descending`.
    fn argsort<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _descending: bool,
    ) -> Result<B::Storage<KInt>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "argsort",
            backend: core::any::type_name::<Self>(),
        })
    }
}

/// Neural-network layer primitives: normalization, embedding lookup,
/// convolution, and pooling. Each takes plain storage (not `Param`/`Module`
/// wrappers) — the `nn` layer types call through to these.
pub trait ModuleOps<B: Backend> {
    /// Layer normalization over the last dimension: normalizes `t` to zero
    /// mean/unit variance (with `eps` added for numerical stability), then
    /// applies an affine `weight` scale and optional `bias` shift.
    fn layer_norm<K: DType>(
        _t: &B::Storage<K>,
        _weight: &B::Storage<K>,
        _bias: Option<&B::Storage<K>>,
        _eps: f32,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "layer_norm",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Batch normalization over the channel dimension: normalizes using
    /// batch statistics (training) or `rm`/`rv` running mean/variance
    /// (inference), with `momentum` controlling running-stat updates and
    /// optional affine `w`/`b`.
    fn batch_norm<K: DType>(
        _t: &B::Storage<K>,
        _w: Option<&B::Storage<K>>,
        _b: Option<&B::Storage<K>>,
        _rm: Option<&B::Storage<K>>,
        _rv: Option<&B::Storage<K>>,
        _e: f32,
        _momentum: f64,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "batch_norm",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Embedding table lookup: gathers rows of the weight matrix `w` at
    /// the integer indices in `t`.
    fn embedding<K: DType, KInt: DType>(
        _t: &B::Storage<KInt>,
        _w: &B::Storage<K>,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "embedding",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// 1-D convolution of `t` with kernel `w` (and optional bias `b`),
    /// with the given `stride`/`padding`/`dilation`/`groups`.
    fn conv1d<K: DType>(
        _t: &B::Storage<K>,
        _w: &B::Storage<K>,
        _b: Option<&B::Storage<K>>,
        _stride: usize,
        _padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "conv1d",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// 2-D convolution of `t` with kernel `w` (and optional bias `b`),
    /// with the given `stride`/`padding`/`dilation`/`groups`.
    fn conv2d<K: DType>(
        _t: &B::Storage<K>,
        _w: &B::Storage<K>,
        _b: Option<&B::Storage<K>>,
        _stride: usize,
        _padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "conv2d",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Transposed ("deconvolution") 2-D convolution — the gradient
    /// operation of `conv2d` used as a forward op for upsampling, with an
    /// extra `output_padding` to resolve the otherwise-ambiguous output size.
    fn conv_transpose2d<K: DType>(
        _t: &B::Storage<K>,
        _w: &B::Storage<K>,
        _b: Option<&B::Storage<K>>,
        _stride: usize,
        _padding: usize,
        _output_padding: usize,
        _dilation: usize,
        _groups: usize,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "conv_transpose2d",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// 2-D max pooling: for each output position, the max over its
    /// `kernel_size` window (given `stride`/`padding`/`dilation`).
    fn max_pool2d<K: DType>(
        _t: &B::Storage<K>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "max_pool2d",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// 2-D average pooling: for each output position, the mean over its
    /// `kernel_size` window (given `stride`/`padding`).
    fn avg_pool2d<K: DType>(
        _t: &B::Storage<K>,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "avg_pool2d",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Average pooling that derives its own window size per output
    /// position so the output spatial size is exactly `output_size`,
    /// regardless of the input size (PyTorch's `AdaptiveAvgPool2d`).
    fn adaptive_avg_pool2d<K: DType>(
        _t: &B::Storage<K>,
        _output_size: (usize, usize),
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "adaptive_avg_pool2d",
            backend: core::any::type_name::<Self>(),
        })
    }
}

/// Loss functions, given a default implementation in terms of
/// `NumericOps`/`FloatOps`/`ReductionOps` so a backend implementing those
/// gets working losses for free (override individually only for a
/// backend-specific fused kernel).
pub trait LossOps<B: Backend>: NumericOps<B> + FloatOps<B> + ReductionOps<B> {
    /// Mean/sum/none-reduced squared error: `(pred - target)^2`.
    fn mse_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<B::Storage<K>> {
        let diff = <B as NumericOps<B>>::sub::<K>(pred, target)?;
        let sq = <B as NumericOps<B>>::mul::<K>(&diff, &diff)?;
        match reduction {
            crate::nn::loss::Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&sq),
            crate::nn::loss::Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&sq),
            crate::nn::loss::Reduction::None => Ok(sq),
        }
    }

    /// Mean/sum/none-reduced absolute error: `|pred - target|`.
    fn l1_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<B::Storage<K>> {
        let diff = <B as NumericOps<B>>::sub::<K>(pred, target)?;
        let abs_diff = <B as FloatOps<B>>::abs::<K>(&diff)?;
        match reduction {
            crate::nn::loss::Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&abs_diff),
            crate::nn::loss::Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&abs_diff),
            crate::nn::loss::Reduction::None => Ok(abs_diff),
        }
    }

    /// Binary cross-entropy computed from raw logits (`pred`, pre-sigmoid),
    /// using the numerically stable formulation
    /// `max(x,0) - x*z + log(1 + exp(-|x|))` so it never evaluates
    /// `sigmoid`/`log` on the raw logit directly.
    fn bce_with_logits_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: crate::nn::loss::Reduction,
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
            crate::nn::loss::Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&loss_elem),
            crate::nn::loss::Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&loss_elem),
            crate::nn::loss::Reduction::None => Ok(loss_elem),
        }
    }

    /// Cross-entropy loss between raw `pred` logits (softmax applied
    /// internally) and integer class-index `target`s — no default
    /// implementation, since it needs a numerically-stable fused
    /// log-softmax rather than composing `softmax` + `log` naively.
    fn cross_entropy_loss<K: DType, KInt: DType>(
        _pred: &B::Storage<K>,
        _target: &B::Storage<KInt>,
        _reduction: crate::nn::loss::Reduction,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "cross_entropy_loss",
            backend: core::any::type_name::<Self>(),
        })
    }
}

/// Block quantization: compresses `FloatDType` storage into a `QuantDType`
/// representation for reduced memory footprint, and the reverse.
pub trait QuantizedOps<B: Backend> {
    /// Compresses `t` from a float dtype into quantized storage `Q`.
    fn quantize<K: FloatDType, Q: QuantDType>(_t: &B::Storage<K>) -> Result<B::Storage<Q>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "quantize",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Expands quantized storage `Q` back into a float dtype `K`
    /// (lossy — the inverse of `quantize` only up to quantization error).
    fn dequantize<Q: QuantDType, K: FloatDType>(_t: &B::Storage<Q>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "dequantize",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Matrix multiplication of two quantized-storage operands, producing
    /// `f32` output without needing to fully dequantize both operands first.
    fn quantized_matmul<Q: QuantDType>(
        _lhs: &B::Storage<Q>,
        _rhs: &B::Storage<Q>,
    ) -> Result<B::Storage<f32>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "quantized_matmul",
            backend: core::any::type_name::<Self>(),
        })
    }
}

/// In-place optimizer update rules, applied directly to a backend's
/// `RawVar` (so the update can be fused/backend-native instead of composed
/// from generic tensor ops where that matters for performance).
pub trait OptimizerOps<B: Backend> {
    /// One AdamW step (Adam with decoupled weight decay): updates `var`
    /// in place from `grad`, with `m`/`v` as the first/second moment
    /// running averages (bias-corrected internally using `step`, the
    /// 1-indexed step count). Has a default implementation composed from
    /// `NumericOps`/`FloatOps`, so any backend implementing those gets a
    /// working (if not backend-fused) AdamW for free.
    fn adamw_step<K: DType>(
        var: &mut B::RawVar,
        grad: &B::Storage<K>,
        m: &mut B::Storage<K>,
        v: &mut B::Storage<K>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        step: usize,
    ) -> Result<()> {
        let mut t = B::var_as_tensor::<K>(var)?;
        let t_step = step as f64;
        let bias_correction1 = 1.0 - beta1.powf(t_step);
        let bias_correction2 = 1.0 - beta2.powf(t_step);

        if weight_decay > 0.0 {
            let decay = B::mul_scalar_float::<K>(&t, weight_decay * lr)?;
            t = B::sub::<K>(&t, &decay)?;
        }

        let term1_m = B::mul_scalar_float::<K>(m, beta1)?;
        let term2_m = B::mul_scalar_float::<K>(grad, 1.0 - beta1)?;
        let m_t = B::add::<K>(&term1_m, &term2_m)?;

        let grad_sq = B::mul::<K>(grad, grad)?;
        let term1_v = B::mul_scalar_float::<K>(v, beta2)?;
        let term2_v = B::mul_scalar_float::<K>(&grad_sq, 1.0 - beta2)?;
        let v_t = B::add::<K>(&term1_v, &term2_v)?;

        *m = m_t.clone();
        *v = v_t.clone();

        let m_hat = B::mul_scalar_float::<K>(&m_t, 1.0 / bias_correction1)?;
        let v_hat = B::mul_scalar_float::<K>(&v_t, 1.0 / bias_correction2)?;

        let denom = B::add_scalar_float::<K>(&B::sqrt::<K>(&v_hat)?, eps)?;
        let step_val = B::mul_scalar_float::<K>(&B::div::<K>(&m_hat, &denom)?, lr)?;

        let updated = B::sub::<K>(&t, &step_val)?;
        B::assign_var::<K>(var, &updated)?;
        Ok(())
    }
}
/// A minimal, allocation-free `Backend` implementation used only by unit
/// tests elsewhere in this crate that need a concrete `Backend` type
/// without depending on `kindle-backends`. See `DummyBackend`.
pub mod dummy {
    use super::*;
    use crate::nn::Reduction;
    use crate::prelude::Result;
    use crate::tensor::device::Device;
    use crate::tensor::device::DeviceId;
    use crate::tensor::dtype::DType;

    /// Test-only stand-in `Backend` used by `tensor/base.rs`'s unit tests to
    /// exercise `Tensor`'s generic-over-`Backend` machinery without pulling
    /// in a real compute backend. Its `Storage<K>` is literally the shape
    /// (`Vec<usize>`) — every op below tracks how an operation would
    /// transform the *shape*, using the same arithmetic real backends use,
    /// but performs no actual data computation and holds no element values.
    pub struct DummyBackend<T, D> {
        _marker: core::marker::PhantomData<(T, D)>,
    }

    impl<T: DType, D: Device + Clone + 'static> Clone for DummyBackend<T, D> {
        /// Cheap: the type carries no state beyond its `PhantomData` markers.
        fn clone(&self) -> Self {
            DummyBackend {
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<T: DType, D: Device + Clone + 'static> Backend for DummyBackend<T, D> {
        /// The device type this stand-in is parameterized over.
        type Device = D;
        /// The float element type this stand-in is parameterized over.
        type FloatElem = T;
        /// Fixed to `i64`, matching every real backend's `IntElem`.
        type IntElem = i64;
        /// A trainable variable is just its shape, like `Storage`.
        type RawVar = alloc::vec::Vec<usize>;
        /// No real gradients are tracked, so this carries no data.
        type Grads = ();
        /// Shape-only storage: `Storage<K>` is the tensor's shape, not its
        /// values, regardless of `K`.
        type Storage<K: DType> = alloc::vec::Vec<usize>;
        /// No dispatch wrapper — this stand-in is always its own inner backend.
        type InnerBackend = Self;

        /// Returns the shape, which is exactly what `Storage<K>` already is.
        fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
            t.clone()
        }
        /// Always `"dummy"` — there are no real values to render.
        fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }
        /// Always `"dummy"` — there are no real values to render.
        fn format_tensor_debug<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }
        /// No-op: there is no tape to run backward through.
        fn backward<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
            Ok(())
        }
        /// No-op: there is no tape to run backward through.
        fn backward_with_nan_check<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
            Ok(())
        }
        /// Always `None`: `Grads` carries no data to look a gradient up in.
        fn get_grad<K: DType>(
            _t: &Self::Storage<K>,
            _grads: &Self::Grads,
        ) -> Result<Option<Self::Storage<K>>> {
            Ok(None)
        }
        /// Always empty: there are no element values to serialize.
        fn to_bytes<K: DType>(_t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
            Ok(alloc::vec::Vec::new())
        }
        /// Ignores `bytes` entirely and reconstructs storage from `shape`
        /// alone, since `Storage<K>` only ever tracks shape.
        fn from_bytes<K: DType>(
            _bytes: &[u8],
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<Self::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// `RawVar` and `Storage<K>` are the same representation, so this
        /// is a plain clone.
        fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
            Ok(var.clone())
        }
        /// `RawVar` and `Storage<K>` are the same representation, so this
        /// is a plain clone.
        fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
            Ok(t.clone())
        }
        /// Overwrites `var`'s shape with `tensor`'s.
        fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
            *var = tensor.clone();
            Ok(())
        }
    }

    impl<T: DType, D: Device + Clone + 'static, K: DType> SupportsDType<K> for DummyBackend<T, D> {}

    /// Output spatial size for conv/pool shape math:
    /// `(in + 2*pad - dilation*(kernel-1) - 1) / stride + 1`. Uses saturating
    /// arithmetic throughout (never panics/wraps on pathological inputs —
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

    impl<T: DType, D: Device + Clone + 'static, NewD: Device + Clone + 'static> TransferTo<NewD>
        for DummyBackend<T, D>
    {
        type Output = DummyBackend<T, NewD>;

        fn transfer_storage<K: DType>(
            storage: &Self::Storage<K>,
            _dtype: &K::Field,
            _device: &NewD::Field,
        ) -> Result<<Self::Output as Backend>::Storage<K>>
        where
            Self::Output: SupportsDType<K>,
        {
            Ok(storage.clone())
        }

        fn transfer_var(
            variable: &Self::RawVar,
            _dtype: &<Self::FloatElem as DType>::Field,
            _device: &NewD::Field,
        ) -> Result<<Self::Output as Backend>::RawVar>
        where
            Self::Output: SupportsDType<Self::FloatElem>,
        {
            Ok(variable.clone())
        }
    }

    /// Shape is preserved by every allocation, since it's the only thing
    /// `Storage`/`RawVar` track — no real fill value is ever written.
    impl<T: DType, D: Device + Clone + 'static> CreationOps<Self> for DummyBackend<T, D> {
        /// Returns `shape` verbatim as the storage handle.
        fn zeros<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn ones<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn rand<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn randn<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_zeros<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_ones<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_rand<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_randn<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
    }

    /// Every binary op is a shape no-op: it returns `lhs`'s shape unchanged
    /// (broadcasting between differing shapes is not modeled).
    impl<T: DType, D: Device + Clone + 'static> NumericOps<Self> for DummyBackend<T, D> {
        /// Returns `lhs`'s shape unchanged.
        fn add<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        /// Returns `lhs`'s shape unchanged.
        fn sub<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        /// Returns `lhs`'s shape unchanged.
        fn mul<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        /// Returns `lhs`'s shape unchanged.
        fn div<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
    }

    /// Every activation and scalar op is shape-preserving, so each is a
    /// plain clone of the input shape.
    impl<T: DType, D: Device + Clone + 'static> FloatOps<Self> for DummyBackend<T, D> {
        /// Returns `t`'s shape unchanged.
        fn add_scalar_float<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn mul_scalar_float<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn relu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn step<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn mish<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn elu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn gelu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn abs<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn exp<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn neg<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn sqrt<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn log<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn tanh<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn sigmoid<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn swish<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged (`dim` is not validated).
        fn softmax<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
    }

    /// `_all` reductions collapse to an (empty) scalar shape; `_dim`
    /// reductions either remove `dim` or clamp it to size 1 (`_keepdim`),
    /// exactly like a real reduction's shape effect — again with no real
    /// values behind either result.
    impl<T: DType, D: Device + Clone + 'static> ReductionOps<Self> for DummyBackend<T, D> {
        /// Collapses to an empty (scalar) shape.
        fn sum_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn mean_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn max_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn min_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Removes `dim` from the shape.
        fn sum_dim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn sum_keepdim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Removes `dim` from the shape.
        fn mean_dim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn mean_keepdim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Removes `dim` from the shape.
        fn max_dim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn max_keepdim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Removes `dim` from the shape.
        fn min_dim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn min_keepdim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Always an empty shape — no indices are actually computed.
        fn argmax<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        /// Always an empty shape — no indices are actually computed.
        fn argmin<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        /// Always an empty `(values, indices)` pair.
        fn topk<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _k: usize,
            _dim: usize,
            _largest: bool,
        ) -> Result<(
            <Self as Backend>::Storage<K>,
            <Self as Backend>::Storage<KInt>,
        )> {
            Ok((alloc::vec![], alloc::vec![]))
        }
        /// Always an empty shape — no indices are actually computed.
        fn argsort<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: usize,
            _descending: bool,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
    }

    /// Each op tracks its real shape-transformation logic (matmul's last
    /// dim, transpose's swap, flatten's dimension collapse, etc.) since
    /// shape *is* everything this stand-in's storage represents — but
    /// still no element values exist behind any of it.
    impl<T: DType, D: Device + Clone + 'static> TensorOps<Self> for DummyBackend<T, D> {
        /// Replaces the trailing dimension with `rhs`'s trailing dimension,
        /// mirroring real matmul's output shape.
        fn matmul<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = lhs.clone();
            if out.len() >= 2 && rhs.len() >= 2 {
                let len = out.len();
                out[len - 1] = rhs[rhs.len() - 1];
            }
            Ok(out)
        }
        /// Always `0.0` — there is no real element value to read.
        fn float_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<f64> {
            Ok(0.0)
        }
        /// Always a single `0.0` — there are no real element values to read.
        fn float_to_vec1<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<f64>> {
            Ok(alloc::vec![0.0])
        }
        /// Always `0` — there is no real element value to read.
        fn int_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<i64> {
            Ok(0)
        }
        /// Always a single `0` — there are no real element values to read.
        fn int_to_vec1<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<i64>> {
            Ok(alloc::vec![0])
        }

        /// Returns the target `shape` verbatim (broadcast compatibility is
        /// not validated).
        fn broadcast_as<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(s.to_vec())
        }
        /// Prepends the target `shape`'s dimensions ahead of `t`'s own.
        fn broadcast_left<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = s.to_vec();
            out.extend_from_slice(t);
            Ok(out)
        }
        /// Returns the target `shape` verbatim (numel compatibility is not
        /// validated).
        fn reshape<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(s.to_vec())
        }
        /// Swaps dimensions `d1`/`d2` in the shape.
        fn transpose<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            d1: usize,
            d2: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            if d1 < out.len() && d2 < out.len() {
                out.swap(d1, d2);
            }
            Ok(out)
        }
        /// Collapses dimensions `[s, e]` into their product.
        fn flatten<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            s: usize,
            e: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            if s > e || e >= t.len() {
                return Ok(t.clone());
            }
            let mut out = t[..s].to_vec();
            out.push(t[s..=e].iter().product());
            out.extend_from_slice(&t[e + 1..]);
            Ok(out)
        }
        /// Sets each dimension's size to its `(start, end)` window length.
        fn slice<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            ranges: &[(usize, usize)],
        ) -> Result<<Self as Backend>::Storage<K>> {
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
            t: &<Self as Backend>::Storage<K>,
            d: usize,
            _s: usize,
            l: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            if d < out.len() {
                out[d] = l;
            }
            Ok(out)
        }
        /// Removes dimension `d` from the shape.
        fn squeeze<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            d: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            if d < out.len() {
                out.remove(d);
            }
            Ok(out)
        }
        /// Inserts a new dimension at `d`, sized to the number of stacked
        /// tensors.
        fn stack<K: DType>(
            t: &[&<Self as Backend>::Storage<K>],
            d: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
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
            t: &[&<Self as Backend>::Storage<K>],
            d: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            if t.is_empty() {
                return Ok(alloc::vec![]);
            }
            let mut out = t[0].clone();
            if d < out.len() {
                out[d] = t.iter().map(|s| s.get(d).copied().unwrap_or(0)).sum();
            }
            Ok(out)
        }
        /// Returns `t`'s shape unchanged — no element values exist to cast.
        fn tensor_to_dtype<K: DType, K2: DType>(
            t: &<Self as Backend>::Storage<K>,
            _dtype: DTypeId,
        ) -> Result<<Self as Backend>::Storage<K2>> {
            Ok(t.clone())
        }
    }

    /// Normalization ops are shape-preserving no-ops; conv/pool ops
    /// compute their real output spatial size via `conv_out_size`/
    /// `conv_transpose_out_size` (the saturating helpers above) so tests
    /// can assert on shape correctness even though no data is computed.
    impl<T: DType, D: Device + Clone + 'static> ModuleOps<Self> for DummyBackend<T, D> {
        /// Returns `t`'s shape unchanged.
        fn layer_norm<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            _e: f32,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn batch_norm<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _w: Option<&<Self as Backend>::Storage<K>>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            _rm: Option<&<Self as Backend>::Storage<K>>,
            _rv: Option<&<Self as Backend>::Storage<K>>,
            _e: f32,
            _m: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Always an empty shape — no real gather is performed.
        fn embedding<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<KInt>,
            _w: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Computes the real output shape: channel dim from `w[0]`, spatial
        /// dim via `conv_out_size`.
        fn conv1d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            s: usize,
            p: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
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
            t: &<Self as Backend>::Storage<K>,
            w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            s: usize,
            p: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
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
            t: &<Self as Backend>::Storage<K>,
            w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            s: usize,
            p: usize,
            op: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
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
            t: &<Self as Backend>::Storage<K>,
            k: (usize, usize),
            s: (usize, usize),
            p: (usize, usize),
            d: (usize, usize),
        ) -> Result<<Self as Backend>::Storage<K>> {
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
            t: &<Self as Backend>::Storage<K>,
            k: (usize, usize),
            s: (usize, usize),
            p: (usize, usize),
        ) -> Result<<Self as Backend>::Storage<K>> {
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
            t: &<Self as Backend>::Storage<K>,
            out: (usize, usize),
        ) -> Result<<Self as Backend>::Storage<K>> {
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
    impl<T: DType, D: Device + Clone + 'static> LossOps<Self> for DummyBackend<T, D> {
        /// Always an empty (scalar) shape.
        fn mse_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Always an empty (scalar) shape.
        fn l1_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Always an empty (scalar) shape.
        fn bce_with_logits_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Always an empty (scalar) shape.
        fn cross_entropy_loss<K: DType, KInt: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<KInt>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
    }

    /// All quantization ops are no-ops returning an empty shape — there is
    /// no real data to (de)quantize.
    impl<T: DType, D: Device + Clone + 'static> QuantizedOps<Self> for DummyBackend<T, D> {
        /// Always an empty shape.
        fn quantize<K: FloatDType, Q: QuantDType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<Q>> {
            Ok(alloc::vec![])
        }
        /// Always an empty shape.
        fn dequantize<Q: QuantDType, K: FloatDType>(
            _t: &<Self as Backend>::Storage<Q>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Always an empty shape.
        fn quantized_matmul<Q: QuantDType>(
            _lhs: &<Self as Backend>::Storage<Q>,
            _rhs: &<Self as Backend>::Storage<Q>,
        ) -> Result<<Self as Backend>::Storage<f32>> {
            Ok(alloc::vec![])
        }
    }
    impl<T: DType, D: Device + Clone + 'static> OptimizerOps<Self> for DummyBackend<T, D> {}
}
