use crate::prelude::{DTypeId, DeviceId, Result};
use crate::tensor::device::Device;
use crate::tensor::dtype::{DType, FloatDType, QuantDType};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Provides the ScalarValue operation or structure.
pub enum ScalarValue {
    /// Provides the Float operation or structure.
    Float(f64),
    /// Provides the Int operation or structure.
    Int(i64),
}

impl ScalarValue {
    /// Provides the to_f64 operation or structure.
    pub fn to_f64(&self) -> f64 {
        match self {
            ScalarValue::Float(f) => *f,
            ScalarValue::Int(i) => *i as f64,
        }
    }

    /// Provides the to_i64 operation or structure.
    pub fn to_i64(&self) -> i64 {
        match self {
            ScalarValue::Float(f) => *f as i64,
            ScalarValue::Int(i) => *i,
        }
    }
}

impl From<f32> for ScalarValue {
    /// Provides the from operation or structure.
    fn from(v: f32) -> Self {
        ScalarValue::Float(v as f64)
    }
}
impl From<f64> for ScalarValue {
    /// Provides the from operation or structure.
    fn from(v: f64) -> Self {
        ScalarValue::Float(v)
    }
}
impl From<i32> for ScalarValue {
    /// Provides the from operation or structure.
    fn from(v: i32) -> Self {
        ScalarValue::Int(v as i64)
    }
}
impl From<i64> for ScalarValue {
    /// Provides the from operation or structure.
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

/// Provides the Backend operation or structure.
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
    /// Provides the Device operation or structure.
    type Device: Device;
    /// Provides the FloatElem operation or structure.
    type FloatElem: DType;
    /// Provides the IntElem operation or structure.
    type IntElem: DType;

    /// Provides the Storage operation or structure.
    type Storage<K: DType>: Clone;
    /// Provides the RawVar operation or structure.
    type RawVar: Clone;
    /// Provides the Grads operation or structure.
    type Grads;

    /// Provides the InnerBackend operation or structure.
    type InnerBackend: Backend;

    /// Provides the shape operation or structure.
    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize>;
    /// Returns the physical storage dtype when the backend can inspect it.
    fn storage_dtype<K: DType>(_t: &Self::Storage<K>) -> Option<DTypeId> {
        None
    }
    /// Returns the physical storage device when the backend can inspect it.
    fn storage_device<K: DType>(_t: &Self::Storage<K>) -> Option<DeviceId> {
        None
    }
    /// Provides the format_tensor_display operation or structure.
    fn format_tensor_display<K: DType>(t: &Self::Storage<K>) -> alloc::string::String;
    /// Provides the format_tensor_debug operation or structure.
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String;

    /// Provides the backward operation or structure.
    fn backward<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads>;
    /// Provides the backward_with_nan_check operation or structure.
    fn backward_with_nan_check<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads>;
    /// Provides the get_grad operation or structure.
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>>;

    /// Provides the to_bytes operation or structure.
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>>;
    /// Provides the from_bytes operation or structure.
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>>;

    /// Provides the var_as_tensor operation or structure.
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>>;
    /// Provides the var_from_tensor operation or structure.
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar>;
    /// Provides the assign_var operation or structure.
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
/// Provides the FloatOps operation or structure.
pub trait FloatOps<B: Backend> {
    /// Provides the relu operation or structure.
    fn relu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "relu",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the step operation or structure.
    fn step<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "step",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the mish operation or structure.
    fn mish<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mish",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the elu operation or structure.
    fn elu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "elu",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the gelu operation or structure.
    fn gelu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "gelu",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the abs operation or structure.
    fn abs<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "abs",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the exp operation or structure.
    fn exp<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "exp",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the neg operation or structure.
    fn neg<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "neg",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the sqrt operation or structure.
    fn sqrt<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sqrt",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the log operation or structure.
    fn log<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "log",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the tanh operation or structure.
    fn tanh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "tanh",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the sigmoid operation or structure.
    fn sigmoid<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sigmoid",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the swish operation or structure.
    fn swish<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "swish",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the softmax operation or structure.
    fn softmax<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "softmax",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the add_scalar_float operation or structure.
    fn add_scalar_float<K: DType>(_t: &B::Storage<K>, _scalar: f64) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "add_scalar_float",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the mul_scalar_float operation or structure.
    fn mul_scalar_float<K: DType>(_t: &B::Storage<K>, _scalar: f64) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mul_scalar_float",
            backend: core::any::type_name::<Self>(),
        })
    }
}

// NumericOps operates generically over any TensorKind!
/// Provides the NumericOps operation or structure.
pub trait NumericOps<B: Backend> {
    /// Provides the add operation or structure.
    fn add<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "add",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the sub operation or structure.
    fn sub<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sub",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the mul operation or structure.
    fn mul<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mul",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the div operation or structure.
    fn div<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "div",
            backend: core::any::type_name::<Self>(),
        })
    }
}

/// Provides the TensorOps operation or structure.
pub trait TensorOps<B: Backend> {
    /// Provides the reshape operation or structure.
    fn reshape<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "reshape",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the transpose operation or structure.
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
    /// Provides the matmul operation or structure.
    fn matmul<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "matmul",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the broadcast_as operation or structure.
    fn broadcast_as<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "broadcast_as",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the narrow operation or structure.
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
    /// Provides the squeeze operation or structure.
    fn squeeze<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "squeeze",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the stack operation or structure.
    fn stack<K: DType>(_t: &[&B::Storage<K>], _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "stack",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the concat operation or structure.
    fn concat<K: DType>(_t: &[&B::Storage<K>], _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "concat",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the slice operation or structure.
    fn slice<K: DType>(_t: &B::Storage<K>, _ranges: &[(usize, usize)]) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "slice",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the flatten operation or structure.
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
    /// Provides the broadcast_left operation or structure.
    fn broadcast_left<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "broadcast_left",
            backend: core::any::type_name::<Self>(),
        })
    }

    /// Provides the float_to_scalar operation or structure.
    fn float_to_scalar<K: DType>(_t: &B::Storage<K>) -> Result<f64> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "float_to_scalar",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the float_to_vec1 operation or structure.
    fn float_to_vec1<K: DType>(_t: &B::Storage<K>) -> Result<alloc::vec::Vec<f64>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "float_to_vec1",
            backend: core::any::type_name::<Self>(),
        })
    }

    /// Provides the int_to_scalar operation or structure.
    fn int_to_scalar<K: DType>(_t: &B::Storage<K>) -> Result<i64> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "int_to_scalar",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the int_to_vec1 operation or structure.
    fn int_to_vec1<K: DType>(_t: &B::Storage<K>) -> Result<alloc::vec::Vec<i64>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "int_to_vec1",
            backend: core::any::type_name::<Self>(),
        })
    }

    /// Provides the tensor_to_dtype operation or structure.
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

/// Provides the CreationOps operation or structure.
pub trait CreationOps<B: Backend> {
    /// Provides the zeros operation or structure.
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
    /// Provides the ones operation or structure.
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
    /// Provides the rand operation or structure.
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
    /// Provides the randn operation or structure.
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

    /// Provides the var_zeros operation or structure.
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
    /// Provides the var_ones operation or structure.
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
    /// Provides the var_rand operation or structure.
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
    /// Provides the var_randn operation or structure.
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

/// Provides the ReductionOps operation or structure.
pub trait ReductionOps<B: Backend> {
    /// Provides the sum_all operation or structure.
    fn sum_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sum_all",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the mean_all operation or structure.
    fn mean_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mean_all",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the max_all operation or structure.
    fn max_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "max_all",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the min_all operation or structure.
    fn min_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "min_all",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the sum_dim operation or structure.
    fn sum_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sum_dim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the sum_keepdim operation or structure.
    fn sum_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "sum_keepdim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the mean_dim operation or structure.
    fn mean_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mean_dim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the mean_keepdim operation or structure.
    fn mean_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "mean_keepdim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the max_dim operation or structure.
    fn max_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "max_dim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the max_keepdim operation or structure.
    fn max_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "max_keepdim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the min_dim operation or structure.
    fn min_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "min_dim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the min_keepdim operation or structure.
    fn min_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "min_keepdim",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the argmax operation or structure.
    fn argmax<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: Option<usize>,
    ) -> Result<B::Storage<KInt>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "argmax",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the argmin operation or structure.
    fn argmin<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: Option<usize>,
    ) -> Result<B::Storage<KInt>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "argmin",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the topk operation or structure.
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
    /// Provides the argsort operation or structure.
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

/// Provides the ModuleOps operation or structure.
pub trait ModuleOps<B: Backend> {
    /// Provides the layer_norm operation or structure.
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
    /// Provides the batch_norm operation or structure.
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
    /// Provides the embedding operation or structure.
    fn embedding<K: DType, KInt: DType>(
        _t: &B::Storage<KInt>,
        _w: &B::Storage<K>,
    ) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "embedding",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the conv1d operation or structure.
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
    /// Provides the conv2d operation or structure.
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
    /// Provides the conv_transpose2d operation or structure.
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
    /// Provides the max_pool2d operation or structure.
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
    /// Provides the avg_pool2d operation or structure.
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
    /// Provides the adaptive_avg_pool2d operation or structure.
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

/// Provides the LossOps operation or structure.
pub trait LossOps<B: Backend>: NumericOps<B> + FloatOps<B> + ReductionOps<B> {
    /// Provides the mse_loss operation or structure.
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

    /// Provides the l1_loss operation or structure.
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

    /// Provides the bce_with_logits_loss operation or structure.
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

    /// Provides the cross_entropy_loss operation or structure.
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

/// Provides the QuantizedOps operation or structure.
pub trait QuantizedOps<B: Backend> {
    /// Provides the quantize operation or structure.
    fn quantize<K: FloatDType, Q: QuantDType>(_t: &B::Storage<K>) -> Result<B::Storage<Q>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "quantize",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the dequantize operation or structure.
    fn dequantize<Q: QuantDType, K: FloatDType>(_t: &B::Storage<Q>) -> Result<B::Storage<K>> {
        Err(crate::prelude::Error::UnsupportedBackendOperation {
            op: "dequantize",
            backend: core::any::type_name::<Self>(),
        })
    }
    /// Provides the quantized_matmul operation or structure.
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

/// Provides the OptimizerOps operation or structure.
pub trait OptimizerOps<B: Backend> {
    /// Provides the adamw_step operation or structure.
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
/// Provides the dummy operation or structure.
pub mod dummy {
    use super::*;
    use crate::nn::Reduction;
    use crate::prelude::Result;
    use crate::tensor::device::Device;
    use crate::tensor::device::DeviceId;
    use crate::tensor::dtype::DType;

    /// Provides the DummyBackend operation or structure.
    pub struct DummyBackend<T, D> {
        _marker: core::marker::PhantomData<(T, D)>,
    }

    impl<T: DType, D: Device + Clone + 'static> Clone for DummyBackend<T, D> {
        /// Provides the clone operation or structure.
        fn clone(&self) -> Self {
            DummyBackend {
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<T: DType, D: Device + Clone + 'static> Backend for DummyBackend<T, D> {
        /// Provides the Device operation or structure.
        type Device = D;
        /// Provides the FloatElem operation or structure.
        type FloatElem = T;
        /// Provides the IntElem operation or structure.
        type IntElem = i64;
        /// Provides the RawVar operation or structure.
        type RawVar = alloc::vec::Vec<usize>;
        /// Provides the Grads operation or structure.
        type Grads = ();
        /// Provides the Storage operation or structure.
        type Storage<K: DType> = alloc::vec::Vec<usize>;
        /// Provides the InnerBackend operation or structure.
        type InnerBackend = Self;

        /// Provides the shape operation or structure.
        fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
            t.clone()
        }
        /// Provides the format_tensor_display operation or structure.
        fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }
        /// Provides the format_tensor_debug operation or structure.
        fn format_tensor_debug<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }
        /// Provides the backward operation or structure.
        fn backward<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
            Ok(())
        }
        /// Provides the backward_with_nan_check operation or structure.
        fn backward_with_nan_check<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
            Ok(())
        }
        /// Provides the get_grad operation or structure.
        fn get_grad<K: DType>(
            _t: &Self::Storage<K>,
            _grads: &Self::Grads,
        ) -> Result<Option<Self::Storage<K>>> {
            Ok(None)
        }
        /// Provides the to_bytes operation or structure.
        fn to_bytes<K: DType>(_t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
            Ok(alloc::vec::Vec::new())
        }
        /// Provides the from_bytes operation or structure.
        fn from_bytes<K: DType>(
            _bytes: &[u8],
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<Self::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Provides the var_as_tensor operation or structure.
        fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
            Ok(var.clone())
        }
        /// Provides the var_from_tensor operation or structure.
        fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
            Ok(t.clone())
        }
        /// Provides the assign_var operation or structure.
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

    impl<T: DType, D: Device + Clone + 'static> CreationOps<Self> for DummyBackend<T, D> {
        /// Provides the zeros operation or structure.
        fn zeros<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Provides the ones operation or structure.
        fn ones<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Provides the rand operation or structure.
        fn rand<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Provides the randn operation or structure.
        fn randn<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Provides the var_zeros operation or structure.
        fn var_zeros<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Provides the var_ones operation or structure.
        fn var_ones<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Provides the var_rand operation or structure.
        fn var_rand<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Provides the var_randn operation or structure.
        fn var_randn<K: DType>(
            shape: &[usize],
            _dtype: DTypeId,
            _device: &DeviceId,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> NumericOps<Self> for DummyBackend<T, D> {
        /// Provides the add operation or structure.
        fn add<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        /// Provides the sub operation or structure.
        fn sub<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        /// Provides the mul operation or structure.
        fn mul<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        /// Provides the div operation or structure.
        fn div<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> FloatOps<Self> for DummyBackend<T, D> {
        /// Provides the add_scalar_float operation or structure.
        fn add_scalar_float<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the mul_scalar_float operation or structure.
        fn mul_scalar_float<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the relu operation or structure.
        fn relu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the step operation or structure.
        fn step<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the mish operation or structure.
        fn mish<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the elu operation or structure.
        fn elu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the gelu operation or structure.
        fn gelu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the abs operation or structure.
        fn abs<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the exp operation or structure.
        fn exp<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the neg operation or structure.
        fn neg<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the sqrt operation or structure.
        fn sqrt<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the log operation or structure.
        fn log<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the tanh operation or structure.
        fn tanh<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the sigmoid operation or structure.
        fn sigmoid<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the swish operation or structure.
        fn swish<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the softmax operation or structure.
        fn softmax<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> ReductionOps<Self> for DummyBackend<T, D> {
        /// Provides the sum_all operation or structure.
        fn sum_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Provides the mean_all operation or structure.
        fn mean_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Provides the max_all operation or structure.
        fn max_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Provides the min_all operation or structure.
        fn min_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Provides the sum_dim operation or structure.
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
        /// Provides the sum_keepdim operation or structure.
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
        /// Provides the mean_dim operation or structure.
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
        /// Provides the mean_keepdim operation or structure.
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
        /// Provides the max_dim operation or structure.
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
        /// Provides the max_keepdim operation or structure.
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
        /// Provides the min_dim operation or structure.
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
        /// Provides the min_keepdim operation or structure.
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
        /// Provides the argmax operation or structure.
        fn argmax<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        /// Provides the argmin operation or structure.
        fn argmin<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        /// Provides the topk operation or structure.
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
        /// Provides the argsort operation or structure.
        fn argsort<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: usize,
            _descending: bool,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
    }

    impl<T: DType, D: Device + Clone + 'static> TensorOps<Self> for DummyBackend<T, D> {
        /// Provides the matmul operation or structure.
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
        /// Provides the float_to_scalar operation or structure.
        fn float_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<f64> {
            Ok(0.0)
        }
        /// Provides the float_to_vec1 operation or structure.
        fn float_to_vec1<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<f64>> {
            Ok(alloc::vec![0.0])
        }
        /// Provides the int_to_scalar operation or structure.
        fn int_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<i64> {
            Ok(0)
        }
        /// Provides the int_to_vec1 operation or structure.
        fn int_to_vec1<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<i64>> {
            Ok(alloc::vec![0])
        }

        /// Provides the broadcast_as operation or structure.
        fn broadcast_as<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(s.to_vec())
        }
        /// Provides the broadcast_left operation or structure.
        fn broadcast_left<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = s.to_vec();
            out.extend_from_slice(t);
            Ok(out)
        }
        /// Provides the reshape operation or structure.
        fn reshape<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(s.to_vec())
        }
        /// Provides the transpose operation or structure.
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
        /// Provides the flatten operation or structure.
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
        /// Provides the slice operation or structure.
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
        /// Provides the narrow operation or structure.
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
        /// Provides the squeeze operation or structure.
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
        /// Provides the stack operation or structure.
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
        /// Provides the concat operation or structure.
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
        /// Provides the tensor_to_dtype operation or structure.
        fn tensor_to_dtype<K: DType, K2: DType>(
            t: &<Self as Backend>::Storage<K>,
            _dtype: DTypeId,
        ) -> Result<<Self as Backend>::Storage<K2>> {
            Ok(t.clone())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> ModuleOps<Self> for DummyBackend<T, D> {
        /// Provides the layer_norm operation or structure.
        fn layer_norm<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            _e: f32,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Provides the batch_norm operation or structure.
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
        /// Provides the embedding operation or structure.
        fn embedding<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<KInt>,
            _w: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Provides the conv1d operation or structure.
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
        /// Provides the conv2d operation or structure.
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
        /// Provides the conv_transpose2d operation or structure.
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
        /// Provides the max_pool2d operation or structure.
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
        /// Provides the avg_pool2d operation or structure.
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
        /// Provides the adaptive_avg_pool2d operation or structure.
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

    impl<T: DType, D: Device + Clone + 'static> LossOps<Self> for DummyBackend<T, D> {
        /// Provides the mse_loss operation or structure.
        fn mse_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Provides the l1_loss operation or structure.
        fn l1_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Provides the bce_with_logits_loss operation or structure.
        fn bce_with_logits_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Provides the cross_entropy_loss operation or structure.
        fn cross_entropy_loss<K: DType, KInt: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<KInt>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
    }

    impl<T: DType, D: Device + Clone + 'static> QuantizedOps<Self> for DummyBackend<T, D> {
        /// Provides the quantize operation or structure.
        fn quantize<K: FloatDType, Q: QuantDType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<Q>> {
            Ok(alloc::vec![])
        }
        /// Provides the dequantize operation or structure.
        fn dequantize<Q: QuantDType, K: FloatDType>(
            _t: &<Self as Backend>::Storage<Q>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Provides the quantized_matmul operation or structure.
        fn quantized_matmul<Q: QuantDType>(
            _lhs: &<Self as Backend>::Storage<Q>,
            _rhs: &<Self as Backend>::Storage<Q>,
        ) -> Result<<Self as Backend>::Storage<f32>> {
            Ok(alloc::vec![])
        }
    }
    impl<T: DType, D: Device + Clone + 'static> OptimizerOps<Self> for DummyBackend<T, D> {}
}
