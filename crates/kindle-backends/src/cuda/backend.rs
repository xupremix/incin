use crate::cuda::storage::CudaStorage;
use kindle_core::prelude::*;
use std::fmt::Debug;

/// A dedicated Cuda backend.
#[derive(Debug, Clone, Copy)]
pub struct CudaBackend<T: DType, D: Device> {
    _marker: core::marker::PhantomData<(T, D)>,
}

#[derive(Clone)]
pub struct CudaVar {
    pub storage: CudaStorage,
}


impl<T: DType, D: Device> TensorOps<Self> for CudaBackend<T, D> {
    fn concat<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cuda::ops::shape::launch_concat(tensors, dim)
    }
}


impl<T: DType, D: Device> NumericOps<Self> for CudaBackend<T, D> {
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        crate::cuda::ops::elementwise::launch_binary_op("add", "a + b", lhs, rhs, &out_shape)
    }

    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        crate::cuda::ops::elementwise::launch_binary_op("sub", "a - b", lhs, rhs, &out_shape)
    }

    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        crate::cuda::ops::elementwise::launch_binary_op("mul", "a * b", lhs, rhs, &out_shape)
    }

    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        crate::cuda::ops::elementwise::launch_binary_op("div", "a / b", lhs, rhs, &out_shape)
    }
}

impl<T: DType, D: Device> FloatOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> CreationOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> ReductionOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> QuantizedOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> OptimizerOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> ModuleOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> LossOps<Self> for CudaBackend<T, D> {}

impl<T: DType, D: Device> Backend for CudaBackend<T, D> {
    type Device = D;
    type FloatElem = T;
    type IntElem = f32; // Placeholder

    type Storage<K: DType> = CudaStorage;
    type RawVar = CudaVar;
    type Grads = ();

    type InnerBackend = Self;

    type BackendWithDevice<NewD: Device> = CudaBackend<T, NewD>;

    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
        t.shape.clone()
    }
    fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
        "CudaTensor(...)".to_string()
    }
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String {
        format!("CudaTensor(shape={:?})", t.shape)
    }
    fn backward<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
        Ok(())
    }
    fn backward_with_nan_check<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
        Ok(())
    }
    fn get_grad<K: DType>(_t: &Self::Storage<K>, _grads: &Self::Grads) -> Result<Option<Self::Storage<K>>> {
        Ok(None)
    }
    fn to_bytes<K: DType>(_t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
        Err(Error::UnsupportedBackendOperation { op: "to_bytes", backend: "CudaBackend" })
    }
    fn from_bytes<K: DType>(_bytes: &[u8], _shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::Storage<K>> {
        Err(Error::UnsupportedBackendOperation { op: "from_bytes", backend: "CudaBackend" })
    }
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        Ok(var.storage.clone())
    }
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        Ok(CudaVar { storage: t.clone() })
    }
    fn var_to_device(var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
        Ok(var.clone())
    }
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var.storage = tensor.clone();
        Ok(())
    }
}
