//! Runtime backend selection used by `KindleBackend<_, Dyn>`.

use kindle_core::prelude::*;

/// Backend whose concrete implementation is selected from a [`DeviceId`].
#[derive(Clone)]
pub struct DispatchBackend<T = f32, D = Dyn>(core::marker::PhantomData<(T, D)>);

impl<T: DType, D: Device, K: DType> SupportsDType<K> for DispatchBackend<T, D> {
    fn resolve_dtype(field: &K::Field, device: &DeviceId) -> Result<DTypeId> {
        let dtype = K::to_kindle(field);
        match device.kind() {
            DeviceKind::Cpu => Ok(dtype),
            DeviceKind::Wgpu | DeviceKind::Cuda if dtype == DTypeId::F32 => Ok(dtype),
            DeviceKind::Wgpu => Err(Error::UnsupportedDType {
                dtype,
                backend: "Wgpu",
                op: "create",
            }),
            DeviceKind::Cuda => Err(Error::UnsupportedDType {
                dtype,
                backend: "Cuda",
                op: "create",
            }),
            _ => Err(Error::BackendUnavailable { backend: "Unknown" }),
        }
    }
}

/// Storage owned by a runtime-selected backend.
#[derive(Clone)]
#[non_exhaustive]
pub enum DispatchStorage {
    #[cfg(feature = "cpu")]
    /// Native CPU storage.
    Cpu(crate::cpu::CpuStorage),
    #[cfg(feature = "wgpu")]
    /// WebGPU storage.
    Wgpu(crate::wgpu::storage::WgpuStorage),
    #[cfg(feature = "cuda")]
    /// CUDA storage.
    Cuda(crate::cuda::storage::CudaStorage),
    #[doc(hidden)]
    Unavailable,
}

/// Mutable parameter storage owned by a runtime-selected backend.
#[derive(Clone)]
#[non_exhaustive]
pub enum DispatchVar {
    #[cfg(feature = "cpu")]
    Cpu(crate::cpu::CpuVar),
    #[cfg(feature = "wgpu")]
    Wgpu(crate::wgpu::backend::WgpuVar),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::backend::CudaVar),
    #[doc(hidden)]
    Unavailable,
}

/// Gradient collection owned by a runtime-selected backend.
#[non_exhaustive]
pub enum DispatchGrads {
    #[cfg(feature = "cpu")]
    Cpu(crate::cpu::CpuGrads),
    #[cfg(feature = "wgpu")]
    Wgpu(crate::wgpu::backend::WgpuGrads),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::backend::CudaGrads),
    #[doc(hidden)]
    Unavailable,
}

fn unavailable(kind: DeviceKind) -> Error {
    Error::BackendUnavailable {
        backend: match kind {
            DeviceKind::Cpu => "Cpu",
            DeviceKind::Wgpu => "Wgpu",
            DeviceKind::Cuda => "Cuda",
            _ => "Unknown",
        },
    }
}

fn storage_device(storage: &DispatchStorage) -> DeviceId {
    match storage {
        #[cfg(feature = "cpu")]
        DispatchStorage::Cpu(_) => DeviceId::cpu(),
        #[cfg(feature = "wgpu")]
        DispatchStorage::Wgpu(_) => DeviceId::wgpu(0),
        #[cfg(feature = "cuda")]
        DispatchStorage::Cuda(value) => DeviceId::cuda(value.buffer.device_id),
        DispatchStorage::Unavailable => DeviceId::cpu(),
    }
}

macro_rules! dispatch_unary {
    ($storage:expr, $method:ident $(, $arg:expr)*) => {
        match $storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => crate::cpu::CpuBackendImpl::<T, Cpu>::$method::<K>(value $(, $arg)*)
                .map(DispatchStorage::Cpu),
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => crate::wgpu::WgpuBackendImpl::<T, Wgpu>::$method::<K>(value $(, $arg)*)
                .map(DispatchStorage::Wgpu),
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => crate::cuda::CudaBackendImpl::<T, Cuda>::$method::<K>(value $(, $arg)*)
                .map(DispatchStorage::Cuda),
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    };
}

macro_rules! dispatch_binary {
    ($lhs:expr, $rhs:expr, $method:ident) => {
        match ($lhs, $rhs) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(lhs), DispatchStorage::Cpu(rhs)) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::$method::<K>(lhs, rhs)
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(lhs), DispatchStorage::Wgpu(rhs)) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::$method::<K>(lhs, rhs)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(lhs), DispatchStorage::Cuda(rhs)) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::$method::<K>(lhs, rhs)
                    .map(DispatchStorage::Cuda)
            }
            (lhs, rhs) => Err(Error::DeviceMismatch {
                left: storage_device(lhs),
                right: storage_device(rhs),
            }),
        }
    };
}

macro_rules! create_dispatch {
    ($method:ident, $shape:expr, $dtype:expr, $device:expr) => {
        match $device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    return crate::cpu::CpuBackendImpl::<T, Cpu>::$method::<K>(
                        $shape, $dtype, $device,
                    )
                    .map(DispatchStorage::Cpu);
                }
                #[cfg(not(feature = "cpu"))]
                Err(unavailable(DeviceKind::Cpu))
            }
            DeviceKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    return crate::wgpu::WgpuBackendImpl::<T, Wgpu>::$method::<K>(
                        $shape, $dtype, $device,
                    )
                    .map(DispatchStorage::Wgpu);
                }
                #[cfg(not(feature = "wgpu"))]
                Err(unavailable(DeviceKind::Wgpu))
            }
            DeviceKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    return crate::cuda::CudaBackendImpl::<T, Cuda>::$method::<K>(
                        $shape, $dtype, $device,
                    )
                    .map(DispatchStorage::Cuda);
                }
                #[cfg(not(feature = "cuda"))]
                Err(unavailable(DeviceKind::Cuda))
            }
            _ => Err(Error::BackendUnavailable { backend: "Unknown" }),
        }
    };
}

macro_rules! create_var_dispatch {
    ($method:ident, $shape:expr, $dtype:expr, $device:expr) => {
        match $device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    return crate::cpu::CpuBackendImpl::<T, Cpu>::$method::<K>(
                        $shape, $dtype, $device,
                    )
                    .map(DispatchVar::Cpu);
                }
                #[cfg(not(feature = "cpu"))]
                Err(unavailable(DeviceKind::Cpu))
            }
            DeviceKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    return crate::wgpu::WgpuBackendImpl::<T, Wgpu>::$method::<K>(
                        $shape, $dtype, $device,
                    )
                    .map(DispatchVar::Wgpu);
                }
                #[cfg(not(feature = "wgpu"))]
                Err(unavailable(DeviceKind::Wgpu))
            }
            DeviceKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    return crate::cuda::CudaBackendImpl::<T, Cuda>::$method::<K>(
                        $shape, $dtype, $device,
                    )
                    .map(DispatchVar::Cuda);
                }
                #[cfg(not(feature = "cuda"))]
                Err(unavailable(DeviceKind::Cuda))
            }
            _ => Err(Error::BackendUnavailable { backend: "Unknown" }),
        }
    };
}

impl<T: DType, D: Device> Backend for DispatchBackend<T, D> {
    type Device = D;
    type FloatElem = T;
    type IntElem = i64;
    type Storage<K: DType> = DispatchStorage;
    type RawVar = DispatchVar;
    type Grads = DispatchGrads;
    type InnerBackend = Self;
    fn shape<K: DType>(storage: &Self::Storage<K>) -> Vec<usize> {
        match storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => crate::cpu::CpuBackendImpl::<T, Cpu>::shape::<K>(value),
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::shape::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::shape::<K>(value)
            }
            DispatchStorage::Unavailable => Vec::new(),
        }
    }

    fn storage_dtype<K: DType>(storage: &Self::Storage<K>) -> Option<DTypeId> {
        match storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::storage_dtype::<K>(value)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::storage_dtype::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::storage_dtype::<K>(value)
            }
            DispatchStorage::Unavailable => None,
        }
    }

    fn storage_device<K: DType>(storage: &Self::Storage<K>) -> Option<DeviceId> {
        Some(storage_device(storage))
    }

    fn format_tensor_display<K: DType>(storage: &Self::Storage<K>) -> String {
        format!("DispatchTensor(shape={:?})", Self::shape::<K>(storage))
    }

    fn format_tensor_debug<K: DType>(storage: &Self::Storage<K>) -> String {
        Self::format_tensor_display::<K>(storage)
    }

    fn backward<K: DType>(storage: &Self::Storage<K>) -> Result<Self::Grads> {
        match storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::backward::<K>(value).map(DispatchGrads::Cpu)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::backward::<K>(value)
                    .map(DispatchGrads::Wgpu)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::backward::<K>(value)
                    .map(DispatchGrads::Cuda)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }

    fn backward_with_nan_check<K: DType>(storage: &Self::Storage<K>) -> Result<Self::Grads> {
        match storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::backward_with_nan_check::<K>(value)
                    .map(DispatchGrads::Cpu)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::backward_with_nan_check::<K>(value)
                    .map(DispatchGrads::Wgpu)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::backward_with_nan_check::<K>(value)
                    .map(DispatchGrads::Cuda)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }

    fn get_grad<K: DType>(
        storage: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        match (storage, grads) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(value), DispatchGrads::Cpu(gs)) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::get_grad::<K>(value, gs)
                    .map(|value| value.map(DispatchStorage::Cpu))
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(value), DispatchGrads::Wgpu(gs)) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::get_grad::<K>(value, gs)
                    .map(|value| value.map(DispatchStorage::Wgpu))
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(value), DispatchGrads::Cuda(gs)) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::get_grad::<K>(value, gs)
                    .map(|value| value.map(DispatchStorage::Cuda))
            }
            _ => Err(Error::DeviceMismatch {
                left: DeviceId::cpu(),
                right: DeviceId::cpu(),
            }),
        }
    }

    fn to_bytes<K: DType>(storage: &Self::Storage<K>) -> Result<Vec<u8>> {
        match storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::to_bytes::<K>(value)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::to_bytes::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::to_bytes::<K>(value)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }

    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        match device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                return crate::cpu::CpuBackendImpl::<T, Cpu>::from_bytes::<K>(
                    bytes, shape, dtype, device,
                )
                .map(DispatchStorage::Cpu);
                #[cfg(not(feature = "cpu"))]
                Err(unavailable(DeviceKind::Cpu))
            }
            DeviceKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                return crate::wgpu::WgpuBackendImpl::<T, Wgpu>::from_bytes::<K>(
                    bytes, shape, dtype, device,
                )
                .map(DispatchStorage::Wgpu);
                #[cfg(not(feature = "wgpu"))]
                Err(unavailable(DeviceKind::Wgpu))
            }
            DeviceKind::Cuda => {
                #[cfg(feature = "cuda")]
                return crate::cuda::CudaBackendImpl::<T, Cuda>::from_bytes::<K>(
                    bytes, shape, dtype, device,
                )
                .map(DispatchStorage::Cuda);
                #[cfg(not(feature = "cuda"))]
                Err(unavailable(DeviceKind::Cuda))
            }
            _ => Err(Error::BackendUnavailable { backend: "Unknown" }),
        }
    }

    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        match var {
            #[cfg(feature = "cpu")]
            DispatchVar::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::var_as_tensor::<K>(value)
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            DispatchVar::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::var_as_tensor::<K>(value)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            DispatchVar::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::var_as_tensor::<K>(value)
                    .map(DispatchStorage::Cuda)
            }
            DispatchVar::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }

    fn var_from_tensor<K: DType>(storage: &Self::Storage<K>) -> Result<Self::RawVar> {
        match storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::var_from_tensor::<K>(value)
                    .map(DispatchVar::Cpu)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::var_from_tensor::<K>(value)
                    .map(DispatchVar::Wgpu)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::var_from_tensor::<K>(value)
                    .map(DispatchVar::Cuda)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }

    fn assign_var<K: DType>(var: &mut Self::RawVar, storage: &Self::Storage<K>) -> Result<()> {
        match (var, storage) {
            #[cfg(feature = "cpu")]
            (DispatchVar::Cpu(var), DispatchStorage::Cpu(value)) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::assign_var::<K>(var, value)
            }
            #[cfg(feature = "wgpu")]
            (DispatchVar::Wgpu(var), DispatchStorage::Wgpu(value)) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::assign_var::<K>(var, value)
            }
            #[cfg(feature = "cuda")]
            (DispatchVar::Cuda(var), DispatchStorage::Cuda(value)) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::assign_var::<K>(var, value)
            }
            _ => Err(Error::UnsupportedBackendOperation {
                op: "assign_var_cross_device",
                backend: "DispatchBackend",
            }),
        }
    }
}

impl<T: DType, D: Device> CreationOps<Self> for DispatchBackend<T, D> {
    fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(zeros, shape, dtype, device)
    }
    fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(ones, shape, dtype, device)
    }
    fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(rand, shape, dtype, device)
    }
    fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(randn, shape, dtype, device)
    }
    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<DispatchVar> {
        create_var_dispatch!(var_zeros, shape, dtype, device)
    }
    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<DispatchVar> {
        create_var_dispatch!(var_ones, shape, dtype, device)
    }
    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<DispatchVar> {
        create_var_dispatch!(var_rand, shape, dtype, device)
    }
    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeId,
        device: &DeviceId,
    ) -> Result<DispatchVar> {
        create_var_dispatch!(var_randn, shape, dtype, device)
    }
}

impl<T: DType, D: Device> TensorOps<Self> for DispatchBackend<T, D> {
    fn reshape<K: DType>(t: &DispatchStorage, shape: &[usize]) -> Result<DispatchStorage> {
        dispatch_unary!(t, reshape, shape)
    }
    fn transpose<K: DType>(
        t: &DispatchStorage,
        dim1: usize,
        dim2: usize,
    ) -> Result<DispatchStorage> {
        dispatch_unary!(t, transpose, dim1, dim2)
    }
    fn matmul<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, matmul)
    }
    fn broadcast_as<K: DType>(t: &DispatchStorage, shape: &[usize]) -> Result<DispatchStorage> {
        dispatch_unary!(t, broadcast_as, shape)
    }
    fn narrow<K: DType>(
        t: &DispatchStorage,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<DispatchStorage> {
        dispatch_unary!(t, narrow, dim, start, len)
    }
    fn squeeze<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, squeeze, dim)
    }
    fn slice<K: DType>(t: &DispatchStorage, ranges: &[(usize, usize)]) -> Result<DispatchStorage> {
        dispatch_unary!(t, slice, ranges)
    }
    fn flatten<K: DType>(t: &DispatchStorage, start: usize, end: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, flatten, start, end)
    }
    fn broadcast_left<K: DType>(t: &DispatchStorage, shape: &[usize]) -> Result<DispatchStorage> {
        dispatch_unary!(t, broadcast_left, shape)
    }
    fn float_to_scalar<K: DType>(t: &DispatchStorage) -> Result<f64> {
        match t {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::float_to_scalar::<K>(value)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::float_to_scalar::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::float_to_scalar::<K>(value)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }
    fn float_to_vec1<K: DType>(t: &DispatchStorage) -> Result<Vec<f64>> {
        match t {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<T, Cpu>::float_to_vec1::<K>(value)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<T, Wgpu>::float_to_vec1::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<T, Cuda>::float_to_vec1::<K>(value)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }
}
impl<T: DType, D: Device> NumericOps<Self> for DispatchBackend<T, D> {
    fn add<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, add)
    }
    fn sub<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, sub)
    }
    fn mul<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, mul)
    }
    fn div<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, div)
    }
}
impl<T: DType, D: Device> FloatOps<Self> for DispatchBackend<T, D> {
    fn relu<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, relu)
    }
    fn step<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, step)
    }
    fn mish<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, mish)
    }
    fn elu<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, elu)
    }
    fn gelu<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, gelu)
    }
    fn abs<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, abs)
    }
    fn exp<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, exp)
    }
    fn neg<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, neg)
    }
    fn sqrt<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, sqrt)
    }
    fn log<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, log)
    }
    fn tanh<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, tanh)
    }
    fn sigmoid<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, sigmoid)
    }
    fn swish<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, swish)
    }
    fn softmax<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, softmax, dim)
    }
    fn add_scalar_float<K: DType>(t: &DispatchStorage, scalar: f64) -> Result<DispatchStorage> {
        dispatch_unary!(t, add_scalar_float, scalar)
    }
    fn mul_scalar_float<K: DType>(t: &DispatchStorage, scalar: f64) -> Result<DispatchStorage> {
        dispatch_unary!(t, mul_scalar_float, scalar)
    }
}
impl<T: DType, D: Device> ReductionOps<Self> for DispatchBackend<T, D> {
    fn sum_all<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, sum_all)
    }
    fn mean_all<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, mean_all)
    }
    fn max_all<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, max_all)
    }
    fn min_all<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, min_all)
    }
    fn sum_dim<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, sum_dim, dim)
    }
    fn sum_keepdim<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, sum_keepdim, dim)
    }
    fn mean_dim<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, mean_dim, dim)
    }
    fn mean_keepdim<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, mean_keepdim, dim)
    }
    fn max_dim<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, max_dim, dim)
    }
    fn max_keepdim<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, max_keepdim, dim)
    }
    fn min_dim<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, min_dim, dim)
    }
    fn min_keepdim<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, min_keepdim, dim)
    }
}
impl<T: DType, D: Device> QuantizedOps<Self> for DispatchBackend<T, D> {}
impl<T: DType, D: Device> OptimizerOps<Self> for DispatchBackend<T, D> {}
impl<T: DType, D: Device> ModuleOps<Self> for DispatchBackend<T, D> {}
impl<T: DType, D: Device> LossOps<Self> for DispatchBackend<T, D> {}
