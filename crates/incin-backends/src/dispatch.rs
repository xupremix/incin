//! Runtime backend selection used by `IncinBackend<_, Dyn>`.

use alloc::vec::Vec;
use incin_core::backend_authoring::SupportsDType;
use incin_core::backend_authoring::*;
use incin_core::error::{BackendError, Error, Result};
use incin_core::shapes::{Dyn, OperationKind};
use incin_core::tensor::device::{Device, DeviceId, DeviceKind};
use incin_core::tensor::dtype::{DType, DTypeDescriptor, bf16, f16};

#[cfg(feature = "cpu")]
use incin_core::tensor::device::Cpu;
#[cfg(feature = "cuda")]
use incin_core::tensor::device::Cuda;
#[cfg(feature = "metal")]
use incin_core::tensor::device::Metal;
#[cfg(feature = "wgpu")]
use incin_core::tensor::device::Wgpu;

/// Backend whose concrete implementation is selected from a [`DeviceId`].
#[derive(Clone)]
pub struct DispatchBackend<D = Dyn>(core::marker::PhantomData<D>);

impl<D> DispatchBackend<D> {
    /// Construct the stateless dispatching executor.
    ///
    /// Without this the descriptor path on the runtime-selected backend is
    /// unreachable from outside the crate: `ExecutionContext` owns a backend
    /// value and the `PhantomData` field is private, so no caller could build
    /// one to hand to `Execute`.
    #[must_use]
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<D> Default for DispatchBackend<D> {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_dispatch_storage_dtype {
    ($($dtype:ty),+ $(,)?) => {
        $(
            impl<D: Device> SupportsDType<$dtype> for DispatchBackend<D> {
                fn resolve_dtype(field: &<$dtype as DType>::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
                    Ok(<$dtype as DType>::descriptor(field))
                }
            }
        )+
    };
}

impl_dispatch_storage_dtype!(f32, f64, f16, bf16, u8, u32, i64, bool);

impl<D: Device> SupportsDType<Dyn> for DispatchBackend<D> {
    fn resolve_dtype(field: &DTypeDescriptor, _device: &DeviceId) -> Result<DTypeDescriptor> {
        Ok(*field)
    }
}

/// Storage owned by a runtime-selected backend.
///
/// Allowed for the same reason `DispatchVar` below is, and only visible in a
/// single-backend build: storage embeds `TensorMeta`, whose `ShapeBuf` and
/// `StrideBuf` are inline to `INLINE_RANK` so eager operations allocate nothing
/// for their metadata (SHP-003), which leaves any one backend's variant far
/// larger than the empty `Unavailable` one. Boxing it would put back the heap
/// allocation SHP-003 removed. Revisit when `EXE-009` settles.
#[allow(clippy::large_enum_variant)]
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
    #[cfg(feature = "metal")]
    /// Metal storage.
    Metal(crate::metal::storage::MetalStorage),
    #[doc(hidden)]
    Unavailable,
}

impl incin_core::backend_authoring::StorageOutput for DispatchStorage {}

/// Mutable parameter storage owned by a runtime-selected backend.
///
/// The variants differ in size by design rather than by oversight. A backend
/// variable embeds its storage, and storage embeds `TensorMeta`, whose
/// `ShapeBuf` and `StrideBuf` are inline to `INLINE_RANK` so that ordinary eager
/// operations allocate nothing for their metadata (SHP-003). That inlining is
/// what makes `WgpuVar` 216 bytes against `CpuVar`'s 8, and boxing the large
/// variant would put the heap allocation back on the path SHP-003 removed it
/// from. This enum is also `EXE-009`'s active target, which is reshaping the
/// dispatch surface; changing the layout of a public enum underneath that work
/// would be churn against a moving file. Revisit when `EXE-009` settles.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
#[non_exhaustive]
pub enum DispatchVar {
    #[cfg(feature = "cpu")]
    Cpu(crate::cpu::CpuVar),
    #[cfg(feature = "wgpu")]
    Wgpu(crate::wgpu::backend::WgpuVar),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::backend::CudaVar),
    #[cfg(feature = "metal")]
    Metal(crate::metal::backend::MetalVar),
    #[doc(hidden)]
    Unavailable,
}

impl incin_core::backend_authoring::ExecuteOutput for DispatchVar {}

/// Gradient collection owned by a runtime-selected backend.
#[non_exhaustive]
pub enum DispatchGrads {
    #[cfg(feature = "cpu")]
    Cpu(crate::cpu::CpuGrads),
    #[cfg(feature = "wgpu")]
    Wgpu(crate::wgpu::backend::WgpuGrads),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::backend::CudaGrads),
    #[cfg(feature = "metal")]
    Metal(crate::metal::tape::MetalGrads),
    #[doc(hidden)]
    Unavailable,
}

/// Report a device this dispatcher cannot reach, by name.
///
/// The wildcard falls back to [`DeviceKind::name`] rather than to `"Unknown"`.
/// `DeviceKind` is `#[non_exhaustive]`, so the arm cannot be removed, but every
/// kind can already say what it is called: spelling three of them out here and
/// answering `"Unknown"` for the rest meant a Metal device — a first-class
/// feature this dispatcher happens not to carry a variant for — reported a
/// failure that named no backend at all.
fn unavailable(kind: DeviceKind) -> Error {
    Error::BackendUnavailable {
        backend: match kind {
            DeviceKind::Cpu => "Cpu",
            DeviceKind::Wgpu => "Wgpu",
            DeviceKind::Cuda => "Cuda",
            DeviceKind::Metal => "Metal",
            other => other.name(),
        },
    }
}

#[cfg(feature = "cpu")]
macro_rules! cpu_unary_call {
    (sub_scalar, $value:expr, $scalar:expr) => {
        crate::cpu::ops::shape_ops::sub_scalar_storage($value, $scalar)
    };
    (div_scalar, $value:expr, $scalar:expr) => {
        crate::cpu::ops::shape_ops::div_scalar_storage($value, $scalar)
    };
    (relu, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_relu($value)
    };
    (step, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_step($value)
    };
    (mish, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_mish($value)
    };
    (elu, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_elu($value)
    };
    (gelu, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_gelu($value)
    };
    (abs, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_abs($value)
    };
    (exp, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_exp($value)
    };
    (neg, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_neg($value)
    };
    (sqrt, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_sqrt($value)
    };
    (log, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_log($value)
    };
    (tanh, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_tanh($value)
    };
    (sigmoid, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_sigmoid($value)
    };
    (swish, $value:expr $(, $arg:expr)*) => {
        crate::cpu::ops::elementwise::canonical_swish($value)
    };
    (softmax, $value:expr, $dim:expr) => {
        crate::cpu::ops::elementwise::canonical_softmax::<Cpu>($value, $dim)
    };
    (add_scalar_float, $value:expr, $scalar:expr) => {
        crate::cpu::ops::elementwise::canonical_add_scalar($value, $scalar)
    };
    (mul_scalar_float, $value:expr, $scalar:expr) => {
        crate::cpu::ops::elementwise::canonical_mul_scalar($value, $scalar)
    };
    (powf, $value:expr, $exponent:expr) => {
        crate::cpu::ops::elementwise::canonical_powf($value, $exponent)
    };
    (clamp, $value:expr, $min:expr, $max:expr) => {
        crate::cpu::ops::elementwise::canonical_clamp($value, $min, $max)
    };
    (tan, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_tan($value)
    };
    (asin, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_asin($value)
    };
    (acos, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_acos($value)
    };
    (atan, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_atan($value)
    };
    (sinh, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_sinh($value)
    };
    (cosh, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_cosh($value)
    };
    (asinh, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_asinh($value)
    };
    (acosh, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_acosh($value)
    };
    (atanh, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_atanh($value)
    };
    (erf, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_erf($value)
    };
    (rsqrt, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_rsqrt($value)
    };
    (trunc, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_trunc($value)
    };
    (frac, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_frac($value)
    };
    (sign, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_unary(
            crate::cpu::ops::elementwise_kernel::UnaryOp::Sign,
            $value,
        )
    };
    (sin, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_unary(
            crate::cpu::ops::elementwise_kernel::UnaryOp::Sin,
            $value,
        )
    };
    (cos, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_unary(
            crate::cpu::ops::elementwise_kernel::UnaryOp::Cos,
            $value,
        )
    };
    (floor, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_unary(
            crate::cpu::ops::elementwise_kernel::UnaryOp::Floor,
            $value,
        )
    };
    (ceil, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_unary(
            crate::cpu::ops::elementwise_kernel::UnaryOp::Ceil,
            $value,
        )
    };
    (round, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_unary(
            crate::cpu::ops::elementwise_kernel::UnaryOp::Round,
            $value,
        )
    };
    (log2, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_unary(
            crate::cpu::ops::elementwise_kernel::UnaryOp::Log2,
            $value,
        )
    };
    (log10, $value:expr) => {
        crate::cpu::ops::elementwise::canonical_unary(
            crate::cpu::ops::elementwise_kernel::UnaryOp::Log10,
            $value,
        )
    };
}

macro_rules! dispatch_unary {
    ($storage:expr, $method:ident $(, $arg:expr)*) => {
        match $storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => cpu_unary_call!($method, value $(, $arg)*)
                .map(DispatchStorage::Cpu),
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => crate::wgpu::WgpuBackendImpl::<Wgpu>::$method::<K>(value $(, $arg)*)
                .map(DispatchStorage::Wgpu),
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => crate::cuda::CudaBackendImpl::<Cuda>::$method::<K>(value $(, $arg)*)
                .map(DispatchStorage::Cuda),
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => crate::metal::MetalBackendImpl::<Metal>::$method::<K>(value $(, $arg)*)
                .map(DispatchStorage::Metal),
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    };
}

macro_rules! create_dispatch {
    ($helper:ident, $method:ident, $shape:expr, $dtype:expr, $device:expr) => {
        match $device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    let total = crate::cpu::stride::checked_numel($shape)?;
                    return crate::cpu::creation::$helper(total, $shape, $dtype, $device)
                        .map(DispatchStorage::Cpu);
                }
                #[cfg(not(feature = "cpu"))]
                Err(unavailable(DeviceKind::Cpu))
            }
            DeviceKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    return crate::wgpu::WgpuBackendImpl::<Wgpu>::$method::<K>(
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
                    return crate::cuda::CudaBackendImpl::<Cuda>::$method::<K>(
                        $shape, $dtype, $device,
                    )
                    .map(DispatchStorage::Cuda);
                }
                #[cfg(not(feature = "cuda"))]
                Err(unavailable(DeviceKind::Cuda))
            }
            DeviceKind::Metal => {
                #[cfg(feature = "metal")]
                {
                    return crate::metal::MetalBackendImpl::<Metal>::$method::<K>(
                        $shape, $dtype, $device,
                    )
                    .map(DispatchStorage::Metal);
                }
                #[cfg(not(feature = "metal"))]
                Err(unavailable(DeviceKind::Metal))
            }
            other => Err(unavailable(other)),
        }
    };
}

macro_rules! variable_executors {
    ($(($operation:ident, $cpu_method:ident, $backend_method:ident)),* $(,)?) => {$ (
        impl<D: Device> Execute<op::$operation> for DispatchBackend<D> {
            type Output = DispatchVar;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> core::result::Result<DispatchVar, BackendError> {
                let attributes = request.operation.descriptor().attributes();
                create_var_execute_dispatch!($cpu_method, $backend_method, &attributes.shape, attributes.dtype, &attributes.device)
                    .map_err(|error| BackendError::Execution {
                        operation: OperationKind::$operation,
                        message: alloc::format!("{error}").into(),
                    })
            }
        }
    )*};
}

macro_rules! create_var_execute_dispatch {
    ($helper:ident, $method:ident, $shape:expr, $dtype:expr, $device:expr) => {
        match $device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    let total = crate::cpu::stride::checked_numel($shape).map_err(|error| {
                        BackendError::Execution {
                            operation: OperationKind::Storage,
                            message: alloc::format!("{error}").into(),
                        }
                    })?;
                    crate::cpu::creation::$helper(total, $shape, $dtype, $device)
                        .map(DispatchVar::Cpu)
                }
                #[cfg(not(feature = "cpu"))]
                Err(unavailable(DeviceKind::Cpu))
            }
            DeviceKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    crate::wgpu::WgpuBackendImpl::<Wgpu>::$method::<Dyn>($shape, $dtype, $device)
                        .map(DispatchVar::Wgpu)
                }
                #[cfg(not(feature = "wgpu"))]
                Err(unavailable(DeviceKind::Wgpu))
            }
            DeviceKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::CudaBackendImpl::<Cuda>::$method::<Dyn>($shape, $dtype, $device)
                        .map(DispatchVar::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                Err(unavailable(DeviceKind::Cuda))
            }
            other => Err(unavailable(other)),
        }
    };
}

variable_executors![
    (VariableZeros, var_zeros_with_total, var_zeros),
    (VariableOnes, var_ones_with_total, var_ones),
    (VariableUniformRandom, var_rand_with_total, var_rand),
    (VariableNormalRandom, var_randn_with_total, var_randn),
];

macro_rules! scalar_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$ (
        impl<D: Device> Execute<op::$operation> for DispatchBackend<D> {
            type Output = DispatchStorage;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> core::result::Result<DispatchStorage, BackendError> {
                let operation = OperationKind::$operation;
                let input = request.inputs.first().ok_or(BackendError::InvalidInput {
                    operation,
                    reason: "scalar operation requires one tensor input",
                })?;
                let input = input.downcast_ref::<DispatchStorage>().ok_or(
                    BackendError::InvalidInput {
                        operation,
                        reason: "input is not DispatchStorage",
                    },
                )?;
                let value = request.operation.descriptor().attributes().value;
                Self::$method::<Dyn>(input, value).map_err(|error| {
                    BackendError::Execution {
                        operation,
                        message: alloc::format!("{error}").into(),
                    }
                })
            }
        }
    )*};
}

scalar_executors![(AddScalar, add_scalar_float), (MulScalar, mul_scalar_float),];

impl<D: Device> incin_core::backend_authoring::StorageBackend for DispatchBackend<D> {
    type Device = D;
    const BACKEND_NAME: &'static str = "Dispatch";
    type Storage<K: DType> = DispatchStorage;

    fn metadata<K: DType>(t: &Self::Storage<K>) -> &incin_core::backend_authoring::TensorMeta {
        match t {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => crate::cpu::CpuBackendImpl::<Cpu>::metadata::<K>(value),
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::metadata::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<Cuda>::metadata::<K>(value)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                crate::metal::MetalBackendImpl::<Metal>::metadata::<K>(value)
            }
            DispatchStorage::Unavailable => unreachable!("dispatch storage unavailable"),
        }
    }

    fn fresh_autograd_identity<K: DType>(storage: Self::Storage<K>) -> Self::Storage<K> {
        match storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => DispatchStorage::Cpu(
                crate::cpu::CpuBackendImpl::<Cpu>::fresh_autograd_identity::<K>(value),
            ),
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                DispatchStorage::Wgpu(
                    crate::wgpu::WgpuBackendImpl::<Wgpu>::fresh_autograd_identity::<K>(value),
                )
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                DispatchStorage::Cuda(
                    crate::cuda::CudaBackendImpl::<Cuda>::fresh_autograd_identity::<K>(value),
                )
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => DispatchStorage::Metal(
                crate::metal::MetalBackendImpl::<Metal>::fresh_autograd_identity::<K>(value),
            ),
            DispatchStorage::Unavailable => DispatchStorage::Unavailable,
        }
    }
}

impl<D: Device> Backend for DispatchBackend<D> {
    type InnerBackend = Self;
    // `host_format_display`/`host_format_debug` use `HostInterop`'s default,
    // which reads real values back through `float_to_vec1`/`int_to_vec1`.
}

impl<D: Device> incin_core::backend_authoring::HostReadback for DispatchBackend<D> {
    fn float_to_vec1<K: DType>(t: &DispatchStorage) -> Result<Vec<f64>> {
        match t {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                <crate::cpu::CpuBackendImpl<Cpu> as incin_core::backend_authoring::HostReadback>::float_to_vec1::<K>(value)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                <crate::wgpu::WgpuBackendImpl<Wgpu> as incin_core::backend_authoring::HostReadback>::float_to_vec1::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                <crate::cuda::CudaBackendImpl<Cuda> as incin_core::backend_authoring::HostReadback>::float_to_vec1::<K>(value)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                <crate::metal::MetalBackendImpl<Metal> as incin_core::backend_authoring::HostReadback>::float_to_vec1::<K>(value)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }

    fn int_to_vec1<K: DType>(t: &DispatchStorage) -> Result<Vec<i64>> {
        match t {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                <crate::cpu::CpuBackendImpl<Cpu> as incin_core::backend_authoring::HostReadback>::int_to_vec1::<K>(value)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                <crate::wgpu::WgpuBackendImpl<Wgpu> as incin_core::backend_authoring::HostReadback>::int_to_vec1::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                <crate::cuda::CudaBackendImpl<Cuda> as incin_core::backend_authoring::HostReadback>::int_to_vec1::<K>(value)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                <crate::metal::MetalBackendImpl<Metal> as incin_core::backend_authoring::HostReadback>::int_to_vec1::<K>(value)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }
}

impl<D: Device> incin_core::backend_authoring::HostInterop for DispatchBackend<D> {
    fn to_bytes<K: DType>(storage: &Self::Storage<K>) -> Result<Vec<u8>> {
        match storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => crate::cpu::CpuBackendImpl::<Cpu>::to_bytes::<K>(value),
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::to_bytes::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<Cuda>::to_bytes::<K>(value)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                crate::metal::MetalBackendImpl::<Metal>::to_bytes::<K>(value)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        match device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                return crate::cpu::CpuBackendImpl::<Cpu>::from_bytes::<K>(
                    bytes, shape, dtype, device,
                )
                .map(DispatchStorage::Cpu);
                #[cfg(not(feature = "cpu"))]
                Err(unavailable(DeviceKind::Cpu))
            }
            DeviceKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                return crate::wgpu::WgpuBackendImpl::<Wgpu>::from_bytes::<K>(
                    bytes, shape, dtype, device,
                )
                .map(DispatchStorage::Wgpu);
                #[cfg(not(feature = "wgpu"))]
                Err(unavailable(DeviceKind::Wgpu))
            }
            DeviceKind::Cuda => {
                #[cfg(feature = "cuda")]
                return crate::cuda::CudaBackendImpl::<Cuda>::from_bytes::<K>(
                    bytes, shape, dtype, device,
                )
                .map(DispatchStorage::Cuda);
                #[cfg(not(feature = "cuda"))]
                Err(unavailable(DeviceKind::Cuda))
            }
            DeviceKind::Metal => {
                #[cfg(feature = "metal")]
                return crate::metal::MetalBackendImpl::<Metal>::from_bytes::<K>(
                    bytes, shape, dtype, device,
                )
                .map(DispatchStorage::Metal);
                #[cfg(not(feature = "metal"))]
                Err(unavailable(DeviceKind::Metal))
            }
            other => Err(unavailable(other)),
        }
    }
}

impl<D: Device> DispatchBackend<D> {
    pub(crate) fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(zeros_with_total, zeros, shape, dtype, device)
    }
    pub(crate) fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(ones_with_total, ones, shape, dtype, device)
    }
    pub(crate) fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(rand_with_total, rand, shape, dtype, device)
    }
    pub(crate) fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(randn_with_total, randn, shape, dtype, device)
    }
    pub(crate) fn full<K: DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        match device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    let total = crate::cpu::stride::checked_numel(shape)?;
                    crate::cpu::creation::full_with_total(total, val, shape, dtype, device)
                        .map(DispatchStorage::Cpu)
                }
                #[cfg(not(feature = "cpu"))]
                Err(unavailable(DeviceKind::Cpu))
            }
            DeviceKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    crate::wgpu::WgpuBackendImpl::<Wgpu>::full::<K>(val, shape, dtype, device)
                        .map(DispatchStorage::Wgpu)
                }
                #[cfg(not(feature = "wgpu"))]
                Err(unavailable(DeviceKind::Wgpu))
            }
            DeviceKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::CudaBackendImpl::<Cuda>::full::<K>(val, shape, dtype, device)
                        .map(DispatchStorage::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                Err(unavailable(DeviceKind::Cuda))
            }
            other => Err(unavailable(other)),
        }
    }
    pub(crate) fn arange<K: DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        match device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    let total = crate::cpu::stride::checked_numel(shape)?;
                    crate::cpu::creation::arange_with_total(
                        total, start, step, shape, dtype, device,
                    )
                    .map(DispatchStorage::Cpu)
                }
                #[cfg(not(feature = "cpu"))]
                Err(unavailable(DeviceKind::Cpu))
            }
            DeviceKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    crate::wgpu::WgpuBackendImpl::<Wgpu>::arange::<K>(
                        start, step, shape, dtype, device,
                    )
                    .map(DispatchStorage::Wgpu)
                }
                #[cfg(not(feature = "wgpu"))]
                Err(unavailable(DeviceKind::Wgpu))
            }
            DeviceKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::CudaBackendImpl::<Cuda>::arange::<K>(
                        start, step, shape, dtype, device,
                    )
                    .map(DispatchStorage::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                Err(unavailable(DeviceKind::Cuda))
            }
            other => Err(unavailable(other)),
        }
    }
    pub(crate) fn linspace<K: DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        match device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    let total = crate::cpu::stride::checked_numel(shape)?;
                    crate::cpu::creation::linspace_with_total(
                        total, start, end, shape, dtype, device,
                    )
                    .map(DispatchStorage::Cpu)
                }
                #[cfg(not(feature = "cpu"))]
                Err(unavailable(DeviceKind::Cpu))
            }
            DeviceKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    crate::wgpu::WgpuBackendImpl::<Wgpu>::linspace::<K>(
                        start, end, shape, dtype, device,
                    )
                    .map(DispatchStorage::Wgpu)
                }
                #[cfg(not(feature = "wgpu"))]
                Err(unavailable(DeviceKind::Wgpu))
            }
            DeviceKind::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::CudaBackendImpl::<Cuda>::linspace::<K>(
                        start, end, shape, dtype, device,
                    )
                    .map(DispatchStorage::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                Err(unavailable(DeviceKind::Cuda))
            }
            other => Err(unavailable(other)),
        }
    }
}
impl<D: Device> DispatchBackend<D> {
    pub(crate) fn add_scalar_float<K: DType>(
        t: &DispatchStorage,
        scalar: f64,
    ) -> Result<DispatchStorage> {
        dispatch_unary!(t, add_scalar_float, scalar)
    }
    pub(crate) fn mul_scalar_float<K: DType>(
        t: &DispatchStorage,
        scalar: f64,
    ) -> Result<DispatchStorage> {
        dispatch_unary!(t, mul_scalar_float, scalar)
    }
}

impl<D: Device> incin_core::backend_authoring::AutogradBackend for DispatchBackend<D> {
    type Grads = DispatchGrads;

    fn backward<K: DType>(storage: &Self::Storage<K>) -> Result<Self::Grads> {
        match storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => <crate::cpu::CpuBackendImpl<Cpu> as incin_core::backend_authoring::AutogradBackend>::backward::<K>(value).map(DispatchGrads::Cpu),
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => <crate::wgpu::WgpuBackendImpl<Wgpu> as incin_core::backend_authoring::AutogradBackend>::backward::<K>(value).map(DispatchGrads::Wgpu),
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => <crate::cuda::CudaBackendImpl<Cuda> as incin_core::backend_authoring::AutogradBackend>::backward::<K>(value).map(DispatchGrads::Cuda),
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => <crate::metal::MetalBackendImpl<Metal> as incin_core::backend_authoring::AutogradBackend>::backward::<K>(value).map(DispatchGrads::Metal),
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }

    #[allow(unreachable_patterns)]
    fn backward_with<K: DType>(
        loss: &Self::Storage<K>,
        seed: &Self::Storage<K>,
    ) -> Result<Self::Grads> {
        match (loss, seed) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(loss), DispatchStorage::Cpu(seed)) => <crate::cpu::CpuBackendImpl<Cpu> as incin_core::backend_authoring::AutogradBackend>::backward_with::<K>(loss, seed).map(DispatchGrads::Cpu),
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(loss), DispatchStorage::Wgpu(seed)) => <crate::wgpu::WgpuBackendImpl<Wgpu> as incin_core::backend_authoring::AutogradBackend>::backward_with::<K>(loss, seed).map(DispatchGrads::Wgpu),
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(loss), DispatchStorage::Cuda(seed)) => <crate::cuda::CudaBackendImpl<Cuda> as incin_core::backend_authoring::AutogradBackend>::backward_with::<K>(loss, seed).map(DispatchGrads::Cuda),
            #[cfg(feature = "metal")]
            (DispatchStorage::Metal(loss), DispatchStorage::Metal(seed)) => <crate::metal::MetalBackendImpl<Metal> as incin_core::backend_authoring::AutogradBackend>::backward_with::<K>(loss, seed).map(DispatchGrads::Metal),
            (DispatchStorage::Unavailable, _) | (_, DispatchStorage::Unavailable) => Err(unavailable(DeviceKind::Cpu)),
            _ => Err(Error::Backend(BackendError::InvalidInput { operation: OperationKind::Storage, reason: "backward seed and loss must use the same backend" })),
        }
    }

    fn get_grad<K: DType>(
        storage: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        match (storage, grads) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(value), DispatchGrads::Cpu(gs)) => <crate::cpu::CpuBackendImpl<Cpu> as incin_core::backend_authoring::AutogradBackend>::get_grad::<K>(value, gs).map(|value| value.map(DispatchStorage::Cpu)),
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(value), DispatchGrads::Wgpu(gs)) => <crate::wgpu::WgpuBackendImpl<Wgpu> as incin_core::backend_authoring::AutogradBackend>::get_grad::<K>(value, gs).map(|value| value.map(DispatchStorage::Wgpu)),
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(value), DispatchGrads::Cuda(gs)) => <crate::cuda::CudaBackendImpl<Cuda> as incin_core::backend_authoring::AutogradBackend>::get_grad::<K>(value, gs).map(|value| value.map(DispatchStorage::Cuda)),
            #[cfg(feature = "metal")]
            (DispatchStorage::Metal(value), DispatchGrads::Metal(gs)) => <crate::metal::MetalBackendImpl<Metal> as incin_core::backend_authoring::AutogradBackend>::get_grad::<K>(value, gs).map(|value| value.map(DispatchStorage::Metal)),
            _ => Err(Error::DeviceMismatch { left: DeviceId::cpu(), right: DeviceId::cpu() }),
        }
    }

    fn set_grad<K: DType>(
        storage: &Self::Storage<K>,
        grads: &mut Self::Grads,
        value: Self::Storage<K>,
    ) -> Result<()> {
        match (storage, grads, value) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(target), DispatchGrads::Cpu(gs), DispatchStorage::Cpu(value)) => <crate::cpu::CpuBackendImpl<Cpu> as incin_core::backend_authoring::AutogradBackend>::set_grad::<K>(target, gs, value),
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(target), DispatchGrads::Wgpu(gs), DispatchStorage::Wgpu(value)) => <crate::wgpu::WgpuBackendImpl<Wgpu> as incin_core::backend_authoring::AutogradBackend>::set_grad::<K>(target, gs, value),
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(target), DispatchGrads::Cuda(gs), DispatchStorage::Cuda(value)) => <crate::cuda::CudaBackendImpl<Cuda> as incin_core::backend_authoring::AutogradBackend>::set_grad::<K>(target, gs, value),
            #[cfg(feature = "metal")]
            (DispatchStorage::Metal(target), DispatchGrads::Metal(gs), DispatchStorage::Metal(value)) => <crate::metal::MetalBackendImpl<Metal> as incin_core::backend_authoring::AutogradBackend>::set_grad::<K>(target, gs, value),
            _ => Err(Error::DeviceMismatch { left: DeviceId::cpu(), right: DeviceId::cpu() }),
        }
    }
}

impl<D: Device> VariableBackend for DispatchBackend<D> {
    type Var<K: DType> = DispatchVar;

    fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> Result<Self::Storage<K>> {
        match var {
            #[cfg(feature = "cpu")]
            DispatchVar::Cpu(value) => crate::cpu::CpuBackendImpl::<Cpu>::var_as_tensor::<K>(value)
                .map(DispatchStorage::Cpu),
            #[cfg(feature = "wgpu")]
            DispatchVar::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::var_as_tensor::<K>(value)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            DispatchVar::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<Cuda>::var_as_tensor::<K>(value)
                    .map(DispatchStorage::Cuda)
            }
            #[cfg(feature = "metal")]
            DispatchVar::Metal(value) => {
                crate::metal::MetalBackendImpl::<Metal>::var_as_tensor::<K>(value)
                    .map(DispatchStorage::Metal)
            }
            DispatchVar::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }
    fn var_from_tensor<K: DType>(storage: &Self::Storage<K>) -> Result<Self::Var<K>> {
        match storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<Cpu>::var_from_tensor::<K>(value).map(DispatchVar::Cpu)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::var_from_tensor::<K>(value)
                    .map(DispatchVar::Wgpu)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<Cuda>::var_from_tensor::<K>(value)
                    .map(DispatchVar::Cuda)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                crate::metal::MetalBackendImpl::<Metal>::var_from_tensor::<K>(value)
                    .map(DispatchVar::Metal)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }
    fn assign_var<K: DType>(var: &mut Self::Var<K>, storage: &Self::Storage<K>) -> Result<()> {
        match (var, storage) {
            #[cfg(feature = "cpu")]
            (DispatchVar::Cpu(var), DispatchStorage::Cpu(value)) => {
                crate::cpu::CpuBackendImpl::<Cpu>::assign_var::<K>(var, value)
            }
            #[cfg(feature = "wgpu")]
            (DispatchVar::Wgpu(var), DispatchStorage::Wgpu(value)) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::assign_var::<K>(var, value)
            }
            #[cfg(feature = "cuda")]
            (DispatchVar::Cuda(var), DispatchStorage::Cuda(value)) => {
                crate::cuda::CudaBackendImpl::<Cuda>::assign_var::<K>(var, value)
            }
            #[cfg(feature = "metal")]
            (DispatchVar::Metal(var), DispatchStorage::Metal(value)) => {
                crate::metal::MetalBackendImpl::<Metal>::assign_var::<K>(var, value)
            }
            _ => Err(Error::UnsupportedBackendOperation {
                op: "assign_var_cross_device",
                backend: "DispatchBackend",
            }),
        }
    }
}
