//! Runtime backend selection used by `IncinBackend<_, Dyn>`.

use incin_core::backend_authoring::*;
use incin_core::__backend_compat::legacy::*;
use crate::legacy::LossOps;
use incin_core::prelude::{
    BackendError, DType, DTypeDescriptor, Device, DeviceId, DeviceKind, Dyn, Error, FloatDType,
    OperationKind, QuantDType, Result, ShapeBuf, StorageBackend, SupportsDType, bf16, f16,
};
use alloc::{string::String, vec::Vec};
use crate::legacy::OptimizerOps;

#[cfg(feature = "cpu")]
use incin_core::prelude::Cpu;
#[cfg(feature = "cuda")]
use incin_core::prelude::Cuda;
#[cfg(feature = "wgpu")]
use incin_core::prelude::Wgpu;
#[cfg(feature = "metal")]
use incin_core::prelude::Metal;

use incin_core::exec::{PrecisionCapabilities, PrecisionRequest, ResolvedPrecision};

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

impl<D: Device> PrecisionCapabilities for DispatchBackend<D> {
    fn native_precision(&self, request: &PrecisionRequest) -> Result<ResolvedPrecision> {
        #[cfg(feature = "cpu")]
        {
            crate::cpu::CpuBackendImpl::<Cpu>::new().native_precision(request)
        }
        #[cfg(not(feature = "cpu"))]
        {
            Err(Error::BackendUnavailable {
                backend: "DispatchBackend",
            })
        }
    }
}

impl<D: Device> incin_core::exec::Capabilities for DispatchBackend<D> {
    fn support(&self, query: &incin_core::exec::CapabilityQuery) -> incin_core::exec::SupportLevel {
        #[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "metal"))]
        {
            // Delegate to whichever concrete backend is selected at runtime.
            // The detailed per-operation answer is filled in by dispatch_executor.rs;
            // here we provide the baseline so the trait is satisfied even in
            // configurations where no single concrete backend is statically known.
            #[cfg(feature = "cpu")]
            {
                crate::cpu::CpuBackendImpl::<Cpu>::default().support(query)
            }
            #[cfg(all(
                not(feature = "cpu"),
                any(feature = "wgpu", feature = "cuda", feature = "metal")
            ))]
            let _ = query;
            #[cfg(all(
                not(feature = "cpu"),
                any(feature = "wgpu", feature = "cuda", feature = "metal")
            ))]
            incin_core::exec::SupportLevel::Native
        }
        #[cfg(not(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "metal")))]
        {
            let incin_core::exec::OperationIdentity::Builtin(operation) = &query.operation else {
                return incin_core::exec::SupportLevel::Unsupported(
                    incin_core::exec::UnsupportedReason::CustomOperation {
                        operation: match &query.operation {
                            incin_core::exec::OperationIdentity::Custom(operation) => {
                                operation.clone()
                            }
                            incin_core::exec::OperationIdentity::Builtin(_) => unreachable!(),
                        },
                    },
                );
            };
            incin_core::exec::SupportLevel::Unsupported(
                incin_core::exec::UnsupportedReason::Operation {
                    operation: *operation,
                },
            )
        }
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

impl DispatchStorage {
    /// Checked physical metadata of whichever backend currently holds the data.
    #[must_use]
    pub fn metadata(&self) -> &incin_core::exec::TensorMeta {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(storage) => storage.metadata(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(storage) => storage.metadata(),
            #[cfg(feature = "cuda")]
            Self::Cuda(storage) => storage.metadata(),
            #[cfg(feature = "metal")]
            Self::Metal(storage) => storage.metadata(),
            // This variant exists only so the enum has a shape when no backend
            // feature is enabled, and it owns no allocation to describe.
            Self::Unavailable => &incin_core::exec::TensorMeta::UNALLOCATED,
        }
    }
}

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

fn storage_device(storage: &DispatchStorage) -> DeviceId {
    match storage {
        #[cfg(feature = "cpu")]
        DispatchStorage::Cpu(_) => DeviceId::cpu(),
        #[cfg(feature = "wgpu")]
        DispatchStorage::Wgpu(_) => DeviceId::wgpu(0),
        #[cfg(feature = "cuda")]
        DispatchStorage::Cuda(value) => DeviceId::cuda(value.buffer.device_id),
        #[cfg(feature = "metal")]
        DispatchStorage::Metal(value) => value.device(),
        DispatchStorage::Unavailable => DeviceId::cpu(),
    }
}

/// Operand extractors for operations that take more than two tensors.
///
/// An operation routed by [`DispatchBackend`] executes on exactly one concrete
/// backend, so every operand has to be sitting on that same backend already.
/// These recover the concrete storage and report [`Error::DeviceMismatch`]
/// against the operand that decided the route, rather than silently reading a
/// buffer that lives on another device.
macro_rules! operand_accessors {
    ($($fname:ident, $optname:ident, $variant:ident, $concrete:ty, $feature:literal;)*) => {
        $(
            #[cfg(feature = $feature)]
            fn $fname<'a>(
                routed: &DispatchStorage,
                operand: &'a DispatchStorage,
            ) -> Result<&'a $concrete> {
                match operand {
                    DispatchStorage::$variant(value) => Ok(value),
                    other => Err(Error::DeviceMismatch {
                        left: storage_device(routed),
                        right: storage_device(other),
                    }),
                }
            }

            #[cfg(feature = $feature)]
            fn $optname<'a>(
                routed: &DispatchStorage,
                operand: Option<&'a DispatchStorage>,
            ) -> Result<Option<&'a $concrete>> {
                operand.map(|value| $fname(routed, value)).transpose()
            }
        )*
    };
}

operand_accessors! {
    cpu_operand, cpu_optional, Cpu, crate::cpu::CpuStorage, "cpu";
    wgpu_operand, wgpu_optional, Wgpu, crate::wgpu::storage::WgpuStorage, "wgpu";
    cuda_operand, cuda_optional, Cuda, crate::cuda::storage::CudaStorage, "cuda";
    metal_operand, metal_optional, Metal, crate::metal::storage::MetalStorage, "metal";
}

/// Routes a multi-operand operation to the backend holding its first operand.
///
/// `req` and `opt` name the remaining required and optional tensor operands;
/// each is checked against the routed backend, so a mixed-device call fails
/// with a device mismatch instead of reaching a kernel.
macro_rules! dispatch_same_device {
    (
        $primary:expr, $method:ident::<$($generic:ty),*>,
        req = [$($req:expr),*], opt = [$($opt:expr),*], args = [$($arg:expr),*]
    ) => {{
        let routed = $primary;
        match routed {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => crate::cpu::CpuBackendImpl::<Cpu>::$method::<$($generic),*>(
                value,
                $(cpu_operand(routed, $req)?,)*
                $(cpu_optional(routed, $opt)?,)*
                $($arg),*
            )
            .map(DispatchStorage::Cpu),
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => crate::wgpu::WgpuBackendImpl::<Wgpu>::$method::<$($generic),*>(
                value,
                $(wgpu_operand(routed, $req)?,)*
                $(wgpu_optional(routed, $opt)?,)*
                $($arg),*
            )
            .map(DispatchStorage::Wgpu),
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => crate::cuda::CudaBackendImpl::<Cuda>::$method::<$($generic),*>(
                value,
                $(cuda_operand(routed, $req)?,)*
                $(cuda_optional(routed, $opt)?,)*
                $($arg),*
            )
            .map(DispatchStorage::Cuda),
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => crate::metal::MetalBackendImpl::<Metal>::$method::<$($generic),*>(
                value,
                $(metal_operand(routed, $req)?,)*
                $(metal_optional(routed, $opt)?,)*
                $($arg),*
            )
            .map(DispatchStorage::Metal),
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }};
}

macro_rules! dispatch_unary {
    ($storage:expr, $method:ident $(, $arg:expr)*) => {
        match $storage {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => crate::cpu::CpuBackendImpl::<Cpu>::$method::<K>(value $(, $arg)*)
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

/// Routes an operation taking a slice of operands to the backend holding the
/// first one, checking every remaining operand against that route.
///
/// `stack` and `concat` are the only `TensorOps` members with this shape. An
/// empty slice names no device to route on, so it is reported here with the
/// same message the concrete backends use rather than reaching a kernel.
macro_rules! dispatch_slice {
    ($operands:expr, $method:ident, $dim:expr) => {{
        let operands: &[&DispatchStorage] = $operands;
        let Some(routed) = operands.first().copied() else {
            return Err(Error::ShapeMismatch {
                op: stringify!($method),
                expected: Vec::new(),
                got: Vec::new(),
                msg: String::from(concat!(
                    stringify!($method),
                    " requires at least one input tensor"
                )),
            });
        };
        match routed {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(_) => {
                let concrete = operands
                    .iter()
                    .map(|operand| cpu_operand(routed, operand))
                    .collect::<Result<Vec<_>>>()?;
                crate::cpu::CpuBackendImpl::<Cpu>::$method::<K>(&concrete, $dim)
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(_) => {
                let concrete = operands
                    .iter()
                    .map(|operand| wgpu_operand(routed, operand))
                    .collect::<Result<Vec<_>>>()?;
                crate::wgpu::WgpuBackendImpl::<Wgpu>::$method::<K>(&concrete, $dim)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(_) => {
                let concrete = operands
                    .iter()
                    .map(|operand| cuda_operand(routed, operand))
                    .collect::<Result<Vec<_>>>()?;
                crate::cuda::CudaBackendImpl::<Cuda>::$method::<K>(&concrete, $dim)
                    .map(DispatchStorage::Cuda)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(_) => {
                let concrete = operands
                    .iter()
                    .map(|operand| metal_operand(routed, operand))
                    .collect::<Result<Vec<_>>>()?;
                crate::metal::MetalBackendImpl::<Metal>::$method::<K>(&concrete, $dim)
                    .map(DispatchStorage::Metal)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }};
}

macro_rules! dispatch_binary {
    ($lhs:expr, $rhs:expr, $method:ident) => {
        match ($lhs, $rhs) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(lhs), DispatchStorage::Cpu(rhs)) => {
                crate::cpu::CpuBackendImpl::<Cpu>::$method::<K>(lhs, rhs).map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(lhs), DispatchStorage::Wgpu(rhs)) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::$method::<K>(lhs, rhs)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(lhs), DispatchStorage::Cuda(rhs)) => {
                crate::cuda::CudaBackendImpl::<Cuda>::$method::<K>(lhs, rhs)
                    .map(DispatchStorage::Cuda)
            }
            #[cfg(feature = "metal")]
            (DispatchStorage::Metal(lhs), DispatchStorage::Metal(rhs)) => {
                crate::metal::MetalBackendImpl::<Metal>::$method::<K>(lhs, rhs)
                    .map(DispatchStorage::Metal)
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
                    return crate::cpu::CpuBackendImpl::<Cpu>::$method::<K>(
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

macro_rules! create_var_dispatch {
    ($method:ident, $shape:expr, $dtype:expr, $device:expr) => {
        match $device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    return crate::cpu::CpuBackendImpl::<Cpu>::$method::<K>(
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
                    return crate::wgpu::WgpuBackendImpl::<Wgpu>::$method::<K>(
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
                    return crate::cuda::CudaBackendImpl::<Cuda>::$method::<K>(
                        $shape, $dtype, $device,
                    )
                    .map(DispatchVar::Cuda);
                }
                #[cfg(not(feature = "cuda"))]
                Err(unavailable(DeviceKind::Cuda))
            }
            other => Err(unavailable(other)),
        }
    };
}

macro_rules! variable_executors {
    ($(($operation:ident, $method:ident)),* $(,)?) => {$ (
        impl<D: Device> Execute<op::$operation> for DispatchBackend<D> {
            type Output = DispatchVar;

            fn execute(
                &self,
                request: ExecutionRequest<'_, op::$operation, Self>,
            ) -> core::result::Result<DispatchVar, BackendError> {
                let attributes = request.operation.descriptor().attributes();
                create_var_execute_dispatch!($method, &attributes.shape, attributes.dtype, &attributes.device)
                    .map_err(|error| BackendError::Execution {
                        operation: OperationKind::$operation,
                        message: alloc::format!("{error}").into(),
                    })
            }
        }
    )*};
}

macro_rules! create_var_execute_dispatch {
    ($method:ident, $shape:expr, $dtype:expr, $device:expr) => {
        match $device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    crate::cpu::CpuBackendImpl::<Cpu>::$method::<Dyn>($shape, $dtype, $device)
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
    (VariableZeros, var_zeros),
    (VariableOnes, var_ones),
    (VariableUniformRandom, var_rand),
    (VariableNormalRandom, var_randn),
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
                <Self as FloatOps<Self>>::$method::<Dyn>(input, value).map_err(|error| {
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

impl<D: Device> CreationOps<Self> for DispatchBackend<D> {
    fn zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(zeros, shape, dtype, device)
    }
    fn ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(ones, shape, dtype, device)
    }
    fn rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(rand, shape, dtype, device)
    }
    fn randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        create_dispatch!(randn, shape, dtype, device)
    }
    fn full<K: DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchStorage> {
        match device.kind() {
            DeviceKind::Cpu => {
                #[cfg(feature = "cpu")]
                {
                    crate::cpu::CpuBackendImpl::<Cpu>::full::<K>(val, shape, dtype, device)
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
    fn arange<K: DType>(
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
                    crate::cpu::CpuBackendImpl::<Cpu>::arange::<K>(
                        start, step, shape, dtype, device,
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
    fn linspace<K: DType>(
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
                    crate::cpu::CpuBackendImpl::<Cpu>::linspace::<K>(
                        start, end, shape, dtype, device,
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
    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchVar> {
        create_var_dispatch!(var_zeros, shape, dtype, device)
    }
    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchVar> {
        create_var_dispatch!(var_ones, shape, dtype, device)
    }
    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchVar> {
        create_var_dispatch!(var_rand, shape, dtype, device)
    }
    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<DispatchVar> {
        create_var_dispatch!(var_randn, shape, dtype, device)
    }
}

impl<D: Device> TensorOps<Self> for DispatchBackend<D> {
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
    fn stack<K: DType>(t: &[&DispatchStorage], dim: usize) -> Result<DispatchStorage> {
        dispatch_slice!(t, stack, dim)
    }
    fn concat<K: DType>(t: &[&DispatchStorage], dim: usize) -> Result<DispatchStorage> {
        dispatch_slice!(t, concat, dim)
    }
    fn slice<K: DType>(t: &DispatchStorage, ranges: &[(usize, usize)]) -> Result<DispatchStorage> {
        dispatch_unary!(t, slice, ranges)
    }
    fn flatten<K: DType>(t: &DispatchStorage, start: usize, end: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, flatten, start, end)
    }
    fn where_cond<K: DType>(
        mask: &DispatchStorage,
        on_true: &DispatchStorage,
        on_false: &DispatchStorage,
    ) -> Result<DispatchStorage> {
        match (mask, on_true, on_false) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(m), DispatchStorage::Cpu(t), DispatchStorage::Cpu(f)) => {
                crate::cpu::CpuBackendImpl::<Cpu>::where_cond::<K>(m, t, f)
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(m), DispatchStorage::Wgpu(t), DispatchStorage::Wgpu(f)) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::where_cond::<K>(m, t, f)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(m), DispatchStorage::Cuda(t), DispatchStorage::Cuda(f)) => {
                crate::cuda::CudaBackendImpl::<Cuda>::where_cond::<K>(m, t, f)
                    .map(DispatchStorage::Cuda)
            }
            _ => Err(Error::DeviceMismatch {
                left: storage_device(on_true),
                right: storage_device(on_false),
            }),
        }
    }
    fn gather<K: DType, KInt: DType>(
        t: &DispatchStorage,
        dim: usize,
        index: &DispatchStorage,
    ) -> Result<DispatchStorage> {
        match (t, index) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(t), DispatchStorage::Cpu(idx)) => {
                crate::cpu::CpuBackendImpl::<Cpu>::gather::<K, KInt>(t, dim, idx)
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(t), DispatchStorage::Wgpu(idx)) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::gather::<K, KInt>(t, dim, idx)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(t), DispatchStorage::Cuda(idx)) => {
                crate::cuda::CudaBackendImpl::<Cuda>::gather::<K, KInt>(t, dim, idx)
                    .map(DispatchStorage::Cuda)
            }
            _ => Err(Error::DeviceMismatch {
                left: storage_device(t),
                right: storage_device(index),
            }),
        }
    }
    fn scatter<K: DType, KInt: DType>(
        t: &DispatchStorage,
        dim: usize,
        index: &DispatchStorage,
        src: &DispatchStorage,
    ) -> Result<DispatchStorage> {
        match (t, index, src) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(t), DispatchStorage::Cpu(idx), DispatchStorage::Cpu(s)) => {
                crate::cpu::CpuBackendImpl::<Cpu>::scatter::<K, KInt>(t, dim, idx, s)
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(t), DispatchStorage::Wgpu(idx), DispatchStorage::Wgpu(s)) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::scatter::<K, KInt>(t, dim, idx, s)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(t), DispatchStorage::Cuda(idx), DispatchStorage::Cuda(s)) => {
                crate::cuda::CudaBackendImpl::<Cuda>::scatter::<K, KInt>(t, dim, idx, s)
                    .map(DispatchStorage::Cuda)
            }
            _ => Err(Error::DeviceMismatch {
                left: storage_device(t),
                right: storage_device(src),
            }),
        }
    }
    fn index_select<K: DType, KInt: DType>(
        t: &DispatchStorage,
        dim: usize,
        index: &DispatchStorage,
    ) -> Result<DispatchStorage> {
        match (t, index) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(t), DispatchStorage::Cpu(idx)) => {
                crate::cpu::CpuBackendImpl::<Cpu>::index_select::<K, KInt>(t, dim, idx)
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(t), DispatchStorage::Wgpu(idx)) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::index_select::<K, KInt>(t, dim, idx)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(t), DispatchStorage::Cuda(idx)) => {
                crate::cuda::CudaBackendImpl::<Cuda>::index_select::<K, KInt>(t, dim, idx)
                    .map(DispatchStorage::Cuda)
            }
            _ => Err(Error::DeviceMismatch {
                left: storage_device(t),
                right: storage_device(index),
            }),
        }
    }
    fn masked_fill<K: DType>(
        t: &DispatchStorage,
        mask: &DispatchStorage,
        value: f64,
    ) -> Result<DispatchStorage> {
        match (t, mask) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(t), DispatchStorage::Cpu(m)) => {
                crate::cpu::CpuBackendImpl::<Cpu>::masked_fill::<K>(t, m, value)
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(t), DispatchStorage::Wgpu(m)) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::masked_fill::<K>(t, m, value)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(t), DispatchStorage::Cuda(m)) => {
                crate::cuda::CudaBackendImpl::<Cuda>::masked_fill::<K>(t, m, value)
                    .map(DispatchStorage::Cuda)
            }
            _ => Err(Error::DeviceMismatch {
                left: storage_device(t),
                right: storage_device(mask),
            }),
        }
    }
    fn unsqueeze<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, unsqueeze, dim)
    }
    fn repeat<K: DType>(t: &DispatchStorage, repeats: &[usize]) -> Result<DispatchStorage> {
        dispatch_unary!(t, repeat, repeats)
    }
    fn pad<K: DType>(
        t: &DispatchStorage,
        padding: &[(usize, usize)],
        val: f64,
    ) -> Result<DispatchStorage> {
        dispatch_unary!(t, pad, padding, val)
    }
    fn triu<K: DType>(t: &DispatchStorage, k: i64) -> Result<DispatchStorage> {
        dispatch_unary!(t, triu, k)
    }
    fn tril<K: DType>(t: &DispatchStorage, k: i64) -> Result<DispatchStorage> {
        dispatch_unary!(t, tril, k)
    }
    fn diag<K: DType>(t: &DispatchStorage, k: i64) -> Result<DispatchStorage> {
        dispatch_unary!(t, diag, k)
    }
    fn broadcast_left<K: DType>(t: &DispatchStorage, shape: &[usize]) -> Result<DispatchStorage> {
        dispatch_unary!(t, broadcast_left, shape)
    }
    fn float_to_scalar<K: DType>(t: &DispatchStorage) -> Result<f64> {
        match t {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<Cpu>::float_to_scalar::<K>(value)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::float_to_scalar::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<Cuda>::float_to_scalar::<K>(value)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                crate::metal::MetalBackendImpl::<Metal>::float_to_scalar::<K>(value)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }
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
                crate::cuda::CudaBackendImpl::<Cuda>::float_to_vec1::<K>(value)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                crate::metal::MetalBackendImpl::<Metal>::float_to_vec1::<K>(value)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }
    fn int_to_scalar<K: DType>(t: &DispatchStorage) -> Result<i64> {
        match t {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<Cpu>::int_to_scalar::<K>(value)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::int_to_scalar::<K>(value)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<Cuda>::int_to_scalar::<K>(value)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                crate::metal::MetalBackendImpl::<Metal>::int_to_scalar::<K>(value)
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
                crate::cuda::CudaBackendImpl::<Cuda>::int_to_vec1::<K>(value)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                crate::metal::MetalBackendImpl::<Metal>::int_to_vec1::<K>(value)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }
    fn tensor_to_dtype<K: DType, K2: DType>(
        t: &DispatchStorage,
        dtype: DTypeDescriptor,
    ) -> Result<DispatchStorage> {
        match t {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<Cpu>::tensor_to_dtype::<K, K2>(value, dtype)
                    .map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::tensor_to_dtype::<K, K2>(value, dtype)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<Cuda>::tensor_to_dtype::<K, K2>(value, dtype)
                    .map(DispatchStorage::Cuda)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                crate::metal::MetalBackendImpl::<Metal>::tensor_to_dtype::<K, K2>(value, dtype)
                    .map(DispatchStorage::Metal)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }

    fn cmp_eq<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, cmp_eq)
    }
    fn cmp_ne<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, cmp_ne)
    }
    fn cmp_lt<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, cmp_lt)
    }
    fn cmp_le<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, cmp_le)
    }
    fn cmp_gt<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, cmp_gt)
    }
    fn cmp_ge<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, cmp_ge)
    }

    fn logical_and(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        match (lhs, rhs) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(lhs), DispatchStorage::Cpu(rhs)) => {
                crate::cpu::CpuBackendImpl::<Cpu>::logical_and(lhs, rhs).map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(lhs), DispatchStorage::Wgpu(rhs)) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::logical_and(lhs, rhs)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(lhs), DispatchStorage::Cuda(rhs)) => {
                crate::cuda::CudaBackendImpl::<Cuda>::logical_and(lhs, rhs)
                    .map(DispatchStorage::Cuda)
            }
            #[cfg(feature = "metal")]
            (DispatchStorage::Metal(lhs), DispatchStorage::Metal(rhs)) => {
                crate::metal::MetalBackendImpl::<Metal>::logical_and(lhs, rhs)
                    .map(DispatchStorage::Metal)
            }
            _ => Err(Error::Msg("mismatched backends in logical_and".into())),
        }
    }
    fn logical_or(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        match (lhs, rhs) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(lhs), DispatchStorage::Cpu(rhs)) => {
                crate::cpu::CpuBackendImpl::<Cpu>::logical_or(lhs, rhs).map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            (DispatchStorage::Wgpu(lhs), DispatchStorage::Wgpu(rhs)) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::logical_or(lhs, rhs)
                    .map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            (DispatchStorage::Cuda(lhs), DispatchStorage::Cuda(rhs)) => {
                crate::cuda::CudaBackendImpl::<Cuda>::logical_or(lhs, rhs)
                    .map(DispatchStorage::Cuda)
            }
            #[cfg(feature = "metal")]
            (DispatchStorage::Metal(lhs), DispatchStorage::Metal(rhs)) => {
                crate::metal::MetalBackendImpl::<Metal>::logical_or(lhs, rhs)
                    .map(DispatchStorage::Metal)
            }
            _ => Err(Error::Msg("mismatched backends in logical_or".into())),
        }
    }
    fn logical_not(t: &DispatchStorage) -> Result<DispatchStorage> {
        match t {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                crate::cpu::CpuBackendImpl::<Cpu>::logical_not(value).map(DispatchStorage::Cpu)
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                crate::wgpu::WgpuBackendImpl::<Wgpu>::logical_not(value).map(DispatchStorage::Wgpu)
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                crate::cuda::CudaBackendImpl::<Cuda>::logical_not(value).map(DispatchStorage::Cuda)
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                crate::metal::MetalBackendImpl::<Metal>::logical_not(value)
                    .map(DispatchStorage::Metal)
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }

    fn sub_scalar<K: DType>(t: &DispatchStorage, val: f64) -> Result<DispatchStorage> {
        dispatch_unary!(t, sub_scalar, val)
    }
    fn div_scalar<K: DType>(t: &DispatchStorage, val: f64) -> Result<DispatchStorage> {
        dispatch_unary!(t, div_scalar, val)
    }

    fn maximum<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, maximum)
    }
    fn minimum<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, minimum)
    }
    fn abs_diff<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, abs_diff)
    }
    fn lerp<K: DType>(
        start: &DispatchStorage,
        end: &DispatchStorage,
        _weight: f64,
    ) -> Result<DispatchStorage> {
        match (start, end) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(s), DispatchStorage::Cpu(e)) => {
                #[cfg(feature = "cpu")]
                {
                    crate::cpu::CpuBackendImpl::<Cpu>::lerp::<K>(s, e, _weight)
                        .map(DispatchStorage::Cpu)
                }
            }
            _ => Err(Error::Msg("mismatched backends in lerp".into())),
        }
    }

    fn addmm<K: DType>(
        mat: &DispatchStorage,
        mat1: &DispatchStorage,
        mat2: &DispatchStorage,
        _beta: f64,
        _alpha: f64,
    ) -> Result<DispatchStorage> {
        match (mat, mat1, mat2) {
            #[cfg(feature = "cpu")]
            (DispatchStorage::Cpu(m), DispatchStorage::Cpu(m1), DispatchStorage::Cpu(m2)) => {
                #[cfg(feature = "cpu")]
                {
                    crate::cpu::CpuBackendImpl::<Cpu>::addmm::<K>(m, m1, m2, _beta, _alpha)
                        .map(DispatchStorage::Cpu)
                }
            }
            _ => Err(Error::Msg("mismatched backends in addmm".into())),
        }
    }
    fn bmm<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, bmm)
    }
    fn scaled_dot_product_attention<K: DType>(
        q: &DispatchStorage,
        k: &DispatchStorage,
        v: &DispatchStorage,
        _mask: Option<&DispatchStorage>,
        _scale: Option<f64>,
    ) -> Result<DispatchStorage> {
        match (q, k, v) {
            #[cfg(feature = "cpu")]
            (
                DispatchStorage::Cpu(q_cpu),
                DispatchStorage::Cpu(k_cpu),
                DispatchStorage::Cpu(v_cpu),
            ) => {
                #[cfg(feature = "cpu")]
                {
                    let m_cpu = match _mask {
                        Some(DispatchStorage::Cpu(m)) => Some(m),
                        None => None,
                        _ => return Err(Error::Msg("mismatched mask backend".into())),
                    };
                    crate::cpu::CpuBackendImpl::<Cpu>::scaled_dot_product_attention::<K>(
                        q_cpu, k_cpu, v_cpu, m_cpu, _scale,
                    )
                    .map(DispatchStorage::Cpu)
                }
            }
            _ => Err(Error::Msg("mismatched backends in SDPA".into())),
        }
    }

    fn unfold<K: DType>(
        t: &DispatchStorage,
        dim: usize,
        size: usize,
        step: usize,
    ) -> Result<DispatchStorage> {
        dispatch_unary!(t, unfold, dim, size, step)
    }
    fn pixel_shuffle<K: DType>(
        t: &DispatchStorage,
        upscale_factor: usize,
    ) -> Result<DispatchStorage> {
        dispatch_unary!(t, pixel_shuffle, upscale_factor)
    }
    fn group_norm<K: DType>(
        t: &DispatchStorage,
        groups: usize,
        eps: f64,
    ) -> Result<DispatchStorage> {
        dispatch_unary!(t, group_norm, groups, eps)
    }
    fn instance_norm<K: DType>(t: &DispatchStorage, eps: f64) -> Result<DispatchStorage> {
        dispatch_unary!(t, instance_norm, eps)
    }
}
impl<D: Device> NumericOps<Self> for DispatchBackend<D> {
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
impl<D: Device> FloatOps<Self> for DispatchBackend<D> {
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
    fn powf<K: DType>(t: &DispatchStorage, exponent: f64) -> Result<DispatchStorage> {
        dispatch_unary!(t, powf, exponent)
    }
    fn clamp<K: DType>(t: &DispatchStorage, min: f64, max: f64) -> Result<DispatchStorage> {
        dispatch_unary!(t, clamp, min, max)
    }
    fn sign<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, sign)
    }
    fn floor<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, floor)
    }
    fn ceil<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, ceil)
    }
    fn round<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, round)
    }
    fn log2<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, log2)
    }
    fn log10<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, log10)
    }
    fn sin<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, sin)
    }
    fn cos<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, cos)
    }
    fn tan<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, tan)
    }
    fn asin<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, asin)
    }
    fn acos<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, acos)
    }
    fn atan<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, atan)
    }
    fn atan2<K: DType>(y: &DispatchStorage, x: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(y, x, atan2)
    }
    fn sinh<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, sinh)
    }
    fn cosh<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, cosh)
    }
    fn asinh<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, asinh)
    }
    fn acosh<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, acosh)
    }
    fn atanh<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, atanh)
    }
    fn erf<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, erf)
    }
    fn rsqrt<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, rsqrt)
    }
    fn trunc<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, trunc)
    }
    fn frac<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, frac)
    }
    fn fmod<K: DType>(lhs: &DispatchStorage, rhs: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, fmod)
    }
    fn remainder<K: DType>(
        lhs: &DispatchStorage,
        rhs: &DispatchStorage,
    ) -> Result<DispatchStorage> {
        dispatch_binary!(lhs, rhs, remainder)
    }
}
impl<D: Device> ReductionOps<Self> for DispatchBackend<D> {
    fn argmax<K: DType, KInt: DType>(
        t: &DispatchStorage,
        dim: Option<usize>,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(t, argmax::<K, KInt>, req = [], opt = [], args = [dim])
    }
    fn argmin<K: DType, KInt: DType>(
        t: &DispatchStorage,
        dim: Option<usize>,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(t, argmin::<K, KInt>, req = [], opt = [], args = [dim])
    }
    fn argsort<K: DType, KInt: DType>(
        t: &DispatchStorage,
        dim: usize,
        descending: bool,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(
            t,
            argsort::<K, KInt>,
            req = [],
            opt = [],
            args = [dim, descending]
        )
    }
    // `topk` returns a value/index pair rather than one storage, so it does
    // not fit `dispatch_same_device!` and is routed by hand.
    fn topk<K: DType, KInt: DType>(
        t: &DispatchStorage,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(DispatchStorage, DispatchStorage)> {
        match t {
            #[cfg(feature = "cpu")]
            DispatchStorage::Cpu(value) => {
                let (values, indices) =
                    crate::cpu::CpuBackendImpl::<Cpu>::topk::<K, KInt>(value, k, dim, largest)?;
                Ok((DispatchStorage::Cpu(values), DispatchStorage::Cpu(indices)))
            }
            #[cfg(feature = "wgpu")]
            DispatchStorage::Wgpu(value) => {
                let (values, indices) =
                    crate::wgpu::WgpuBackendImpl::<Wgpu>::topk::<K, KInt>(value, k, dim, largest)?;
                Ok((
                    DispatchStorage::Wgpu(values),
                    DispatchStorage::Wgpu(indices),
                ))
            }
            #[cfg(feature = "cuda")]
            DispatchStorage::Cuda(value) => {
                let (values, indices) =
                    crate::cuda::CudaBackendImpl::<Cuda>::topk::<K, KInt>(value, k, dim, largest)?;
                Ok((
                    DispatchStorage::Cuda(values),
                    DispatchStorage::Cuda(indices),
                ))
            }
            #[cfg(feature = "metal")]
            DispatchStorage::Metal(value) => {
                let (values, indices) = crate::metal::MetalBackendImpl::<Metal>::topk::<K, KInt>(
                    value, k, dim, largest,
                )?;
                Ok((
                    DispatchStorage::Metal(values),
                    DispatchStorage::Metal(indices),
                ))
            }
            DispatchStorage::Unavailable => Err(unavailable(DeviceKind::Cpu)),
        }
    }
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
    fn prod_all<K: DType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_unary!(t, prod_all)
    }
    fn prod_dim<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, prod_dim, dim)
    }
    fn cumsum<K: DType>(t: &DispatchStorage, dim: usize) -> Result<DispatchStorage> {
        dispatch_unary!(t, cumsum, dim)
    }
}
impl<D: Device> QuantizedOps<Self> for DispatchBackend<D> {
    fn quantize<K: FloatDType, Q: QuantDType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_same_device!(t, quantize::<K, Q>, req = [], opt = [], args = [])
    }
    fn dequantize<Q: QuantDType, K: FloatDType>(t: &DispatchStorage) -> Result<DispatchStorage> {
        dispatch_same_device!(t, dequantize::<Q, K>, req = [], opt = [], args = [])
    }
    fn quantized_matmul<Q: QuantDType>(
        lhs: &DispatchStorage,
        rhs: &DispatchStorage,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(lhs, quantized_matmul::<Q>, req = [rhs], opt = [], args = [])
    }
}

// `OptimizerOps`'s methods are composed from `NumericOps`/`FloatOps` rather
// than defaulted to an unsupported error, so the routed backend supplies a
// working update rule and this impl has nothing to override.
impl<D: Device> OptimizerOps<Self> for DispatchBackend<D> {}

impl<D: Device> ModuleOps<Self> for DispatchBackend<D> {
    fn layer_norm<K: DType>(
        t: &DispatchStorage,
        weight: &DispatchStorage,
        bias: Option<&DispatchStorage>,
        eps: f32,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(
            t,
            layer_norm::<K>,
            req = [weight],
            opt = [bias],
            args = [eps]
        )
    }
    fn batch_norm<K: DType>(
        t: &DispatchStorage,
        w: Option<&DispatchStorage>,
        b: Option<&DispatchStorage>,
        rm: Option<&DispatchStorage>,
        rv: Option<&DispatchStorage>,
        e: f32,
        momentum: f64,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(
            t,
            batch_norm::<K>,
            req = [],
            opt = [w, b, rm, rv],
            args = [e, momentum]
        )
    }
    fn embedding<K: DType, KInt: DType>(
        t: &DispatchStorage,
        w: &DispatchStorage,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(t, embedding::<K, KInt>, req = [w], opt = [], args = [])
    }
    fn conv1d<K: DType>(
        t: &DispatchStorage,
        w: &DispatchStorage,
        b: Option<&DispatchStorage>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(
            t,
            conv1d::<K>,
            req = [w],
            opt = [b],
            args = [stride, padding, dilation, groups]
        )
    }
    fn conv2d<K: DType>(
        t: &DispatchStorage,
        w: &DispatchStorage,
        b: Option<&DispatchStorage>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(
            t,
            conv2d::<K>,
            req = [w],
            opt = [b],
            args = [stride, padding, dilation, groups]
        )
    }
    fn conv_transpose2d<K: DType>(
        t: &DispatchStorage,
        w: &DispatchStorage,
        b: Option<&DispatchStorage>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(
            t,
            conv_transpose2d::<K>,
            req = [w],
            opt = [b],
            args = [stride, padding, output_padding, dilation, groups]
        )
    }
    fn max_pool2d<K: DType>(
        t: &DispatchStorage,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(
            t,
            max_pool2d::<K>,
            req = [],
            opt = [],
            args = [kernel_size, stride, padding, dilation]
        )
    }
    fn avg_pool2d<K: DType>(
        t: &DispatchStorage,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(
            t,
            avg_pool2d::<K>,
            req = [],
            opt = [],
            args = [kernel_size, stride, padding]
        )
    }
    fn adaptive_avg_pool2d<K: DType>(
        t: &DispatchStorage,
        output_size: (usize, usize),
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(
            t,
            adaptive_avg_pool2d::<K>,
            req = [],
            opt = [],
            args = [output_size]
        )
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

    fn backward_with<K: DType>(loss: &Self::Storage<K>, seed: &Self::Storage<K>) -> Result<Self::Grads> {
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

    fn get_grad<K: DType>(storage: &Self::Storage<K>, grads: &Self::Grads) -> Result<Option<Self::Storage<K>>> {
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


impl<D: Device> LossOps<Self> for DispatchBackend<D> {
    // `mse_loss`, `l1_loss`, and `bce_with_logits_loss` are composed from the
    // routed backend's numeric and reduction ops. `cross_entropy_loss` is the
    // one method with no composed default, so it has to be routed explicitly.
    fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &DispatchStorage,
        target: &DispatchStorage,
        reduction: incin_core::tensor::reduction::Reduction,
    ) -> Result<DispatchStorage> {
        dispatch_same_device!(
            pred,
            cross_entropy_loss::<K, KInt>,
            req = [target],
            opt = [],
            args = [reduction]
        )
    }
}
