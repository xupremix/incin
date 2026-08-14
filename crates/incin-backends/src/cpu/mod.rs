//! # Incin Cpu
//!
//! `incin-cpu` is a pure-Rust, ownership-flavored, CPU-only implementation
//! of the `Backend` trait defined in `incin-core`. It provides its own
//! strided-view tensor storage (`storage`) and shape/stride math (`stride`),
//! independent of any external tensor compute library (Candle, ndarray, burn).
//!
//! Canonical CPU execution is implemented through the descriptor/`Execute`
//! contracts. Backend-local helpers are called directly by exact operation
//! executors; historical operation-family traits are not part of this backend.

pub use incin_core::backend_authoring::{
    Alignment, AttributeContract, Backend, CanonicalOperation, Capabilities, CapabilityQuery,
    CapabilityRegistry, Descriptor, DescriptorError, Execute, ExecuteOutput, ExecutionContext,
    ExecutionDescriptor, ExecutionRequest, LogicalTensorMeta, Operation, OperationCatalogEntry,
    OperationIdentity, OperationKey, ShapeBuf, StorageBackend, StorageOutput, SupportLevel,
    SupportsDType, TensorBackend, TensorMeta, TransferTo, UnsupportedReason, Validated, execute,
    execute_shaped, execute_shaped_with_payload, execute_with_payload,
};
pub use incin_core::prelude::{
    BackendError, ConversionFailure, Cpu, Device, DeviceId, DeviceKind, DType, DTypeDescriptor,
    DTypeId, Error, ErrorMessage, FloatDType, OperationKind, QuantDType, Result,
};

mod canonical;
mod legacy;
#[cfg(feature = "compiled")]
pub mod compiled;
#[cfg(feature = "compiled")]
pub use compiled::{
    CpuCompiledFunction, CpuCompiledInvocation, CpuCompiledPlan, CpuCompiledSupport,
    compiled_support,
};
pub(crate) mod creation;
/// GPU dispatcher modules (CUDA/Metal) — internal only.
pub(crate) mod gradcheck;
pub(crate) mod ops;
/// Internal storage types.
pub(crate) mod storage;
pub(crate) mod stride;
pub(crate) mod tape;
pub mod typed_kernel;
pub(crate) mod var;

// ── Public re-exports ─────────────────────────────────────────────────────────
// Only the three types a downstream crate legitimately needs to name:
//   - CpuBackendImpl<D>  to parameterise Tensor
//   - CpuStorage        as Backend::Storage<K>
//   - CpuVar            as Backend::Var<K>
//   - CpuGrads          as Backend::Grads
//   - CpuBuffer         for pattern-matching in to_bytes / from_bytes
pub use storage::{CpuBuffer, CpuStorage};
pub use tape::CpuGrads;
/// Number of entries currently on this backend's autograd tape.
///
/// Re-exported since `GRD-002`: the row claims a `NoGrad` chain records
/// nothing, and its evidence test lives outside this crate. A guarantee
/// nothing outside can observe is not a guarantee.
pub use tape::depth as tape_depth;
pub use var::CpuVar;

/// The CPU pure-Rust `Backend` implementor.
/// Also accessible as `IncinBackend<D>` from the `incin` facade
/// when only the `cpu` feature is active.
#[derive(Clone)]
pub struct CpuBackendImpl<D = Cpu>(core::marker::PhantomData<D>);

impl<D> CpuBackendImpl<D> {
    /// Construct the stateless CPU executor.
    #[must_use]
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<D> Default for CpuBackendImpl<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Device, K: DType> SupportsDType<K> for CpuBackendImpl<D> {
    fn resolve_dtype(field: &K::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
        let descriptor = K::descriptor(field);
        if descriptor.builtin_id().is_some() {
            Ok(descriptor)
        } else {
            Err(incin_core::prelude::Error::UnsupportedDType {
                dtype: descriptor,
                backend: "Cpu",
                op: "pointwise",
            })
        }
    }
}

pub(crate) fn validate_cpu_dtype(
    dtype: impl Into<DTypeDescriptor>,
    op: &'static str,
) -> Result<DTypeId> {
    let desc = dtype.into();
    if let Some(id) = desc.builtin_id()
        && matches!(
            id,
            DTypeId::F32
                | DTypeId::F64
                | DTypeId::F16
                | DTypeId::BF16
                | DTypeId::U8
                | DTypeId::U32
                | DTypeId::I64
                | DTypeId::Q8_0
                | DTypeId::Bool
        )
    {
        return Ok(id);
    }
    Err(Error::UnsupportedDType {
        dtype: desc,
        backend: "Cpu",
        op,
    })
}

impl<D: Device> incin_core::exec::PrecisionCapabilities for CpuBackendImpl<D> {
    fn native_precision(
        &self,
        request: &incin_core::exec::PrecisionRequest,
    ) -> Result<incin_core::exec::ResolvedPrecision> {
        validate_cpu_dtype(request.storage, "native_precision")?;

        let storage_id = request.storage.builtin_id();
        let compute = match storage_id {
            Some(DTypeId::F16 | DTypeId::BF16) => DTypeId::F32.descriptor(),
            _ => request.storage,
        };

        let accumulator = match request.operation {
            OperationKind::Reduction | OperationKind::Normalization
                if matches!(storage_id, Some(DTypeId::F16 | DTypeId::BF16)) =>
            {
                DTypeId::F32.descriptor()
            }
            _ => compute,
        };

        Ok(incin_core::exec::ResolvedPrecision::new(
            request.storage,
            compute,
            accumulator,
            request.output,
            incin_core::exec::LossScaling::None,
        ))
    }
}

impl<D: Device> incin_core::backend_authoring::StorageBackend for CpuBackendImpl<D> {
    type Device = D;
    const BACKEND_NAME: &'static str = "Cpu";
    type Storage<K: DType> = storage::CpuStorage;

    fn metadata<K: DType>(t: &Self::Storage<K>) -> &incin_core::backend_authoring::TensorMeta {
        &t.meta
    }

    fn fresh_autograd_identity<K: DType>(storage: Self::Storage<K>) -> Self::Storage<K> {
        storage.with_fresh_autograd_identity()
    }
}

impl incin_core::backend_authoring::StorageOutput for storage::CpuStorage {}

impl<D: Device> incin_core::backend_authoring::HostReadback for CpuBackendImpl<D> {
    fn float_to_vec1<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<f64>> {
        ops::shape_ops::float_to_vec1_storage(t)
    }

    fn int_to_vec1<K: DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<i64>> {
        ops::shape_ops::int_to_vec1_storage(t)
    }
}

impl<D: Device> incin_core::prelude::Backend for CpuBackendImpl<D> {

    /// `InnerBackend`.
    type InnerBackend = Self;
    // `host_format_display`/`host_format_debug` use `HostInterop`'s default,
    // which reads real values back through `float_to_vec1`/`int_to_vec1`.



}

impl<D: Device> incin_core::backend_authoring::HostInterop for CpuBackendImpl<D> {
    /// `to_bytes`.
        fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
            let t: &storage::CpuStorage = t;
            let t_contig = t.contiguous()?;
            let num_elements = stride::checked_numel(&t_contig.shape)?;
            let offset = t_contig.offset_elements;
            match &*t_contig.buffer {
                storage::CpuBuffer::F32(v) => {
                    Ok(bytemuck::cast_slice(&v[offset..offset + num_elements]).to_vec())
                }
                storage::CpuBuffer::F64(v) => {
                    Ok(bytemuck::cast_slice(&v[offset..offset + num_elements]).to_vec())
                }
                storage::CpuBuffer::U8(v) => Ok(v[offset..offset + num_elements].to_vec()),
                storage::CpuBuffer::Bool(v) => Ok(v[offset..offset + num_elements].to_vec()),
                storage::CpuBuffer::U32(v) => {
                    Ok(bytemuck::cast_slice(&v[offset..offset + num_elements]).to_vec())
                }
                storage::CpuBuffer::I64(v) => {
                    Ok(bytemuck::cast_slice(&v[offset..offset + num_elements]).to_vec())
                }
                storage::CpuBuffer::F16(v) => Ok(v[offset..offset + num_elements]
                    .iter()
                    .flat_map(|value| value.to_bits().to_ne_bytes())
                    .collect()),
                storage::CpuBuffer::BF16(v) => Ok(v[offset..offset + num_elements]
                    .iter()
                    .flat_map(|value| value.to_bits().to_ne_bytes())
                    .collect()),
                storage::CpuBuffer::Q8_0(v) => Ok(v
                    .iter()
                    .flat_map(|block| {
                        block
                            .d
                            .to_bits()
                            .to_ne_bytes()
                            .into_iter()
                            .chain(block.qs.iter().map(|value| *value as u8))
                    })
                    .collect()),
            }
        }
    /// `from_bytes`.
        fn from_bytes<K: DType>(
            bytes: &[u8],
            shape: &[usize],
            dtype: DTypeDescriptor,
            device: &DeviceId,
        ) -> Result<Self::Storage<K>> {
            if device.kind() != DeviceKind::Cpu || device.ordinal() != 0 {
                return Err(Error::DeviceInitializationError {
                    expected: "cpu:0".into(),
                    got: alloc::format!("{:?}:{}", device.kind(), device.ordinal()),
                });
            }
            let elements = shape.iter().try_fold(1usize, |count, dim| {
                count.checked_mul(*dim).ok_or(Error::InvalidByteLength {
                    expected: usize::MAX,
                    got: bytes.len(),
                })
            })?;
            let expected =
                dtype.size_bytes(elements, incin_core::shapes::error::OperationKind::Storage)?;
            if bytes.len() != expected {
                return Err(Error::InvalidByteLength {
                    expected,
                    got: bytes.len(),
                });
            }

            macro_rules! decode {
                ($ty:ty, $size:literal, $variant:ident) => {{
                    let values = bytes
                        .chunks_exact($size)
                        .map(|chunk| {
                            <$ty>::from_ne_bytes(chunk.try_into().expect("checked chunk size"))
                        })
                        .collect();
                    storage::CpuBuffer::$variant(values)
                }};
            }

            let builtin_id = dtype.builtin_id().ok_or_else(|| Error::UnsupportedDType {
                dtype,
                backend: "Cpu",
                op: "storage",
            })?;

            let buffer = match builtin_id {
                DTypeId::F32 => decode!(f32, 4, F32),
                DTypeId::F64 => decode!(f64, 8, F64),
                DTypeId::U8 => storage::CpuBuffer::U8(bytes.to_vec()),
                DTypeId::U32 => decode!(u32, 4, U32),
                DTypeId::I64 => decode!(i64, 8, I64),
                DTypeId::F16 => storage::CpuBuffer::F16(
                    bytes
                        .chunks_exact(2)
                        .map(|chunk| half::f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])))
                        .collect(),
                ),
                DTypeId::BF16 => storage::CpuBuffer::BF16(
                    bytes
                        .chunks_exact(2)
                        .map(|chunk| half::bf16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])))
                        .collect(),
                ),
                DTypeId::Q8_0 => storage::CpuBuffer::Q8_0(
                    bytes
                        .chunks_exact(34)
                        .map(|chunk| {
                            let mut qs = [0i8; 32];
                            for (dst, src) in qs.iter_mut().zip(&chunk[2..]) {
                                *dst = *src as i8;
                            }
                            storage::BlockQ8_0 {
                                d: half::f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])),
                                qs,
                            }
                        })
                        .collect(),
                ),
                DTypeId::Bool => {
                    if bytes.iter().any(|&b| b > 1) {
                        return Err(Error::Msg(
                            "invalid boolean byte representation: bytes must be 0 or 1".into(),
                        ));
                    }
                    storage::CpuBuffer::Bool(bytes.to_vec())
                }
                _ => {
                    return Err(Error::UnsupportedDType {
                        dtype,
                        backend: "Cpu",
                        op: "from_bytes",
                    });
                }
            };
            Ok(storage::CpuStorage::from_contiguous(buffer, shape.to_vec()))
        }
}

impl<D: Device> incin_core::backend_authoring::AutogradBackend for CpuBackendImpl<D> {
    type Grads = tape::CpuGrads;

    fn backward<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads> {
        tape::backward(t)
    }

    fn backward_with<K: DType>(
        t: &Self::Storage<K>,
        seed: &Self::Storage<K>,
    ) -> Result<Self::Grads> {
        tape::backward_with(t, seed)
    }

    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id).cloned())
    }
}


impl<D: Device> incin_core::backend_authoring::VariableBackend for CpuBackendImpl<D> {
    type Var<K: DType> = var::CpuVar;

    /// `var_as_tensor`.
    fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> Result<Self::Storage<K>> {
        var::var_as_tensor(var)
    }
    /// `var_from_tensor`.
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::Var<K>> {
        var::var_from_tensor(t)
    }
    /// `assign_var`.
    fn assign_var<K: DType>(var: &mut Self::Var<K>, tensor: &Self::Storage<K>) -> Result<()> {
        var::assign_var(var, tensor)
    }
}
