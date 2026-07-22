//! # Kindle Cpu
//!
//! `kindle-cpu` is a pure-Rust, ownership-flavored, CPU-only implementation
//! of the `Backend` trait defined in `kindle-core`. It provides its own
//! strided-view tensor storage (`storage`) and shape/stride math (`stride`),
//! independent of any external tensor compute library (Candle, ndarray, burn).
//!
//! This crate is built incrementally across multiple phases. This plan wires
//! up `CpuBackendImpl<T, D>`'s `Backend` trait impl (associated types,
//! `shape`/`backward`/`get_grad`/`format_*`/`var_*`/`assign_var`/`to_bytes`/
//! `from_bytes`) plus `CreationOps` and a minimal `NumericOps`/`FloatOps`
//! subset. `CpuBackendImpl` is not yet a fully `Backend`-complete implementor
//! after this plan — `TensorOps`/`ReductionOps`/`ModuleOps`/`LossOps` land in
//! later plans.

pub use kindle_core::prelude::*;

pub(crate) mod creation;
/// GPU dispatcher modules (CUDA/Metal) — internal only.
pub(crate) mod gradcheck;
pub(crate) mod ops;
/// Internal storage types.
pub(crate) mod storage;
pub(crate) mod stride;
pub(crate) mod tape;
pub(crate) mod var;

// ── Public re-exports ─────────────────────────────────────────────────────────
// Only the three types a downstream crate legitimately needs to name:
//   - CpuBackendImpl<T, D>  to parameterise Tensor
//   - CpuStorage        as Backend::Storage<K>
//   - CpuVar            as Backend::RawVar
//   - CpuGrads          as Backend::Grads
//   - CpuBuffer         for pattern-matching in to_bytes / from_bytes
pub use storage::{CpuBuffer, CpuStorage};
pub use tape::CpuGrads;
pub use var::CpuVar;

/// The CPU pure-Rust `Backend` implementor. `T` genuinely drives
/// `Backend::FloatElem` (CPUBACK-01, D-03).
/// Also accessible as `KindleBackend<T, D>` from the `kindle` facade
/// when only the `cpu` feature is active.
#[derive(Clone)]
pub struct CpuBackendImpl<T = f32, D = Cpu>(core::marker::PhantomData<(T, D)>);

impl<T: DType, D: Device, K: DType> SupportsDType<K> for CpuBackendImpl<T, D> {}

impl<T: DType, D: Device> kindle_core::prelude::Backend for CpuBackendImpl<T, D> {
    /// `Device`.
    type Device = D;
    // CPUBACK-01: genuinely dispatched from T, NOT hardcoded f32.
    /// `FloatElem`.
    type FloatElem = T;
    /// `IntElem`.
    type IntElem = i64;
    /// `Storage`.
    type Storage<K: DType> = storage::CpuStorage;
    /// `RawVar`.
    type RawVar = var::CpuVar;
    /// `Grads`.
    type Grads = tape::CpuGrads;
    /// `InnerBackend`.
    type InnerBackend = Self;
    /// `shape`.
    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
        t.shape.clone()
    }

    fn storage_dtype<K: DType>(t: &Self::Storage<K>) -> Option<DTypeId> {
        Some(match &*t.buffer {
            CpuBuffer::F32(_) => DTypeId::F32,
            CpuBuffer::F64(_) => DTypeId::F64,
            CpuBuffer::U8(_) => DTypeId::U8,
            CpuBuffer::U32(_) => DTypeId::U32,
            CpuBuffer::I64(_) => DTypeId::I64,
            CpuBuffer::F16(_) => DTypeId::F16,
            CpuBuffer::BF16(_) => DTypeId::BF16,
            CpuBuffer::Q8_0(_) => DTypeId::Q8_0,
        })
    }

    fn storage_device<K: DType>(_t: &Self::Storage<K>) -> Option<DeviceId> {
        Some(DeviceId::cpu())
    }

    /// `format_tensor_display`.
    fn format_tensor_display<K: DType>(t: &Self::Storage<K>) -> alloc::string::String {
        alloc::format!("CpuStorage(shape={:?})", t.shape)
    }

    /// `format_tensor_debug`.
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String {
        alloc::format!(
            "CpuStorage(shape={:?}, strides={:?}, offset={})",
            t.shape,
            t.strides,
            t.offset
        )
    }

    /// `backward`.
    fn backward<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads> {
        tape::backward(t)
    }

    /// `backward_with_nan_check`.
    fn backward_with_nan_check<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads> {
        tape::backward_with_nan_check(t)
    }

    /// `get_grad`.
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id).cloned())
    }

    /// `to_bytes`.
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
        let t_contig = t.contiguous();
        let num_elements = t_contig.shape.iter().product::<usize>();
        let offset = t_contig.offset;
        match &*t_contig.buffer {
            storage::CpuBuffer::F32(v) => {
                Ok(bytemuck::cast_slice(&v[offset..offset + num_elements]).to_vec())
            }
            storage::CpuBuffer::F64(v) => {
                Ok(bytemuck::cast_slice(&v[offset..offset + num_elements]).to_vec())
            }
            storage::CpuBuffer::U8(v) => Ok(v[offset..offset + num_elements].to_vec()),
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
        dtype: DTypeId,
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
        let expected = match dtype {
            DTypeId::Q8_0 => elements.div_ceil(32) * 34,
            _ => elements * dtype.element_size(),
        };
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

        let buffer = match dtype {
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

    /// `var_as_tensor`.
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        var::var_as_tensor(var)
    }

    /// `var_from_tensor`.
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        var::var_from_tensor(t)
    }

    /// `assign_var`.
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var::assign_var(var, tensor)
    }
}
