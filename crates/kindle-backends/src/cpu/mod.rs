//! # Kindle Cpu
//!
//! `kindle-cpu` is a pure-Rust, ownership-flavored, CPU-only implementation
//! of the `Backend` trait defined in `kindle-core`. It provides its own
//! strided-view tensor storage (`storage`) and shape/stride math (`stride`),
//! independent of any external tensor compute library (Candle, ndarray, burn).
//!
//! This crate is built incrementally across multiple phases. This plan wires
//! up `CpuBackend<T, D>`'s `Backend` trait impl (associated types,
//! `shape`/`backward`/`get_grad`/`format_*`/`var_*`/`assign_var`/`to_bytes`/
//! `from_bytes`) plus `CreationOps` and a minimal `NumericOps`/`FloatOps`
//! subset. `CpuBackend` is not yet a fully `Backend`-complete implementor
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
//   - CpuBackend<T, D>  to parameterise Tensor
//   - CpuStorage        as Backend::Storage<K>
//   - CpuVar            as Backend::RawVar
//   - CpuGrads          as Backend::Grads
//   - CpuBuffer         for pattern-matching in to_bytes / from_bytes
pub use storage::{CpuBuffer, CpuStorage};
pub use tape::CpuGrads;
pub use var::CpuVar;

/// The cpu, pure-Rust `Backend` implementor. `T` genuinely drives
/// `Backend::FloatElem` (CPUBACK-01, D-03) — unlike `CandleBackend`/
/// `NdarrayBackend`, which hardcode `type FloatElem = f32;` regardless of
/// their own `T` generic.
#[derive(Clone)]
pub struct CpuBackend<T, D>(core::marker::PhantomData<(T, D)>);

impl<T: DType, D: Device> kindle_core::prelude::Backend for CpuBackend<T, D> {
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
    /// `BackendWithDevice`.
    type BackendWithDevice<NewD: Device> = CpuBackend<T, NewD>;

    /// `shape`.
    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
        t.shape.clone()
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
                let slice = &v[offset..offset + num_elements];
                Ok(bytemuck::cast_slice(slice).to_vec())
            }
            _ => Err(Error::UnsupportedBackendOperation {
                op: "to_bytes",
                backend: "Cpu",
            }),
        }
    }

    /// `from_bytes`.
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<Self::Storage<K>> {
        match dtype {
            KindleDType::F32 => {
                let floats: &[f32] = bytemuck::cast_slice(bytes);
                Ok(storage::CpuStorage::from_contiguous(
                    storage::CpuBuffer::F32(floats.to_vec()),
                    shape.to_vec(),
                ))
            }
            _ => Err(Error::UnsupportedBackendOperation {
                op: "from_bytes",
                backend: "Cpu",
            }),
        }
    }

    /// `var_as_tensor`.
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        var::var_as_tensor(var)
    }

    /// `var_from_tensor`.
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        var::var_from_tensor(t)
    }

    /// `var_to_device`.
    fn var_to_device(var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
        // CPU-only no-op this phase (single-device target) — matches
        // CandleBackend's real to_device call structurally, but as a plain
        // clone since there is no multi-device support yet.
        Ok(var.clone())
    }

    /// `assign_var`.
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var::assign_var(var, tensor)
    }
}
