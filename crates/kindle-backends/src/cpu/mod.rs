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
    /// Core abstraction for `Device` within the Kindle framework..
    type Device = D;
    // CPUBACK-01: genuinely dispatched from T, NOT hardcoded f32.
    /// Core abstraction for `FloatElem` within the Kindle framework..
    type FloatElem = T;
    /// Core abstraction for `IntElem` within the Kindle framework..
    type IntElem = i64;
    /// Core abstraction for `Storage` within the Kindle framework..
    type Storage<K: DType> = storage::CpuStorage;
    /// Core abstraction for `RawVar` within the Kindle framework..
    type RawVar = var::CpuVar;
    /// Core abstraction for `Grads` within the Kindle framework..
    type Grads = tape::CpuGrads;
    /// Core abstraction for `InnerBackend` within the Kindle framework..
    type InnerBackend = Self;
    /// Core abstraction for `BackendWithDevice` within the Kindle framework..
    type BackendWithDevice<NewD: Device> = CpuBackend<T, NewD>;

    /// Core abstraction for `shape` within the Kindle framework..
    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
        t.shape.clone()
    }

    /// Core abstraction for `format_tensor_display` within the Kindle framework..
    fn format_tensor_display<K: DType>(t: &Self::Storage<K>) -> alloc::string::String {
        alloc::format!("CpuStorage(shape={:?})", t.shape)
    }

    /// Core abstraction for `format_tensor_debug` within the Kindle framework..
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String {
        alloc::format!(
            "CpuStorage(shape={:?}, strides={:?}, offset={})",
            t.shape,
            t.strides,
            t.offset
        )
    }

    /// Core abstraction for `backward` within the Kindle framework..
    fn backward<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads> {
        tape::backward(t)
    }

    /// Core abstraction for `backward_with_nan_check` within the Kindle framework..
    fn backward_with_nan_check<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads> {
        tape::backward_with_nan_check(t)
    }

    /// Core abstraction for `get_grad` within the Kindle framework..
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.get(t.id).cloned())
    }

    /// Core abstraction for `to_bytes` within the Kindle framework..
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

    /// Core abstraction for `from_bytes` within the Kindle framework..
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

    /// Core abstraction for `var_as_tensor` within the Kindle framework..
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        var::var_as_tensor(var)
    }

    /// Core abstraction for `var_from_tensor` within the Kindle framework..
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        var::var_from_tensor(t)
    }

    /// Core abstraction for `var_to_device` within the Kindle framework..
    fn var_to_device(var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
        // CPU-only no-op this phase (single-device target) — matches
        // CandleBackend's real to_device call structurally, but as a plain
        // clone since there is no multi-device support yet.
        Ok(var.clone())
    }

    /// Core abstraction for `assign_var` within the Kindle framework..
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var::assign_var(var, tensor)
    }
}
