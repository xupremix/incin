//! # Kindle Native
//!
//! `kindle-native` is a pure-Rust, ownership-flavored, CPU-only implementation
//! of the `Backend` trait defined in `kindle-core`. It provides its own
//! strided-view tensor storage (`storage`) and shape/stride math (`stride`),
//! independent of any external tensor compute library (Candle, ndarray, burn).
//!
//! This crate is built incrementally across multiple phases. This plan wires
//! up `NativeBackend<T, D>`'s `Backend` trait impl (associated types,
//! `shape`/`backward`/`get_grad`/`format_*`/`var_*`/`assign_var`/`to_bytes`/
//! `from_bytes`) plus `CreationOps` and a minimal `NumericOps`/`FloatOps`
//! subset. `NativeBackend` is not yet a fully `Backend`-complete implementor
//! after this plan — `TensorOps`/`ReductionOps`/`ModuleOps`/`LossOps` land in
//! later plans.
#[macro_use]
extern crate alloc;


pub use kindle_core::prelude::*;

pub(crate) mod creation;
pub mod gpu;
pub(crate) mod gradcheck;
pub(crate) mod ops;
pub mod storage;
pub(crate) mod stride;
pub(crate) mod tape;
pub(crate) mod var;

/// The native, pure-Rust `Backend` implementor. `T` genuinely drives
/// `Backend::FloatElem` (NATBACK-01, D-03) — unlike `CandleBackend`/
/// `NdarrayBackend`, which hardcode `type FloatElem = f32;` regardless of
/// their own `T` generic.
#[derive(Clone)]
pub struct NativeBackend<T, D>(core::marker::PhantomData<(T, D)>);

impl<T: DType, D: Device> kindle_core::prelude::Backend for NativeBackend<T, D> {
    type Device = D;
    // NATBACK-01: genuinely dispatched from T, NOT hardcoded f32.
    type FloatElem = T;
    type IntElem = i64;
    type Storage<K: DType> = storage::NativeStorage;
    type RawVar = var::NativeVar;
    type Grads = tape::NativeGrads;
    type InnerBackend = Self;
    type BackendWithDevice<NewD: Device> = NativeBackend<T, NewD>;

    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
        t.shape.clone()
    }

    fn format_tensor_display<K: DType>(t: &Self::Storage<K>) -> alloc::string::String {
        alloc::format!("NativeStorage(shape={:?})", t.shape)
    }

    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String {
        alloc::format!(
            "NativeStorage(shape={:?}, strides={:?}, offset={})",
            t.shape,
            t.strides,
            t.offset
        )
    }

    fn backward<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads> {
        tape::backward(t)
    }

    fn backward_with_nan_check<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads> {
        tape::backward_with_nan_check(t)
    }

    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.grads.get(&t.id).cloned())
    }

    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
        match &*t.buffer {
            storage::NativeBuffer::F32(v) => Ok(bytemuck::cast_slice(&v).to_vec()),
            _ => Err(Error::UnsupportedBackendOperation {
                op: "to_bytes",
                backend: "Native",
            }),
        }
    }

    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<Self::Storage<K>> {
        match dtype {
            KindleDType::F32 => {
                let floats: &[f32] = bytemuck::cast_slice(bytes);
                Ok(storage::NativeStorage::from_contiguous(
                    storage::NativeBuffer::F32(floats.to_vec()),
                    shape.to_vec(),
                ))
            }
            _ => Err(Error::UnsupportedBackendOperation {
                op: "from_bytes",
                backend: "Native",
            }),
        }
    }

    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        var::var_as_tensor(var)
    }

    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        var::var_from_tensor(t)
    }

    fn var_to_device(var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
        // CPU-only no-op this phase (single-device target) — matches
        // CandleBackend's real to_device call structurally, but as a plain
        // clone since there is no multi-device support yet.
        Ok(var.clone())
    }

    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var::assign_var(var, tensor)
    }
}
