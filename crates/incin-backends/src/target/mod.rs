//! A device value as the user-facing allocation target.
//!
//! # What problem this solves
//!
//! Constructing a tensor today means naming a backend type and packing shape,
//! dtype, device and grad arguments into one positional tuple whose arity
//! depends on which of those four happen to be static:
//!
//! ```text
//! Tensor::<s![2, 3], IncinBackend<Cuda>>::zeros(((), Cuda::new(2)))
//! ```
//!
//! The leading `()` is not decoration. A fully static shape's `Shape::Arg` is
//! a tuple of units, `arg_into`'s `NotUnit` marker counts that as an argument
//! the caller supplied, and the device selector therefore has to be shifted
//! into second position. Get it wrong and the diagnostic is an unsatisfied
//! `ArgInto<TensorArgsData<..>>` bound that names none of the four things it
//! is actually talking about.
//!
//! Here the same allocation is:
//!
//! ```text
//! gpu.zeros(shape![2, 3])
//! ```
//!
//! Three things now come from three places that cannot be confused for one
//! another, instead of from one tuple whose arity you had to work out:
//!
//! | What | Comes from | Example |
//! |---|---|---|
//! | backend, device | the target value | `gpu` |
//! | geometry | the shape argument | `shape![batch, 784]` |
//! | dtype | the *data*, or the target's bound float | `[0_i64, 1]` / `gpu` |
//!
//! None of the three can be written in another's position, so there is no
//! order to remember and no `()` placeholder to get wrong.
//!
//! # Why a device value and not a backend, runtime or context
//!
//! - **Backends are zero-sized.** `CpuBackendImpl<D>` is a `PhantomData`
//!   and `IncinBackend<D>` is a *type alias* for
//!   `<Native as EngineOn<D>>::Backend`. Making a caller name one is making them
//!   name nothing.
//! - **There is no runtime state to own.** WGPU's device and queue live in a
//!   process-global `OnceLock`; CUDA's contexts live in a global map that is
//!   deliberately never evicted because releasing the last handle costs a
//!   131 ms re-initialization. A `Runtime` object would claim ownership of
//!   resources it does not hold.
//! - **`ExecutionContext` is policy, not placement.** It pairs a backend value
//!   with an [`ExecutionPolicy`]; that is a real job and a different one.
//!
//! [`ExecutionPolicy`]: incin_core::exec::ExecutionPolicy
//!
//! # Why the target carries a float dtype
//!
//! [`EngineOn<D>`](crate::target::EngineOn) maps *engine × device* to a
//! backend family. Carrying a default float dtype as an associated type on
//! [`TensorTarget`] rather than a generic parameter keeps type inference
//! total: `Cpu` is a target at `f32`, and
//! [`dtype`](DtypeTarget::dtype) produces the view for any other
//! dtype.
//!
//! The float is the dtype of *generated* tensors and layer parameters only.
//! Data tensors take their dtype from the data and are never cast — see
//! [`TargetExt::tensor`].
//!
//! This module is split by concern per `docs/CONVENTIONS.md`: `place` is
//! the `TensorTarget`/`DtypeTarget` abstraction itself, `data` is Rust data
//! entering a tensor, `ext` is the creation surface application code calls,
//! `engine` resolves an engine and device to a backend, and `concrete` is
//! the `Target<E, D, P>` value type.

pub(crate) use alloc::vec::Vec;
pub(crate) use core::fmt::Debug;
pub(crate) use core::hash::Hash;

pub(crate) use incin_core::backend_authoring::{
    Backend, HostInterop, StorageBackend, SupportsDType, VariableBackend,
};
pub(crate) use incin_core::error::{Error, Result};
pub use incin_core::exec::precision;
pub use incin_core::exec::{PrecisionSpec, RuntimePrecisionPolicy};
pub(crate) use incin_core::shapes::dynamic::Dyn;
pub(crate) use incin_core::shapes::{
    ConstDim, Dim, DimCons, DynShape, Nil, Shape, ShapeBuf, ShapeSpec,
};
pub(crate) use incin_core::tensor::base::Tensor;
#[cfg(any(feature = "cpu", feature = "external-candle"))]
pub(crate) use incin_core::tensor::device::Cpu;
#[cfg(feature = "metal")]
pub(crate) use incin_core::tensor::device::Metal;
#[cfg(feature = "cuda")]
pub(crate) use incin_core::tensor::device::{Cuda, CudaN};
pub(crate) use incin_core::tensor::device::{Device, DeviceId};
#[cfg(feature = "wgpu")]
pub(crate) use incin_core::tensor::device::{Wgpu, WgpuN};
pub(crate) use incin_core::tensor::dtype::{
    BuiltinDType, ConstDType, DType, DTypeDescriptor, FloatDType, PlainDType,
};
pub(crate) use incin_core::tensor::grad::{Grad, NoGrad, RequiresGrad};

mod concrete;
mod data;
mod engine;
mod ext;
mod place;

pub use concrete::Target;
pub use data::TensorData;
#[cfg(feature = "external-candle")]
pub use engine::Candle;
pub use engine::{EngineBackend, EngineOn, EngineSpec, Native, NativeBackend, RuntimeEngine};
pub use ext::{GeneratedFill, TargetExt};
pub use place::{
    DtypeTarget, DtypeView, TargetBackend, TargetBackendFor, TargetTensor, TensorTarget,
};
