#![allow(clippy::extra_unused_type_parameters)]

//! CUDA compute backend implementation for Incin.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `types` is the backend
//! value types every other module attaches its impls to; `shape_ops` is
//! structural/layout operations; `elementwise` is tape-tracked binary,
//! unary, and scalar arithmetic; `creation` is zero-operand and generator
//! operations; `reduce` is reductions; `nn` is pooling and convolution;
//! `contract` is the `StorageBackend`/`Backend`/`HostInterop` trait
//! implementations; `autograd` is `AutogradBackend`/`VariableBackend`.

pub(crate) use crate::bytes::checked_numel;
pub(crate) use crate::cuda::storage::CudaStorage;
pub(crate) use alloc::sync::Arc;
pub(crate) use incin_core::backend_authoring::*;
pub(crate) use incin_core::error::{BackendError, Error, Result};
pub(crate) use incin_core::shapes::{OperationKind, ShapeError};
pub(crate) use incin_core::tensor::device::{Cuda, Device, DeviceId, DeviceKind};
pub(crate) use incin_core::tensor::dtype::{DType, DTypeDescriptor, DTypeId};

pub(crate) use crate::cuda::capability::{
    native_precision, require_cuda_builtin_dtype, validate_cuda_storage_dtype,
};

mod autograd;
mod contract;
mod creation;
mod elementwise;
mod nn;
mod reduce;
mod shape_ops;
#[cfg(test)]
mod tests;
mod types;

pub use types::{CudaBackendImpl, CudaGrads, CudaVar};

pub(crate) use contract::{cuda_from_f32, cuda_require_f32};
// Test-only re-exports: these four are otherwise private to `contract` and
// are reached only by `tests`, so a non-test build reports them unused.
#[allow(unused_imports)]
pub(crate) use contract::{
    checked_storage_byte_len, cuda_from_bytes, download_f32_host, validate_cuda_storage,
};
pub(crate) use elementwise::{
    cuda_add_storage, cuda_div_storage, cuda_exp_storage, cuda_log_storage, cuda_mul_storage,
    cuda_relu_storage, cuda_sigmoid_storage, cuda_softmax, cuda_sqrt_storage, cuda_sub_storage,
    cuda_tanh_storage, push_unary_tape_entry,
};
