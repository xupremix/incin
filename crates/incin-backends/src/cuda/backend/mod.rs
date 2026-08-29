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

pub(crate) mod autograd;
pub(crate) mod contract;
pub(crate) mod creation;
pub(crate) mod elementwise;
pub(crate) mod nn;
pub(crate) mod reduce;
pub(crate) mod shape_ops;
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
#[allow(unused_imports)]
pub(crate) use elementwise::{
    cuda_abs_diff_storage, cuda_abs_storage, cuda_acos_storage, cuda_acosh_storage,
    cuda_add_scalar_float, cuda_add_storage, cuda_asin_storage, cuda_asinh_storage,
    cuda_atan_storage, cuda_atan2_storage, cuda_atanh_storage, cuda_ceil_storage,
    cuda_clamp_storage, cuda_cos_storage, cuda_cosh_storage, cuda_div_scalar_float,
    cuda_div_storage, cuda_elu_storage, cuda_erf_storage, cuda_exp_storage, cuda_floor_storage,
    cuda_fmod_storage, cuda_frac_storage, cuda_gelu_storage, cuda_lerp_storage, cuda_log_softmax,
    cuda_log_storage, cuda_log2_storage, cuda_log10_storage, cuda_maximum_storage,
    cuda_minimum_storage, cuda_mish_storage, cuda_mul_scalar_float, cuda_mul_storage,
    cuda_neg_storage, cuda_powf_storage, cuda_relu_storage, cuda_remainder_storage,
    cuda_round_storage, cuda_rsqrt_storage, cuda_sigmoid_storage, cuda_sign_storage,
    cuda_sin_storage, cuda_sinh_storage, cuda_softmax, cuda_sqrt_storage, cuda_step_storage,
    cuda_sub_scalar_float, cuda_sub_storage, cuda_swish_storage, cuda_tan_storage,
    cuda_tanh_storage, cuda_trunc_storage, push_unary_tape_entry,
};
#[allow(unused_imports)]
pub(crate) use elementwise::{
    cuda_add_scalar_float as cuda_add_scalar_storage,
    cuda_div_scalar_float as cuda_div_scalar_storage,
    cuda_mul_scalar_float as cuda_mul_scalar_storage,
    cuda_sub_scalar_float as cuda_sub_scalar_storage,
};
