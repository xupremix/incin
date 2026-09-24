//! WebGPU compute backend implementation for Incin.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `types` is the backend
//! value types every other module attaches its impls to; `util` is shared
//! element-count/checked-conversion/validation helpers; `contract` is the
//! `StorageBackend`/`Backend`/`HostInterop` trait implementations;
//! `creation` is zero-operand and generator operations; `elementwise`
//! is tape-tracked binary/unary/scalar arithmetic; `compare` is the nine
//! bool-output comparison/logical identities; `shape_ops` is
//! structural operations; `reduce` is reductions; `nn` is pooling and
//! convolution; `indexing` is `embedding`/`gather`/`index_select` host-walks
//! plus the `masked_fill`/`where_cond` GPU selection kernel; `autograd` is
//! `AutogradBackend`/`VariableBackend`.

pub(crate) use crate::wgpu::capability::validate_wgpu_dtype;
pub(crate) use crate::wgpu::dispatch;
pub(crate) use crate::wgpu::storage::{WgpuBuffer, WgpuStorage};
pub(crate) use incin_core::backend_authoring::*;
pub(crate) use incin_core::error::{BackendError, Error, Result};
pub(crate) use incin_core::shapes::{OperationKind, ShapeError, StrideBuf};
pub(crate) use incin_core::tensor::device::{Device, DeviceId, DeviceKind, Wgpu};
pub(crate) use incin_core::tensor::dtype::{DType, DTypeDescriptor, DTypeId};

mod autograd;
mod compare;
mod contract;
mod creation;
mod elementwise;
mod indexing;
mod nn;
mod reduce;
mod shape_ops;
mod types;
mod util;

pub use types::{WgpuBackendImpl, WgpuGrads, WgpuVar};

pub(crate) use elementwise::{broadcast_storage, broadcast_storage_raw, scalar_op};
pub(crate) use indexing::require_f32;
pub(crate) use util::{checked_u32, checked_u32_array, num_elements, validate_wgpu};
