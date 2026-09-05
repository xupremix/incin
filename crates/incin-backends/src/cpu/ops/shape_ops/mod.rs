//! CPU tensor and shape operation helpers.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `view` is layout-preserving
//! reinterpretation (reshape, transpose, narrow, slice, squeeze/unsqueeze,
//! flatten, broadcast); `combine` assembles or windows storages (concat,
//! stack, unfold, pixel shuffle, repeat, pad); `convert` moves values
//! between storage and host scalars/vectors or between dtypes; `select`
//! reads or writes by index or predicate (masked fill, where, gather,
//! index select, scatter); `linalg` is matmul and its neighbors (addmm,
//! lerp, attention); `cmp` is elementwise comparison and scalar arithmetic;
//! `triangular` is triu/tril/diag; `norm` is group/instance normalization.

use incin_core::error::{BackendError, Error, Result};
use incin_core::shapes::{OperationKind, ShapeBuf, ShapeError};
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

use crate::cpu::ops::elementwise::{
    add_storage, canonical_add_scalar, canonical_mul_scalar, canonical_softmax, canonical_sqrt,
    elementwise_unary,
};
use crate::cpu::ops::matmul::{batched_matmul_impl, matmul_impl};
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::tape::{self, TapeEntry};

// Canonical executors and dynamic dispatch share these helpers so each
// operation has one CPU implementation.

mod cmp;
mod combine;
mod convert;
mod linalg;
mod norm;
mod select;
#[cfg(test)]
mod tests;
mod triangular;
mod view;

pub(crate) use cmp::{div_scalar_storage, elementwise_cmp, sub_scalar_storage};
pub(crate) use combine::{
    concat_storage, pad_storage, pixel_shuffle_storage, repeat_storage, stack_storage,
    unfold_storage,
};
pub(crate) use convert::{
    float_to_scalar_storage, float_to_vec1_storage, int_to_scalar_storage, int_to_vec1_storage,
    tensor_to_dtype_storage,
};
pub(crate) use linalg::{
    addmm_storage, lerp_storage, matmul_storage, scaled_dot_product_attention_storage,
};
pub(crate) use norm::{group_norm_storage, instance_norm_storage};
pub(crate) use select::{
    gather_storage, index_select_storage, masked_fill_storage, one_hot_storage,
    scatter_add_storage, scatter_storage, where_storage,
};
pub(crate) use triangular::{diag_storage, tril_storage, triu_storage};
pub(crate) use view::{
    broadcast_as_storage, broadcast_left_storage, flatten_storage, narrow_storage, reshape_storage,
    slice_storage, squeeze_storage, transpose_storage, unsqueeze_storage,
};
