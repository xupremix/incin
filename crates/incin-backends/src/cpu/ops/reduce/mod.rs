//! CPU reduction kernels: every method now has a real
//! implementation — `sum_all`/`mean_all`/`sum_dim`/`sum_keepdim` (Phase 1),
//! `mean_dim`/`mean_keepdim`/`max_dim`/`max_keepdim`/`min_dim`/`min_keepdim`/
//! `max_all`/`min_all` (Phase 2, gradcheck-verified backward), and
//! `argmax`/`argmin` (Phase 2, forward-only by structural design). Zero
//! remaining unsupported-backend-operation error stubs.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `helpers` is the shared
//! index/dense-read/fold machinery every reducer below builds on; `all`
//! is the whole-tensor reducers (`sum_all`, `mean_all`, `max_all`,
//! `min_all`, `prod_all`); `dim` is the axis-parameterized reducers
//! (`sum_dim`/`sum_keepdim` and friends, `prod_dim`, `cumsum`); `select`
//! is the index-producing family (`argmax`, `argmin`, `topk`, `argsort`).
//!
//! ## Design Notes
//!
//! * `sum_all` / `mean_all` backward: the incoming scalar gradient must be
//!   *broadcast* back to every element of the original shape — the exact
//!   inverse of sum. This is NOT a call to `tape::unbroadcast` (which handles
//!   the opposite direction); instead, the backward closure fills a new
//!   contiguous storage with `grad_scalar / n` (for `mean_all`) or
//!   `grad_scalar` (for `sum_all`) repeated across the original shape.
//!
//! * `sum_dim` / `sum_keepdim` need real implementations even though
//!   PATTERNS.md marks them "stub acceptable" at the public-trait level,
//!   because `tape::unbroadcast` (Plan 02) depends on the same axis-reduce
//!   logic internally. Rather than making tape.rs's private helpers
//!   `pub(crate)` and introducing a dependency, this file carries its own
//!   `sum_axis_keepdim` / `sum_axis_squeeze` helpers — identical in logic to
//!   tape.rs's private versions, independent in scope, so that neither side
//!   regresses the other's tests.
//!
//! * `mean_dim` / `mean_keepdim` are thin wrappers over `sum_axis_squeeze` /
//!   `sum_axis_keepdim`, divided by axis length (forward and backward both).
//!
//! * `max_dim` / `min_dim` / `max_keepdim` / `min_keepdim` / `max_all` /
//!   `min_all` route gradient to exactly one winning element per output
//!   position via `max_axis_with_indices` / `min_axis_with_indices` (strict
//!   `>`/`<` comparison — first-encountered winner on ties, never splitting
//!   or duplicating gradient mass) and the shared `scatter_axis_grad`
//!   backward helper.
//!
//! * `argmax` / `argmin` are forward-only — `incin-core`'s
//!   `Tensor::argmax`/`argmin` structurally force `G = NoGrad` on their
//!   output regardless of the input's own `G`, so neither method calls
//!   `tape::push` (the one deliberate exception to this file's
//!   every-other-method unconditional-push convention).
//!
//! * Any leftover unimplemented method (there are none as of Phase 2) would
//!   keep returning the typed unsupported-backend-operation error — never a
//!   silent `Ok(t.clone())` placeholder (T-01-15 mitigation).

use incin_core::error::Error;
use incin_core::error::Result;
use incin_core::shapes::{OperationKind, ShapeError};
use incin_core::tensor::dtype::{DType, DTypeDescriptor, DTypeId};

use crate::cpu::ops::elementwise::increment_index;
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::stride::contiguous_strides;
use crate::cpu::tape::{self, TapeEntry};

mod all;
mod dim;
mod helpers;
mod select;
#[cfg(test)]
/// `tests`.
mod tests;

pub(crate) use all::{max_all, mean_all, min_all, prod_all, sum_all};
pub(crate) use dim::{
    cumsum, max_dim, max_keepdim, mean_dim, mean_keepdim, min_dim, min_keepdim, prod_dim, sum_dim,
    sum_keepdim,
};
pub(crate) use helpers::sum_axis_keepdim;
pub(crate) use select::{argmax, argmin, argsort, topk};

// `helpers`'s cross-file machinery is `pub(super)` (this module's own reach,
// not wider), so `all`/`dim`/`select` see it through their own `use
// super::*;` once it is imported here.
use helpers::{
    DenseReader, dense_reader, fill_like, flatten_index, fold_all_f64, index_buffer,
    max_axis_with_indices, min_axis_with_indices, scatter_axis_grad, sum_axis_squeeze,
    total_sum_f64, unflatten_index,
};
