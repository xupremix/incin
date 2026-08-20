//! Stride-aware CPU `matmul`, batched and unbatched, driven by an iteration
//! plan over the batch axes.
//!
//! Every kernel here addresses its operands through a `MatrixView`: a base
//! offset plus a row and a column stride into a shared `CpuBuffer`. A
//! transposed operand, a batch slice, and a batch axis that is only being
//! broadcast are then the same thing to the inner loop, so none of them ever
//! forces a `.contiguous()` materialization first (CPUBACK-02 / Pitfall 3).
//!
//! Batching reuses `crate::iteration`'s existing normalization rather than a
//! second copy of the broadcast rule: the batch axes of both operands are
//! normalized to the broadcast batch shape, a broadcast axis becomes a
//! zero stride, and the per-slice base offset is one `physical_index` call.
//! The previous implementation instead expanded both operands to the full
//! broadcast shape and reshaped, which is metadata-only for a contiguous
//! operand but materializes the entire expansion for any other one, so a
//! `[1, 3, 4]` operand batched against `[64, 4, 5]` copied `lhs` 64 times
//! before computing anything.
//!
//! With `cpu-blas` enabled, large `f32` GEMMs are handed to a blocked,
//! register-tiled kernel instead. That path is off by default and changes
//! only the order floating-point terms are accumulated in; the pure-Rust
//! path below stays complete and is what a default build runs.
//!
//! This module only contributes plain functions rather than its own `impl
//! ` block: Rust does not allow two separate `impl <..> for
//! CpuBackendImpl<..>` blocks for the same trait+type across two files, so
//! `ops/shape_ops/linalg.rs`'s single impl block calls into
//! `matmul_impl`/`batched_matmul_impl` for its `matmul` method.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `types` is `MatrixView`, the
//! stride-view every kernel below reads through; `transpose` is the two
//! transpose primitives the backward closures compose; `batched` is the
//! batched entry point and its tape-free forward kernel; `unbatched` is
//! the unbatched entry point and its own forward kernel (named `unbatched`
//! rather than `core` to avoid shadowing the `core` crate under this
//! module's own `use super::*;` globs); `gemm` is the actual `[m,k] @
//! [k,n]` inner kernel, tried in decreasing order of specificity
//! (blocked/BLAS, architecture SIMD, then the always-correct scalar path).

use incin_core::error::{Error, Result};
use incin_core::shapes::{OperationKind, ShapeBuf};

use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::stride;
use crate::cpu::tape::{self, TapeEntry};
use crate::iteration::{IterationPlan, OperandLayout};

mod batched;
mod gemm;
#[cfg(test)]
/// `tests`.
mod tests;
mod transpose;
mod types;
mod unbatched;

pub(crate) use batched::batched_matmul_impl;
pub(crate) use transpose::transpose_last2;
pub(crate) use unbatched::matmul_impl;

// `gemm`'s kernels and `types`'s `MatrixView` are `pub(super)` (this
// module's own reach, not wider), so `batched`/`core`/`tests` see them
// through their own `use super::*;` once they are imported here.
use gemm::{gemm, gemm_f64, writes_f32};
use transpose::transpose_2d;
use types::MatrixView;

// Test-only: `scalar_gemm` is otherwise private to `gemm` (only `gemm`
// itself calls it to pick a kernel) and reached only by `tests`, so a
// non-test build reports it unused.
#[cfg(test)]
#[allow(unused_imports)]
use gemm::scalar_gemm;
// Test-only, and `blocked_gemm` itself only exists with `cpu-blas` on: the
// cfg here must match `blocked_gemm`'s own gate in `gemm.rs`, or a
// test build without `cpu-blas` fails to resolve the import outright.
#[cfg(all(test, feature = "cpu-blas"))]
#[allow(unused_imports)]
use gemm::blocked_gemm;
