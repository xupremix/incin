//! Typed CPU pointwise kernel families.
//!
//! Operations are represented once and evaluated using the compute type
//! selected for each storage dtype. F16/BF16 use F32 compute; F32 and F64
//! stay native. The dispatcher specializes contiguous and scalar-broadcast
//! layouts and uses normalized typed iteration for general views.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `types` is `BinaryOp`/
//! `UnaryOp` and AVX2 availability probes; `dispatch` is the entry points
//! that pick a layout-specific execution path; `scalar` is the
//! architecture-independent scalar fallback kernels; `avx2`, `neon`, and
//! `wasm` are the per-architecture SIMD kernels; `strided` is the shared
//! generic-iteration mapping helpers; `util` is bounds/range/erf helpers.

use core::ops::Range;

use half::{bf16, f16};
use incin_core::error::{Error, Result};
use rayon::prelude::*;

use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::stride;
use crate::cpu::typed_kernel::{TypedKernel, map_binary_typed, map_unary_typed};
use crate::iteration::{IterationPlan, OperandIteration, OperandLayout, UnaryIterationPlan};
use crate::simd_lanes;

// The release microbenchmark shows thread-pool dispatch dominates through
// tens of thousands of elements while large tensors benefit substantially.
// Keep this conservative initial crossover explicit and retune per
// architecture once distributions and thread-count metadata are recorded.
const PARALLEL_GRAIN: usize = 256 * 1024;
// Explicit AVX2 remains faster beyond the generic scalar loop's parallel
// crossover. This separate cutoff is benchmarked by the ignored release test.
const DENSE_PARALLEL_GRAIN: usize = 2 * 1024 * 1024;
const SIMD_PARALLEL_CHUNK: usize = 128 * 1024;

mod avx2;
mod dispatch;
mod neon;
mod scalar;
mod strided;
#[cfg(test)]
mod tests;
mod types;
mod util;
mod wasm;

pub(crate) use dispatch::{execute_binary, execute_unary};
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
pub(crate) use types::{BinaryOp, UnaryOp};
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(crate) use types::{BinaryOp, UnaryOp, avx2_f32_available, avx2_f64_available};
pub(crate) use util::dense_range;

// Cross-submodule wiring: `dispatch` calls into every SIMD/scalar/strided
// family to pick a layout-specific execution path, and `scalar` calls into
// the per-architecture SIMD kernels to fall back when a SIMD path declines
// (empty input, unsupported stride pattern, feature not detected at
// runtime). None of these are part of the crate's public surface, so each
// is `pub(super)` in its defining file (visible in `elementwise_kernel` and
// its descendants); re-exported here with plain (private) `use`, which
// carries that same visibility to every sibling file's `use super::*;`
// without widening it further.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
use avx2::{
    avx2_binary_f32, avx2_binary_f64, avx2_scalar_f32, avx2_scalar_f64, map_iteration_avx2_f32,
    map_iteration_avx2_f64, parallel_avx2_binary_f32, parallel_avx2_binary_f64,
    parallel_avx2_scalar_f32, parallel_avx2_scalar_f64,
};
#[cfg(target_arch = "aarch64")]
use neon::{neon_binary_f32, neon_binary_f64, neon_scalar_f32, neon_scalar_f64};
#[cfg(all(feature = "std", target_arch = "aarch64"))]
use neon::{
    parallel_neon_binary_f32, parallel_neon_binary_f64, parallel_neon_scalar_f32,
    parallel_neon_scalar_f64,
};
use util::{erf_approx_f64, scalar_value, validate_bounds};
// Test-only: `binary_iteration_plan` is otherwise private to `dispatch` and
// reached only by `tests`, so a non-test build reports it unused.
#[allow(unused_imports)]
use dispatch::binary_iteration_plan;
use scalar::{map_binary, map_binary_f32, map_binary_f64, map_scalar_f32, map_scalar_f64};
use strided::{
    map_binary_strided, map_scalar_left, map_scalar_right, map_unary, map_unary_strided,
};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use wasm::{wasm_binary_f32, wasm_binary_f64, wasm_scalar_f32, wasm_scalar_f64};
