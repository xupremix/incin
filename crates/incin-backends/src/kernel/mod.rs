//! Internal kernel specialization vocabulary and source rendering.
//!
//! Kernels are described once and specialized lazily by dtype. This keeps
//! source maintenance proportional to operation families rather than to the
//! Cartesian product of operations, dtypes, layouts, and devices.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `types` is the key/kernel
//! vocabulary every renderer shares (`KernelFamily`, `KernelAccess`,
//! `KernelDType`, `KernelIndexWidth`, `KernelKey`, `RenderedKernel`);
//! `scalar` renders one-element-per-thread unary/binary kernels; `packed`
//! renders vectorized (multiple-elements-per-thread) unary/binary kernels;
//! `reduction` and `normalization` are their own template families, each
//! large enough (and specific enough) to own a file. Every renderer is
//! `#[cfg(any(feature = "cuda", test))]`, matching the original file.

#[cfg(any(feature = "cuda", test))]
use crate::codegen::ScalarFragment;
#[cfg(any(feature = "cuda", test))]
use alloc::boxed::Box;
use alloc::string::String;
use incin_core::error::{Error, Result};
#[cfg(feature = "cuda")]
use incin_core::exec::PrecisionRequest;
use incin_core::exec::{LayoutClass, MathMode};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DTypeId;
const KERNEL_KEY_SCHEMA_VERSION: u8 = 1;

mod normalization;
mod packed;
mod reduction;
mod scalar;
#[cfg(test)]
mod tests;
mod types;

#[cfg(any(feature = "cuda", test))]
pub(crate) use normalization::render_cuda_normalization;
// The `_body` entry points are the live ones; the `&str` wrappers beside them
// are retained for the literal-expression call sites in `tests`.
#[cfg(any(feature = "cuda", test))]
#[allow(unused_imports)]
pub(crate) use packed::{
    render_cuda_binary_packed, render_cuda_binary_packed_body, render_cuda_unary_packed,
    render_cuda_unary_packed_body,
};
#[cfg(any(feature = "cuda", test))]
pub(crate) use reduction::render_cuda_reduction;
#[cfg(any(feature = "cuda", test))]
#[allow(unused_imports)]
pub(crate) use scalar::{
    render_cuda_binary_for_layout, render_cuda_binary_for_layout_body,
    render_cuda_unary_for_layout, render_cuda_unary_for_layout_body,
};
// The IR-lowered counterparts of the two entry points above. They render the
// same templates through the same cache keys; only the producer of the body
// expression differs. See `crate::codegen::fragment`.
#[cfg(any(feature = "cuda", test))]
#[allow(unused_imports)]
pub(crate) use scalar::{lower_binary_body, lower_unary_body};
// Test-only from a non-test build's perspective: every current call site is
// inside `#[cfg(test)]` code (`cuda::ops::elementwise`'s own tests exercise
// it directly rather than through `render_cuda_unary_for_layout`).
#[cfg(any(feature = "cuda", test))]
#[allow(unused_imports)]
pub(crate) use scalar::render_cuda_unary;
#[cfg(any(feature = "cuda", test))]
pub(crate) use types::RenderedKernel;
#[cfg(any(feature = "cuda", test))]
use types::source_scoped_cache_id;
// Test-only: `KernelDType` is otherwise private to `types` (every non-test
// caller goes through `KernelKey`'s own constructors) and reached only by
// `tests`, so a non-test build reports it unused.
#[allow(unused_imports)]
use types::KernelDType;
// `KernelFamily`/`KernelAccess`/`KernelKey` are genuinely `pub` in their
// defining file (reached crate-externally only via `pub(crate) mod kernel`'s
// own cap, same as before the split) so the re-export must match, not
// narrow to `pub(crate)`.
pub use types::{KernelAccess, KernelFamily, KernelKey};

// Test-only: `render_cuda_binary` is otherwise private to `scalar` (nothing
// outside this module calls the unlayouted binary entry point directly) and
// reached only by `tests`, so a non-test build reports it unused.
#[cfg(any(feature = "cuda", test))]
#[allow(unused_imports)]
use scalar::render_cuda_binary;
// `packed`/`reduction`/`normalization` each specialize their own template
// from the same per-dtype scalar spec and identifier check `scalar` builds;
// both are `pub(super)` there for exactly this reach.
#[cfg(any(feature = "cuda", test))]
use scalar::{CudaScalarSpec, validate_identifier};
