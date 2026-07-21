//! Operator implementations for `CpuBackend<T, D>`.
//!
//! * `elementwise` — `NumericOps::{add,sub,mul,div}` and
//!   `FloatOps::{add_scalar_float,mul_scalar_float,relu,gelu,…}` (all
//!   unary-kernel stubs included).
//! * `matmul` — naive stride-aware 2D `matmul_impl` (plain function, called
//!   by `shape_ops`'s `TensorOps::matmul` method).
//! * `shape_ops` — full `TensorOps` impl: real
//!   `reshape`/`transpose`/`broadcast_as`/`matmul`/`float_to_scalar`/
//!   `float_to_vec1`, typed stubs for the rest.
//! * `reduce` — full `ReductionOps` impl: real
//!   `sum_all`/`mean_all`/`sum_dim`/`sum_keepdim`, typed stubs for the rest.
//! * `loss` — full `LossOps` impl: real `mse_loss`/`l1_loss`/
//!   `bce_with_logits_loss`/`cross_entropy_loss` (all 4 methods real,
//!   no stubs remain — Plan 04-01/02).
//! * `norm` — free-function helpers: `layer_norm_impl`/`batch_norm_impl`
//!   (shared by `module`'s trait methods).
//! * `embedding` — free-function helper: `embedding_impl` (per-row gather
//!   forward + scatter-add backward, shared by `module`'s `embedding` method).
//! * `conv` — free-function helpers: `conv1d_impl`/`conv2d_impl` (im2col +
//!   `matmul::batched_matmul_impl` forward, hand-composed col2im-fold
//!   backward, shared by `module`'s `conv1d`/`conv2d` methods — Plan 04-05)
//!   and `conv_transpose2d_impl` (reuses `conv2d_impl`'s own internal
//!   `col2im_2d` fold subroutine directly as its forward, per the standard
//!   conv-transpose-is-conv2d's-backward-data equivalence — Plan 04-07).
//! * `pool` — free-function helpers: `max_pool2d_impl`/`avg_pool2d_impl`/
//!   `adaptive_avg_pool2d_impl` (2D sliding-window max/mean reductions,
//!   generalizing `reduce.rs`'s `max_axis_with_indices`/`scatter_axis_grad`
//!   pattern to 2D with the overlap-safe `+=` accumulation fix pooling's
//!   overlapping windows require — Plan 04-06). `adaptive_avg_pool2d_impl`
//!   computes per-output-position variable window boundaries rather than a
//!   fixed kernel_size/stride derivation.
//! * `module` — `ModuleOps`: ALL NINE methods are real as of Plan 04-07
//!   (`layer_norm`/`batch_norm`/`embedding`/`conv1d`/`conv2d`/
//!   `conv_transpose2d`/`max_pool2d`/`avg_pool2d`/`adaptive_avg_pool2d`) —
//!   zero `Error::UnsupportedBackendOperation` stubs remain.

/// Core abstraction for `conv` within the Kindle framework..
pub mod conv;
/// Core abstraction for `elementwise` within the Kindle framework..
pub mod elementwise;
/// Core abstraction for `embedding` within the Kindle framework..
pub mod embedding;
/// Core abstraction for `loss` within the Kindle framework..
pub mod loss;
/// Core abstraction for `matmul` within the Kindle framework..
pub mod matmul;
/// Core abstraction for `module` within the Kindle framework..
pub mod module;
/// Core abstraction for `norm` within the Kindle framework..
pub mod norm;
/// Core abstraction for `optimizer` within the Kindle framework..
pub mod optimizer;
/// Core abstraction for `pool` within the Kindle framework..
pub mod pool;
/// Core abstraction for `quant` within the Kindle framework..
pub mod quant;
/// Core abstraction for `reduce` within the Kindle framework..
pub mod reduce;
/// Core abstraction for `shape_ops` within the Kindle framework..
pub mod shape_ops;
