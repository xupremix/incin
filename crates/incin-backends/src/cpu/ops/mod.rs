//! Operator implementations for `CpuBackendImpl<D>`.
//!
//! * `elementwise` — `::{add,sub,mul,div}` and
//!   `::{add_scalar_float,mul_scalar_float,relu,gelu,…}` (all
//!   unary-kernel stubs included).
//! * `matmul` — naive stride-aware 2D `matmul_impl` (plain function, called
//!   by `shape_ops`'s `TensorOps::matmul` method).
//! * `shape_ops` — full `TensorOps` impl: real
//!   `reshape`/`transpose`/`broadcast_as`/`matmul`/`float_to_scalar`/
//!   `float_to_vec1`, typed stubs for the rest.
//! * `reduce` — concrete reduction kernels
//!   `sum_all`/`mean_all`/`sum_dim`/`sum_keepdim`, typed stubs for the rest.
//! * `loss` — concrete loss helpers: real `mse_loss`/`l1_loss`/
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

/// `conv`.
pub mod conv;
/// `elementwise`.
pub mod elementwise;
pub(crate) mod elementwise_kernel;
/// `embedding`.
pub mod embedding;
/// `loss`.
pub mod loss;
/// `matmul`.
pub mod matmul;
/// `norm`.
pub mod norm;
/// `pool`.
pub mod pool;
/// `quant`.
pub mod quant;
/// `reduce`.
pub mod reduce;
/// `shape_ops`.
pub mod shape_ops;
