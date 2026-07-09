//! Operator implementations for `NativeBackend<T, D>`.
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
//! * `loss` — full `LossOps` impl: real (composed) `mse_loss`;
//!   `unimplemented!()` stubs for `l1_loss`/`bce_with_logits_loss`/
//!   `cross_entropy_loss`.
//! * `stubs` — `ModuleOps` only (all methods typed
//!   `Error::UnsupportedBackendOperation`).

pub mod elementwise;
pub mod loss;
pub mod matmul;
pub mod reduce;
pub mod shape_ops;
pub mod stubs;
