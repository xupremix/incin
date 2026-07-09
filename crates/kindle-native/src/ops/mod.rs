//! Operator implementations for `NativeBackend<T, D>`.
//!
//! This plan populates `elementwise` (`NumericOps::{add,sub,mul,div}` and
//! `FloatOps::{add_scalar_float, mul_scalar_float}`). `TensorOps::matmul`,
//! `ReductionOps`, `ModuleOps`, and the rest of `FloatOps`/`LossOps` land in
//! later plans.

pub mod elementwise;
pub mod stubs;
