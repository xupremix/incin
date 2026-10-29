//! Declarative builders for [`KernelDefinition`].
//!
//! Makes defining a custom pointwise operation from a closure over
//! [`IrExpr`] effortless, producing:
//! - A strongly typed Intermediate Representation (IR) forward graph
//! - Symbolically derived backward derivatives, one per input, via
//!   [`KernelDefinition::new`]
//!
//! Rendering that definition to CUDA C lives in
//! [`KernelDefinition`]'s `render_*` methods and
//! execution lives in [`jit`](super::jit): [`CpuJitKernel`](super::jit::CpuJitKernel)
//! as a host `f64` tree-walking reference, `CudaJitKernel` through the NVRTC
//! dispatcher. This module itself generates no kernels and no executors.

use super::ir::{IrExpr, KernelDefinition};
use incin_core::tensor::dtype::DTypeId;

/// Helper builder for defining custom unary pointwise operations.
#[must_use]
pub fn define_unary_custom_op(
    name: &'static str,
    dtype: DTypeId,
    forward_fn: impl Fn(IrExpr) -> IrExpr,
) -> KernelDefinition {
    let x = IrExpr::arg(0);
    let expr = forward_fn(x);
    KernelDefinition::new(name, 1, dtype, expr)
}

/// Helper builder for defining custom binary pointwise operations.
#[must_use]
pub fn define_binary_custom_op(
    name: &'static str,
    dtype: DTypeId,
    forward_fn: impl Fn(IrExpr, IrExpr) -> IrExpr,
) -> KernelDefinition {
    let a = IrExpr::arg(0);
    let b = IrExpr::arg(1);
    let expr = forward_fn(a, b);
    KernelDefinition::new(name, 2, dtype, expr)
}
