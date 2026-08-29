//! Declarative Operation Definition DSL and Builders.
//!
//! Makes defining custom forward and backward tensor operations effortless, generating:
//! - Strongly typed Intermediate Representation (IR) graphs
//! - Automatic symbolic analytical derivatives for backward autograd
//! - Optimized CUDA C++ kernels with vectorization support
//! - Native CPU evaluation functions

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

/// Helper builder for defining custom ternary pointwise operations.
#[must_use]
pub fn define_ternary_custom_op(
    name: &'static str,
    dtype: DTypeId,
    forward_fn: impl Fn(IrExpr, IrExpr, IrExpr) -> IrExpr,
) -> KernelDefinition {
    let a = IrExpr::arg(0);
    let b = IrExpr::arg(1);
    let c = IrExpr::arg(2);
    let expr = forward_fn(a, b, c);
    KernelDefinition::new(name, 3, dtype, expr)
}
