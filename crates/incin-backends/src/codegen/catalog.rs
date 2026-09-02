//! IR definitions for the pointwise operations the CUDA backend launches.
//!
//! `cuda::backend::elementwise` declares each pointwise operation three times:
//! a forward CUDA C literal, a derivative CUDA C literal, and (for binaries) a
//! second derivative literal. Nothing checks that the derivative is the
//! derivative of the forward. They are separate strings that a reader has to
//! differentiate by hand to review, and some of them are long enough that this
//! is not realistic: the `gelu` derivative literal repeats its `tanhf(...)`
//! subterm three times across 180 characters.
//!
//! This module gives those operations a single definition as an `IrExpr` over
//! `Arg(0)` (and `Arg(1)` for binaries). The derivative then comes from
//! `IrExpr::diff`, which cannot disagree with the forward because it is computed
//! from it. `lower_scalar` renders either one into the `{OP}` slot of the
//! templates in `crate::kernel::scalar`, so adopting a definition here changes
//! where the expression text comes from and nothing else about how the kernel
//! is compiled, cached, tuned or launched.
//!
//! Coverage is deliberately partial. An operation appears here only when the IR
//! can express it exactly, either through a dedicated operator or as a
//! composition of operators it already has. The inverse and hyperbolic
//! transcendentals (`asin`, `atan`, `sinh`, `asinh`, `erf`, `tan`) and the
//! rounding family have neither, so they keep their hand-written literals until
//! the vocabulary grows, and `unary_forward` returns `None` for them rather than
//! guessing.

use super::ir::{IrExpr, IrTernaryOp, IrUnaryOp};

/// The forward IR for a unary pointwise operation, keyed by the name the CUDA
/// backend registers it under.
///
/// Returns `None` for an operation the IR cannot express exactly.
#[must_use]
pub fn unary_forward(op_name: &str) -> Option<IrExpr> {
    let x = IrExpr::arg(0);
    Some(match op_name {
        "relu" => x.relu(),
        "neg" => x.neg(),
        "abs" => IrExpr::unary(IrUnaryOp::Abs, x),
        "log" => x.log(),
        "exp" => x.exp(),
        "sqrt" => x.sqrt(),
        "rsqrt" => x.rsqrt(),
        "sin" => IrExpr::unary(IrUnaryOp::Sin, x),
        "cos" => IrExpr::unary(IrUnaryOp::Cos, x),
        "tanh" => x.tanh(),
        "sigmoid" => x.sigmoid(),
        // The backend registers Swish under its own name; it is SiLU.
        "swish" | "silu" => x.silu(),
        "gelu" => x.gelu(),
        "square" => IrExpr::unary(IrUnaryOp::Square, x),
        "reciprocal" => IrExpr::unary(IrUnaryOp::Reciprocal, x),
        "step" => IrExpr::unary(IrUnaryOp::Step, x),
        // The next four have no dedicated `IrUnaryOp`, but each is exactly a
        // composition of operators the IR already has, so they need no new
        // vocabulary and their derivatives still come from `diff`.
        //
        // `elu` is a two-way select. The backend spells its condition `x >= 0`
        // and `Select` tests `> 0`; both branches evaluate to `0` at `x == 0`,
        // so the two agree everywhere.
        "elu" => IrExpr::ternary(
            IrTernaryOp::Select,
            x.clone(),
            x.clone(),
            x.exp().sub(IrExpr::constant(1.0)),
        ),
        // mish(x) = x * tanh(softplus(x)). The backend writes softplus with
        // `log1pf`; `log(1 + exp(x))` is the same function, and the difference
        // only matters for `exp(x)` near the edge of representable precision.
        "mish" => x
            .clone()
            .mul(IrExpr::constant(1.0).add(x.exp()).log().tanh()),
        // Change of base. Writing these as a scaled natural log rather than as
        // `log2f`/`log10f` keeps the derivative exact: `diff` already knows
        // `d/dx ln(x) = 1/x`, so the constant simply carries through.
        "log2" => x.log().mul(IrExpr::constant(core::f64::consts::LOG2_E)),
        "log10" => x.log().mul(IrExpr::constant(core::f64::consts::LOG10_E)),
        // `sign` is the three-way signum. `Select` is two-way and resolves the
        // tie at zero to the false branch, so it is spelled as nested selects
        // rather than reused from the `Abs` derivative.
        "sign" => IrExpr::ternary(
            IrTernaryOp::Select,
            x.clone(),
            IrExpr::constant(1.0),
            IrExpr::ternary(
                IrTernaryOp::Select,
                x.neg(),
                IrExpr::constant(-1.0),
                IrExpr::constant(0.0),
            ),
        ),
        _ => return None,
    })
}

/// The forward IR for a binary pointwise operation.
///
/// Returns `None` for an operation the IR cannot express exactly.
#[must_use]
pub fn binary_forward(op_name: &str) -> Option<IrExpr> {
    let a = IrExpr::arg(0);
    let b = IrExpr::arg(1);
    Some(match op_name {
        "add" => a.add(b),
        "sub" => a.sub(b),
        "mul" => a.mul(b),
        "div" => a.div(b),
        "maximum" => IrExpr::binary(super::ir::IrBinaryOp::Max, a, b),
        "minimum" => IrExpr::binary(super::ir::IrBinaryOp::Min, a, b),
        "abs_diff" => IrExpr::unary(IrUnaryOp::Abs, a.sub(b)),
        _ => return None,
    })
}

/// The fused backward expression for a unary operation: `grad_out * f'(x)`.
///
/// The backend currently computes a unary backward in two launches -- one kernel
/// evaluates `f'(x)` into a fresh full-size buffer, a second multiplies that
/// buffer by the incoming gradient. Both are pointwise over the same shape, so
/// there is no reason for the intermediate to exist. Differentiating the forward
/// symbolically and multiplying inside the IR collapses them into a single
/// binary kernel, which removes one launch and one allocation of `numel`
/// elements per operation per backward pass.
///
/// The result is expressed over two operands: `Arg(0)` is the incoming gradient
/// and `Arg(1)` is the forward input, matching the `a`/`b` naming the binary
/// templates load.
///
/// Returns `None` for an operation the IR cannot express exactly.
#[must_use]
pub fn unary_fused_backward(op_name: &str) -> Option<IrExpr> {
    let forward = unary_forward(op_name)?;
    // `diff` is taken with respect to the forward's own `Arg(0)`, then shifted
    // to `Arg(1)` so `Arg(0)` is free for the incoming gradient.
    let derivative = forward.diff(0).remap_args(&|index| index + 1);
    Some(IrExpr::arg(0).mul(derivative))
}

/// Every unary operation name this module defines, for exhaustive testing.
pub const UNARY_OPS: &[&str] = &[
    "relu",
    "neg",
    "abs",
    "log",
    "exp",
    "sqrt",
    "rsqrt",
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "swish",
    "gelu",
    "square",
    "reciprocal",
    "step",
    "sign",
    "elu",
    "mish",
    "log2",
    "log10",
];

/// Every binary operation name this module defines, for exhaustive testing.
pub const BINARY_OPS: &[&str] = &["add", "sub", "mul", "div", "maximum", "minimum", "abs_diff"];
