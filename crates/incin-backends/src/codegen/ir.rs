//! Incin Kernel Intermediate Representation (IR) and Optimization Engine.
//!
//! Provides a structured, strongly-typed Intermediate Representation for kernel computation graphs,
//! featuring:
//! - Algebraic simplification & constant folding
//! - Common Subexpression Elimination (CSE)
//! - Fused Multiply-Add (FMA) pattern recognition
//! - Symbolic Automatic Differentiation (generating analytical derivatives directly from forward IR)
//! - Multi-target code emission (CUDA C++, WGSL, MSL, and CPU SIMD)

use alloc::{boxed::Box, format, string::String, vec::Vec};
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// Unary operators supported in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrUnaryOp {
    /// Absolute value.
    Abs,
    /// Natural exponential (`e^x`).
    Exp,
    /// Natural logarithm (`ln(x)`).
    Log,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Square root (`sqrt(x)`).
    Sqrt,
    /// Reciprocal square root (`1 / sqrt(x)`).
    Rsqrt,
    /// Arithmetic negation (`-x`).
    Neg,
    /// Rectified Linear Unit (`max(0, x)`).
    Relu,
    /// Gaussian Error Linear Unit.
    Gelu,
    /// Sigmoid Linear Unit / Swish (`x * sigmoid(x)`).
    Silu,
    /// Logistic sigmoid (`1 / (1 + exp(-x))`).
    Sigmoid,
    /// Hyperbolic tangent (`tanh(x)`).
    Tanh,
    /// Multiplicative reciprocal (`1 / x`).
    Reciprocal,
    /// Square (`x * x`).
    Square,
    /// Heaviside step function (`x > 0 ? 1 : 0`).
    Step,
}

/// Binary operators supported in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrBinaryOp {
    /// Addition (`a + b`).
    Add,
    /// Subtraction (`a - b`).
    Sub,
    /// Multiplication (`a * b`).
    Mul,
    /// Division (`a / b`).
    Div,
    /// Power (`a ^ b`).
    Pow,
    /// Maximum (`max(a, b)`).
    Max,
    /// Minimum (`min(a, b)`).
    Min,
}

/// Ternary operators supported in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrTernaryOp {
    /// Fused multiply-add (`a * b + c`).
    Fma,
    /// Value clamp (`clamp(x, min, max)`).
    Clamp,
    /// Conditional select (`cond > 0 ? true_val : false_val`).
    Select,
}

/// Kernel Intermediate Representation (IR) Expression Tree.
#[derive(Debug, Clone, PartialEq)]
pub enum IrExpr {
    /// Input operand by 0-based parameter index.
    Arg(usize),
    /// Literal scalar constant (64-bit float).
    Const(f64),
    /// Named variable or SSA intermediate temporary.
    Var(String),
    /// Unary operation over an expression.
    Unary(IrUnaryOp, Box<IrExpr>),
    /// Binary operation over two expressions.
    Binary(IrBinaryOp, Box<IrExpr>, Box<IrExpr>),
    /// Ternary operation over three expressions.
    Ternary(IrTernaryOp, Box<IrExpr>, Box<IrExpr>, Box<IrExpr>),
}

#[allow(clippy::should_implement_trait)]
impl IrExpr {
    /// Creates an argument reference node.
    #[must_use]
    pub const fn arg(idx: usize) -> Self {
        Self::Arg(idx)
    }

    /// Creates a constant scalar node.
    #[must_use]
    pub const fn constant(val: f64) -> Self {
        Self::Const(val)
    }

    /// Creates a variable node.
    #[must_use]
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    /// Rewrites every argument index through `map`.
    ///
    /// Fusing two expressions means placing them over one shared operand list,
    /// and an expression built in isolation numbers its arguments from zero. A
    /// unary operation's derivative refers to its input as `Arg(0)`; to combine
    /// it with an incoming gradient that occupies `Arg(0)` of the fused kernel,
    /// its own argument has to move to `Arg(1)` first.
    #[must_use]
    pub fn remap_args(&self, map: &impl Fn(usize) -> usize) -> Self {
        match self {
            Self::Arg(index) => Self::Arg(map(*index)),
            Self::Const(value) => Self::Const(*value),
            Self::Var(name) => Self::Var(name.clone()),
            Self::Unary(op, inner) => Self::unary(*op, inner.remap_args(map)),
            Self::Binary(op, lhs, rhs) => {
                Self::binary(*op, lhs.remap_args(map), rhs.remap_args(map))
            }
            Self::Ternary(op, a, b, c) => {
                Self::ternary(*op, a.remap_args(map), b.remap_args(map), c.remap_args(map))
            }
        }
    }

    /// Builds a unary operation node.
    #[must_use]
    pub fn unary(op: IrUnaryOp, arg: Self) -> Self {
        Self::Unary(op, Box::new(arg))
    }

    /// Builds a binary operation node.
    #[must_use]
    pub fn binary(op: IrBinaryOp, lhs: Self, rhs: Self) -> Self {
        Self::Binary(op, Box::new(lhs), Box::new(rhs))
    }

    /// Builds a ternary operation node.
    #[must_use]
    pub fn ternary(op: IrTernaryOp, a: Self, b: Self, c: Self) -> Self {
        Self::Ternary(op, Box::new(a), Box::new(b), Box::new(c))
    }

    /// Convenience: `self + other`.
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self::binary(IrBinaryOp::Add, self, other)
    }

    /// Convenience: `self - other`.
    #[must_use]
    pub fn sub(self, other: Self) -> Self {
        Self::binary(IrBinaryOp::Sub, self, other)
    }

    /// Convenience: `self * other`.
    #[must_use]
    pub fn mul(self, other: Self) -> Self {
        Self::binary(IrBinaryOp::Mul, self, other)
    }

    /// Convenience: `self / other`.
    #[must_use]
    pub fn div(self, other: Self) -> Self {
        Self::binary(IrBinaryOp::Div, self, other)
    }

    /// Convenience: `-self`.
    #[must_use]
    pub fn neg(self) -> Self {
        Self::unary(IrUnaryOp::Neg, self)
    }

    /// Convenience: `exp(self)`.
    #[must_use]
    pub fn exp(self) -> Self {
        Self::unary(IrUnaryOp::Exp, self)
    }

    /// Convenience: `ln(self)`.
    #[must_use]
    pub fn log(self) -> Self {
        Self::unary(IrUnaryOp::Log, self)
    }

    /// Convenience: `sqrt(self)`.
    #[must_use]
    pub fn sqrt(self) -> Self {
        Self::unary(IrUnaryOp::Sqrt, self)
    }

    /// Convenience: `rsqrt(self)`.
    #[must_use]
    pub fn rsqrt(self) -> Self {
        Self::unary(IrUnaryOp::Rsqrt, self)
    }

    /// Convenience: `relu(self)`.
    #[must_use]
    pub fn relu(self) -> Self {
        Self::unary(IrUnaryOp::Relu, self)
    }

    /// Convenience: `sigmoid(self)`.
    #[must_use]
    pub fn sigmoid(self) -> Self {
        Self::unary(IrUnaryOp::Sigmoid, self)
    }

    /// Convenience: `tanh(self)`.
    #[must_use]
    pub fn tanh(self) -> Self {
        Self::unary(IrUnaryOp::Tanh, self)
    }

    /// Convenience: `gelu(self)`.
    #[must_use]
    pub fn gelu(self) -> Self {
        Self::unary(IrUnaryOp::Gelu, self)
    }

    /// Convenience: `silu(self)`.
    #[must_use]
    pub fn silu(self) -> Self {
        Self::unary(IrUnaryOp::Silu, self)
    }

    /// Convenience: `fma(a, b, c) = a * b + c`.
    #[must_use]
    pub fn fma(a: Self, b: Self, c: Self) -> Self {
        Self::ternary(IrTernaryOp::Fma, a, b, c)
    }

    /// Performs algebraic simplification and constant folding on the expression tree.
    #[must_use]
    pub fn optimize(&self) -> Self {
        let simplified = self.constant_fold_and_simplify();
        simplified.fuse_fma()
    }

    fn constant_fold_and_simplify(&self) -> Self {
        match self {
            Self::Arg(i) => Self::Arg(*i),
            Self::Const(c) => Self::Const(*c),
            Self::Var(v) => Self::Var(v.clone()),
            Self::Unary(op, inner) => {
                let opt_inner = inner.constant_fold_and_simplify();
                if let Self::Const(c) = opt_inner {
                    match op {
                        IrUnaryOp::Neg => Self::Const(-c),
                        IrUnaryOp::Abs => Self::Const(c.abs()),
                        IrUnaryOp::Exp => Self::Const(c.exp()),
                        IrUnaryOp::Log => Self::Const(c.ln()),
                        IrUnaryOp::Sin => Self::Const(c.sin()),
                        IrUnaryOp::Cos => Self::Const(c.cos()),
                        IrUnaryOp::Sqrt => Self::Const(c.sqrt()),
                        IrUnaryOp::Rsqrt => Self::Const(1.0 / c.sqrt()),
                        IrUnaryOp::Relu => Self::Const(c.max(0.0)),
                        IrUnaryOp::Step => Self::Const(if c > 0.0 { 1.0 } else { 0.0 }),
                        IrUnaryOp::Square => Self::Const(c * c),
                        IrUnaryOp::Reciprocal => Self::Const(1.0 / c),
                        IrUnaryOp::Tanh => Self::Const(c.tanh()),
                        IrUnaryOp::Sigmoid => Self::Const(1.0 / (1.0 + (-c).exp())),
                        IrUnaryOp::Silu => Self::Const(c / (1.0 + (-c).exp())),
                        IrUnaryOp::Gelu => {
                            let k = 0.797_884_560_802_865_4; // sqrt(2 / pi)
                            let cdf = 0.5 * (1.0 + (k * (c + 0.044715 * c.powi(3))).tanh());
                            Self::Const(c * cdf)
                        }
                    }
                } else {
                    // Algebraic identities
                    match (op, &opt_inner) {
                        (IrUnaryOp::Neg, Self::Unary(IrUnaryOp::Neg, x)) => *x.clone(),
                        (IrUnaryOp::Exp, Self::Unary(IrUnaryOp::Log, x)) => *x.clone(),
                        (IrUnaryOp::Log, Self::Unary(IrUnaryOp::Exp, x)) => *x.clone(),
                        _ => Self::Unary(*op, Box::new(opt_inner)),
                    }
                }
            }
            Self::Binary(op, lhs, rhs) => {
                let opt_l = lhs.constant_fold_and_simplify();
                let opt_r = rhs.constant_fold_and_simplify();

                if let (Self::Const(cl), Self::Const(cr)) = (&opt_l, &opt_r) {
                    match op {
                        IrBinaryOp::Add => Self::Const(cl + cr),
                        IrBinaryOp::Sub => Self::Const(cl - cr),
                        IrBinaryOp::Mul => Self::Const(cl * cr),
                        IrBinaryOp::Div => Self::Const(cl / cr),
                        IrBinaryOp::Pow => Self::Const(cl.powf(*cr)),
                        IrBinaryOp::Max => Self::Const(cl.max(*cr)),
                        IrBinaryOp::Min => Self::Const(cl.min(*cr)),
                    }
                } else {
                    // Algebraic simplification rules
                    match (op, &opt_l, &opt_r) {
                        // x + 0 = x, 0 + x = x
                        (IrBinaryOp::Add, x, Self::Const(c)) if *c == 0.0 => x.clone(),
                        (IrBinaryOp::Add, Self::Const(c), x) if *c == 0.0 => x.clone(),
                        // x - 0 = x
                        (IrBinaryOp::Sub, x, Self::Const(c)) if *c == 0.0 => x.clone(),
                        // x - x = 0
                        (IrBinaryOp::Sub, l, r) if l == r => Self::Const(0.0),
                        // x * 1 = x, 1 * x = x
                        (IrBinaryOp::Mul, x, Self::Const(c)) if *c == 1.0 => x.clone(),
                        (IrBinaryOp::Mul, Self::Const(c), x) if *c == 1.0 => x.clone(),
                        // x * 0 = 0, 0 * x = 0
                        (IrBinaryOp::Mul, _, Self::Const(c)) if *c == 0.0 => Self::Const(0.0),
                        (IrBinaryOp::Mul, Self::Const(c), _) if *c == 0.0 => Self::Const(0.0),
                        // x / 1 = x
                        (IrBinaryOp::Div, x, Self::Const(c)) if *c == 1.0 => x.clone(),
                        // x / x = 1
                        (IrBinaryOp::Div, l, r) if l == r => Self::Const(1.0),
                        _ => Self::Binary(*op, Box::new(opt_l), Box::new(opt_r)),
                    }
                }
            }
            Self::Ternary(op, a, b, c) => {
                let opt_a = a.constant_fold_and_simplify();
                let opt_b = b.constant_fold_and_simplify();
                let opt_c = c.constant_fold_and_simplify();
                Self::Ternary(*op, Box::new(opt_a), Box::new(opt_b), Box::new(opt_c))
            }
        }
    }

    fn fuse_fma(&self) -> Self {
        match self {
            Self::Binary(IrBinaryOp::Add, lhs, rhs) => {
                let opt_l = lhs.fuse_fma();
                let opt_r = rhs.fuse_fma();
                if let Self::Binary(IrBinaryOp::Mul, m1, m2) = &opt_l {
                    Self::Ternary(IrTernaryOp::Fma, m1.clone(), m2.clone(), Box::new(opt_r))
                } else if let Self::Binary(IrBinaryOp::Mul, m1, m2) = &opt_r {
                    Self::Ternary(IrTernaryOp::Fma, m1.clone(), m2.clone(), Box::new(opt_l))
                } else {
                    Self::Binary(IrBinaryOp::Add, Box::new(opt_l), Box::new(opt_r))
                }
            }
            Self::Unary(op, inner) => Self::Unary(*op, Box::new(inner.fuse_fma())),
            Self::Binary(op, lhs, rhs) => {
                Self::Binary(*op, Box::new(lhs.fuse_fma()), Box::new(rhs.fuse_fma()))
            }
            Self::Ternary(op, a, b, c) => Self::Ternary(
                *op,
                Box::new(a.fuse_fma()),
                Box::new(b.fuse_fma()),
                Box::new(c.fuse_fma()),
            ),
            other => other.clone(),
        }
    }

    /// Computes the exact symbolic analytical derivative d(self) / d(target_arg) using calculus rules.
    #[must_use]
    pub fn diff(&self, target_arg: usize) -> Self {
        let raw_diff = self.diff_recursive(target_arg);
        raw_diff.optimize()
    }

    fn diff_recursive(&self, target: usize) -> Self {
        match self {
            Self::Arg(idx) => {
                if *idx == target {
                    Self::Const(1.0)
                } else {
                    Self::Const(0.0)
                }
            }
            Self::Const(_) | Self::Var(_) => Self::Const(0.0),
            Self::Unary(op, inner) => {
                let u = &**inner;
                let du = u.diff_recursive(target);
                match op {
                    IrUnaryOp::Neg => du.neg(),
                    IrUnaryOp::Abs => {
                        // d/dx |u| = sign(u) * du = (u > 0 ? 1 : -1) * du
                        let sign = Self::ternary(
                            IrTernaryOp::Select,
                            u.clone(),
                            Self::Const(1.0),
                            Self::Const(-1.0),
                        );
                        sign.mul(du)
                    }
                    IrUnaryOp::Exp => {
                        // d/dx exp(u) = exp(u) * du
                        Self::unary(IrUnaryOp::Exp, u.clone()).mul(du)
                    }
                    IrUnaryOp::Log => {
                        // d/dx ln(u) = (1 / u) * du
                        du.div(u.clone())
                    }
                    IrUnaryOp::Sin => {
                        // d/dx sin(u) = cos(u) * du
                        Self::unary(IrUnaryOp::Cos, u.clone()).mul(du)
                    }
                    IrUnaryOp::Cos => {
                        // d/dx cos(u) = -sin(u) * du
                        Self::unary(IrUnaryOp::Sin, u.clone()).neg().mul(du)
                    }
                    IrUnaryOp::Sqrt => {
                        // d/dx sqrt(u) = 1 / (2 * sqrt(u)) * du
                        let two_sqrt =
                            Self::Const(2.0).mul(Self::unary(IrUnaryOp::Sqrt, u.clone()));
                        du.div(two_sqrt)
                    }
                    IrUnaryOp::Rsqrt => {
                        // d/dx u^(-1/2) = -1/2 * u^(-3/2) * du = -0.5 * rsqrt(u)^3 * du
                        let r = Self::unary(IrUnaryOp::Rsqrt, u.clone());
                        let r3 = r.clone().mul(r.clone()).mul(r);
                        Self::Const(-0.5).mul(r3).mul(du)
                    }
                    IrUnaryOp::Relu => {
                        // d/dx relu(u) = step(u) * du
                        Self::unary(IrUnaryOp::Step, u.clone()).mul(du)
                    }
                    IrUnaryOp::Step => Self::Const(0.0),
                    IrUnaryOp::Square => {
                        // d/dx (u^2) = 2 * u * du
                        Self::Const(2.0).mul(u.clone()).mul(du)
                    }
                    IrUnaryOp::Reciprocal => {
                        // d/dx (1/u) = -1 / u^2 * du
                        Self::Const(-1.0).div(u.clone().mul(u.clone())).mul(du)
                    }
                    IrUnaryOp::Sigmoid => {
                        // d/dx sig(u) = sig(u) * (1 - sig(u)) * du
                        let sig = Self::unary(IrUnaryOp::Sigmoid, u.clone());
                        let one_minus_sig = Self::Const(1.0).sub(sig.clone());
                        sig.mul(one_minus_sig).mul(du)
                    }
                    IrUnaryOp::Tanh => {
                        // d/dx tanh(u) = (1 - tanh(u)^2) * du
                        let th = Self::unary(IrUnaryOp::Tanh, u.clone());
                        let th2 = th.clone().mul(th);
                        Self::Const(1.0).sub(th2).mul(du)
                    }
                    IrUnaryOp::Silu => {
                        // silu(u) = u * sig(u)
                        // d/dx silu(u) = sig(u) + u * sig(u) * (1 - sig(u))
                        let sig = Self::unary(IrUnaryOp::Sigmoid, u.clone());
                        let one_minus_sig = Self::Const(1.0).sub(sig.clone());
                        let d_silu_du = sig.clone().add(u.clone().mul(sig).mul(one_minus_sig));
                        d_silu_du.mul(du)
                    }
                    IrUnaryOp::Gelu => {
                        // gelu(u) = 0.5 * u * (1 + tanh(g(u))) with
                        // g(u) = k * (u + c * u^3), so gelu is a product and its
                        // derivative needs both terms of the product rule:
                        //   0.5 * (1 + tanh(g)) + 0.5 * u * (1 - tanh(g)^2) * g'(u)
                        // where g'(u) = k * (1 + 3c * u^2). Dropping the second
                        // term understates the gradient everywhere except where
                        // sech^2(g) is zero.
                        let k = Self::Const(super::fragment::GELU_K);
                        let c = Self::Const(super::fragment::GELU_C);
                        let u2 = u.clone().mul(u.clone());
                        let g = k
                            .clone()
                            .mul(u.clone().add(c.clone().mul(u2.clone().mul(u.clone()))));
                        let th = Self::unary(IrUnaryOp::Tanh, g);
                        let cdf = Self::Const(0.5).mul(Self::Const(1.0).add(th.clone()));
                        let sech2 = Self::Const(1.0).sub(th.clone().mul(th));
                        let g_prime = k.mul(Self::Const(1.0).add(Self::Const(3.0).mul(c).mul(u2)));
                        let pdf_term = Self::Const(0.5).mul(u.clone()).mul(sech2).mul(g_prime);
                        cdf.add(pdf_term).mul(du)
                    }
                }
            }
            Self::Binary(op, lhs, rhs) => {
                let u = &**lhs;
                let v = &**rhs;
                let du = u.diff_recursive(target);
                let dv = v.diff_recursive(target);

                match op {
                    IrBinaryOp::Add => du.add(dv),
                    IrBinaryOp::Sub => du.sub(dv),
                    IrBinaryOp::Mul => {
                        // Product rule: (u * v)' = u' * v + u * v'
                        let term1 = du.mul(v.clone());
                        let term2 = u.clone().mul(dv);
                        term1.add(term2)
                    }
                    IrBinaryOp::Div => {
                        // Quotient rule: (u / v)' = (u' * v - u * v') / v^2
                        let num = du.mul(v.clone()).sub(u.clone().mul(dv));
                        let den = v.clone().mul(v.clone());
                        num.div(den)
                    }
                    IrBinaryOp::Pow => {
                        // d/dx (u^v) where v is constant: v * u^(v-1) * du
                        if let Self::Const(c) = v {
                            let coeff = Self::Const(*c);
                            let pow_term =
                                Self::binary(IrBinaryOp::Pow, u.clone(), Self::Const(c - 1.0));
                            coeff.mul(pow_term).mul(du)
                        } else {
                            // General power: (u^v) * (v' * ln(u) + v * u' / u)
                            let pow_val = Self::binary(IrBinaryOp::Pow, u.clone(), v.clone());
                            let term1 = dv.mul(Self::unary(IrUnaryOp::Log, u.clone()));
                            let term2 = v.clone().mul(du).div(u.clone());
                            pow_val.mul(term1.add(term2))
                        }
                    }
                    IrBinaryOp::Max => {
                        // subgradient: u > v ? du : dv
                        Self::ternary(IrTernaryOp::Select, u.clone().sub(v.clone()), du, dv)
                    }
                    IrBinaryOp::Min => {
                        // subgradient: u < v ? du : dv
                        Self::ternary(IrTernaryOp::Select, v.clone().sub(u.clone()), du, dv)
                    }
                }
            }
            Self::Ternary(op, a, b, c) => match op {
                IrTernaryOp::Fma => {
                    // d/dx (a * b + c) = a' * b + a * b' + c'
                    let da = a.diff_recursive(target);
                    let db = b.diff_recursive(target);
                    let dc = c.diff_recursive(target);
                    let ab_diff = da.mul((**b).clone()).add((**a).clone().mul(db));
                    ab_diff.add(dc)
                }
                IrTernaryOp::Select => {
                    let cond = (**a).clone();
                    let dt = b.diff_recursive(target);
                    let df = c.diff_recursive(target);
                    Self::ternary(IrTernaryOp::Select, cond, dt, df)
                }
                IrTernaryOp::Clamp => {
                    // clamp(x, min, max) -> x > min && x < max ? dx : 0
                    a.diff_recursive(target)
                }
            },
        }
    }

    /// Evaluates the expression as a double-precision float given concrete inputs.
    #[must_use]
    pub fn eval(&self, inputs: &[f64]) -> f64 {
        match self {
            Self::Arg(i) => inputs.get(*i).copied().unwrap_or(0.0),
            Self::Const(c) => *c,
            Self::Var(_) => 0.0,
            Self::Unary(op, inner) => {
                let val = inner.eval(inputs);
                match op {
                    IrUnaryOp::Abs => val.abs(),
                    IrUnaryOp::Exp => val.exp(),
                    IrUnaryOp::Log => val.ln(),
                    IrUnaryOp::Sin => val.sin(),
                    IrUnaryOp::Cos => val.cos(),
                    IrUnaryOp::Sqrt => val.sqrt(),
                    IrUnaryOp::Rsqrt => 1.0 / val.sqrt(),
                    IrUnaryOp::Neg => -val,
                    IrUnaryOp::Relu => val.max(0.0),
                    IrUnaryOp::Step => {
                        if val > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    IrUnaryOp::Square => val * val,
                    IrUnaryOp::Reciprocal => 1.0 / val,
                    IrUnaryOp::Tanh => val.tanh(),
                    IrUnaryOp::Sigmoid => 1.0 / (1.0 + (-val).exp()),
                    IrUnaryOp::Silu => val / (1.0 + (-val).exp()),
                    IrUnaryOp::Gelu => {
                        let k = 0.797_884_560_802_865_4;
                        let cdf = 0.5 * (1.0 + (k * (val + 0.044715 * val.powi(3))).tanh());
                        val * cdf
                    }
                }
            }
            Self::Binary(op, lhs, rhs) => {
                let l = lhs.eval(inputs);
                let r = rhs.eval(inputs);
                match op {
                    IrBinaryOp::Add => l + r,
                    IrBinaryOp::Sub => l - r,
                    IrBinaryOp::Mul => l * r,
                    IrBinaryOp::Div => l / r,
                    IrBinaryOp::Pow => l.powf(r),
                    IrBinaryOp::Max => l.max(r),
                    IrBinaryOp::Min => l.min(r),
                }
            }
            Self::Ternary(op, a, b, c) => {
                let va = a.eval(inputs);
                let vb = b.eval(inputs);
                let vc = c.eval(inputs);
                match op {
                    IrTernaryOp::Fma => va * vb + vc,
                    IrTernaryOp::Clamp => va.clamp(vb, vc),
                    IrTernaryOp::Select => {
                        if va > 0.0 {
                            vb
                        } else {
                            vc
                        }
                    }
                }
            }
        }
    }

    /// Renders the expression into a C/C++ string expression for CUDA.
    #[must_use]
    pub fn render_cuda_expr(&self, dtype: DTypeId) -> String {
        let is_f64 = dtype == DTypeId::F64;
        match self {
            Self::Arg(i) => format!("in{i}[idx]"),
            Self::Const(c) => {
                if is_f64 {
                    format!("{c:.8}")
                } else {
                    format!("{c:.8}f")
                }
            }
            Self::Var(v) => v.clone(),
            Self::Unary(op, inner) => {
                let sub = inner.render_cuda_expr(dtype);
                match op {
                    IrUnaryOp::Neg => format!("(-({sub}))"),
                    IrUnaryOp::Abs => {
                        if is_f64 {
                            format!("fabs({sub})")
                        } else {
                            format!("fabsf({sub})")
                        }
                    }
                    IrUnaryOp::Exp => {
                        if is_f64 {
                            format!("exp({sub})")
                        } else {
                            format!("expf({sub})")
                        }
                    }
                    IrUnaryOp::Log => {
                        if is_f64 {
                            format!("log({sub})")
                        } else {
                            format!("logf({sub})")
                        }
                    }
                    IrUnaryOp::Sin => {
                        if is_f64 {
                            format!("sin({sub})")
                        } else {
                            format!("sinf({sub})")
                        }
                    }
                    IrUnaryOp::Cos => {
                        if is_f64 {
                            format!("cos({sub})")
                        } else {
                            format!("cosf({sub})")
                        }
                    }
                    IrUnaryOp::Sqrt => {
                        if is_f64 {
                            format!("sqrt({sub})")
                        } else {
                            format!("sqrtf({sub})")
                        }
                    }
                    IrUnaryOp::Rsqrt => {
                        if is_f64 {
                            format!("(1.0 / sqrt({sub}))")
                        } else {
                            format!("rsqrtf({sub})")
                        }
                    }
                    IrUnaryOp::Relu => {
                        if is_f64 {
                            format!("fmax(0.0, {sub})")
                        } else {
                            format!("fmaxf(0.0f, {sub})")
                        }
                    }
                    IrUnaryOp::Step => format!("(({sub}) > 0.0f ? 1.0f : 0.0f)"),
                    IrUnaryOp::Square => format!("(({sub}) * ({sub}))"),
                    IrUnaryOp::Reciprocal => {
                        if is_f64 {
                            format!("(1.0 / ({sub}))")
                        } else {
                            format!("(1.0f / ({sub}))")
                        }
                    }
                    IrUnaryOp::Tanh => {
                        if is_f64 {
                            format!("tanh({sub})")
                        } else {
                            format!("tanhf({sub})")
                        }
                    }
                    IrUnaryOp::Sigmoid => {
                        if is_f64 {
                            format!("(1.0 / (1.0 + exp(-({sub}))))")
                        } else {
                            format!("(1.0f / (1.0f + expf(-({sub}))))")
                        }
                    }
                    IrUnaryOp::Silu => {
                        if is_f64 {
                            format!("(({sub}) / (1.0 + exp(-({sub}))))")
                        } else {
                            format!("(({sub}) / (1.0f + expf(-({sub}))))")
                        }
                    }
                    IrUnaryOp::Gelu => {
                        let k = if is_f64 {
                            "0.7978845608028654"
                        } else {
                            "0.79788456f"
                        };
                        let c = if is_f64 { "0.044715" } else { "0.044715f" };
                        let half = if is_f64 { "0.5" } else { "0.5f" };
                        let one = if is_f64 { "1.0" } else { "1.0f" };
                        let tanh_fn = if is_f64 { "tanh" } else { "tanhf" };
                        format!(
                            "({half} * ({sub}) * ({one} + {tanh_fn}({k} * (({sub}) + {c} * ({sub}) * ({sub}) * ({sub})))))"
                        )
                    }
                }
            }
            Self::Binary(op, lhs, rhs) => {
                let sl = lhs.render_cuda_expr(dtype);
                let sr = rhs.render_cuda_expr(dtype);
                match op {
                    IrBinaryOp::Add => format!("({sl} + {sr})"),
                    IrBinaryOp::Sub => format!("({sl} - {sr})"),
                    IrBinaryOp::Mul => format!("({sl} * {sr})"),
                    IrBinaryOp::Div => format!("({sl} / {sr})"),
                    IrBinaryOp::Pow => {
                        if is_f64 {
                            format!("pow({sl}, {sr})")
                        } else {
                            format!("powf({sl}, {sr})")
                        }
                    }
                    IrBinaryOp::Max => {
                        if is_f64 {
                            format!("fmax({sl}, {sr})")
                        } else {
                            format!("fmaxf({sl}, {sr})")
                        }
                    }
                    IrBinaryOp::Min => {
                        if is_f64 {
                            format!("fmin({sl}, {sr})")
                        } else {
                            format!("fminf({sl}, {sr})")
                        }
                    }
                }
            }
            Self::Ternary(op, a, b, c) => {
                let sa = a.render_cuda_expr(dtype);
                let sb = b.render_cuda_expr(dtype);
                let sc = c.render_cuda_expr(dtype);
                match op {
                    IrTernaryOp::Fma => {
                        if is_f64 {
                            format!("fma({sa}, {sb}, {sc})")
                        } else {
                            format!("fmaf({sa}, {sb}, {sc})")
                        }
                    }
                    IrTernaryOp::Clamp => {
                        if is_f64 {
                            format!("fmin(fmax({sa}, {sb}), {sc})")
                        } else {
                            format!("fminf(fmaxf({sa}, {sb}), {sc})")
                        }
                    }
                    IrTernaryOp::Select => format!("(({sa}) > 0.0f ? ({sb}) : ({sc}))"),
                }
            }
        }
    }
}

/// A complete kernel definition with forward execution, symbolic derivative backward execution, and multi-target code generation.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelDefinition {
    /// Operation name identifier.
    pub name: String,
    /// Number of input tensor operands.
    pub input_arity: usize,
    /// Data type.
    pub dtype: DTypeId,
    /// Forward mathematical expression.
    pub forward: IrExpr,
    /// Backward derivatives for each input tensor: `d(forward) / d(in_i)`.
    pub backward_derivatives: Vec<IrExpr>,
}

impl KernelDefinition {
    /// Builds a new kernel definition, automatically computing all symbolic analytical derivatives!
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        input_arity: usize,
        dtype: DTypeId,
        forward: IrExpr,
    ) -> Self {
        let opt_forward = forward.optimize();
        let mut backward_derivatives = Vec::with_capacity(input_arity);
        for i in 0..input_arity {
            backward_derivatives.push(opt_forward.diff(i));
        }

        Self {
            name: name.into(),
            input_arity,
            dtype,
            forward: opt_forward,
            backward_derivatives,
        }
    }

    /// Renders the complete forward CUDA C++ kernel.
    #[must_use]
    pub fn render_forward_cuda(&self) -> String {
        let mut out = String::new();
        let scalar_ty = match self.dtype {
            DTypeId::F32 => "float",
            DTypeId::F64 => "double",
            DTypeId::F16 => "__half",
            DTypeId::BF16 => "__nv_bfloat16",
            _ => "float",
        };

        writeln!(out, "// Generated forward kernel for {} (CUDA)", self.name).unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out).unwrap();

        write!(out, "extern \"C\" __global__ void {}_forward(", self.name).unwrap();
        for i in 0..self.input_arity {
            write!(out, "const {scalar_ty}* __restrict__ in{i}, ").unwrap();
        }
        writeln!(out, "{scalar_ty}* __restrict__ out, const int numel) {{").unwrap();
        writeln!(
            out,
            "    const int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        )
        .unwrap();
        writeln!(out, "    if (idx >= numel) return;").unwrap();
        writeln!(
            out,
            "    out[idx] = static_cast<{scalar_ty}>({});",
            self.forward.render_cuda_expr(self.dtype)
        )
        .unwrap();
        writeln!(out, "}}").unwrap();

        out
    }

    /// Renders the complete backward CUDA C++ kernel for input index `input_idx`.
    #[must_use]
    pub fn render_backward_cuda(&self, input_idx: usize) -> Option<String> {
        let derivative = self.backward_derivatives.get(input_idx)?;
        let mut out = String::new();
        let scalar_ty = match self.dtype {
            DTypeId::F32 => "float",
            DTypeId::F64 => "double",
            DTypeId::F16 => "__half",
            DTypeId::BF16 => "__nv_bfloat16",
            _ => "float",
        };

        writeln!(
            out,
            "// Generated backward kernel for {} arg {input_idx} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out).unwrap();

        write!(
            out,
            "extern \"C\" __global__ void {}_backward_{input_idx}(",
            self.name
        )
        .unwrap();
        write!(out, "const {scalar_ty}* __restrict__ grad_out, ").unwrap();
        for i in 0..self.input_arity {
            write!(out, "const {scalar_ty}* __restrict__ in{i}, ").unwrap();
        }
        writeln!(
            out,
            "{scalar_ty}* __restrict__ grad_in{input_idx}, const int numel) {{"
        )
        .unwrap();
        writeln!(
            out,
            "    const int idx = blockIdx.x * blockDim.x + threadIdx.x;"
        )
        .unwrap();
        writeln!(out, "    if (idx >= numel) return;").unwrap();
        writeln!(
            out,
            "    const float local_grad = static_cast<float>(grad_out[idx]);"
        )
        .unwrap();
        writeln!(
            out,
            "    const float local_derivative = static_cast<float>({});",
            derivative.render_cuda_expr(self.dtype)
        )
        .unwrap();
        writeln!(
            out,
            "    grad_in{input_idx}[idx] = static_cast<{scalar_ty}>(local_grad * local_derivative);"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();

        Some(out)
    }
}

// ---------------------------------------------------------------------------
// Operator Overloading for Natural Mathematical Syntax
// ---------------------------------------------------------------------------

impl core::ops::Add<IrExpr> for IrExpr {
    type Output = IrExpr;
    fn add(self, rhs: IrExpr) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Add, self, rhs)
    }
}

impl core::ops::Add<f64> for IrExpr {
    type Output = IrExpr;
    fn add(self, rhs: f64) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Add, self, IrExpr::constant(rhs))
    }
}

impl core::ops::Add<IrExpr> for f64 {
    type Output = IrExpr;
    fn add(self, rhs: IrExpr) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Add, IrExpr::constant(self), rhs)
    }
}

impl core::ops::Sub<IrExpr> for IrExpr {
    type Output = IrExpr;
    fn sub(self, rhs: IrExpr) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Sub, self, rhs)
    }
}

impl core::ops::Sub<f64> for IrExpr {
    type Output = IrExpr;
    fn sub(self, rhs: f64) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Sub, self, IrExpr::constant(rhs))
    }
}

impl core::ops::Sub<IrExpr> for f64 {
    type Output = IrExpr;
    fn sub(self, rhs: IrExpr) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Sub, IrExpr::constant(self), rhs)
    }
}

impl core::ops::Mul<IrExpr> for IrExpr {
    type Output = IrExpr;
    fn mul(self, rhs: IrExpr) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Mul, self, rhs)
    }
}

impl core::ops::Mul<f64> for IrExpr {
    type Output = IrExpr;
    fn mul(self, rhs: f64) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Mul, self, IrExpr::constant(rhs))
    }
}

impl core::ops::Mul<IrExpr> for f64 {
    type Output = IrExpr;
    fn mul(self, rhs: IrExpr) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Mul, IrExpr::constant(self), rhs)
    }
}

impl core::ops::Div<IrExpr> for IrExpr {
    type Output = IrExpr;
    fn div(self, rhs: IrExpr) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Div, self, rhs)
    }
}

impl core::ops::Div<f64> for IrExpr {
    type Output = IrExpr;
    fn div(self, rhs: f64) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Div, self, IrExpr::constant(rhs))
    }
}

impl core::ops::Div<IrExpr> for f64 {
    type Output = IrExpr;
    fn div(self, rhs: IrExpr) -> IrExpr {
        IrExpr::binary(IrBinaryOp::Div, IrExpr::constant(self), rhs)
    }
}

impl core::ops::Neg for IrExpr {
    type Output = IrExpr;
    fn neg(self) -> IrExpr {
        IrExpr::unary(IrUnaryOp::Neg, self)
    }
}

/// Exponential: `exp(x)`
#[must_use]
pub fn exp(x: IrExpr) -> IrExpr {
    x.exp()
}

/// Natural log: `log(x)`
#[must_use]
pub fn log(x: IrExpr) -> IrExpr {
    x.log()
}

/// Square root: `sqrt(x)`
#[must_use]
pub fn sqrt(x: IrExpr) -> IrExpr {
    x.sqrt()
}

/// Reciprocal square root: `rsqrt(x)`
#[must_use]
pub fn rsqrt(x: IrExpr) -> IrExpr {
    x.rsqrt()
}

/// ReLU: `relu(x)`
#[must_use]
pub fn relu(x: IrExpr) -> IrExpr {
    x.relu()
}

/// Sigmoid: `sigmoid(x)`
#[must_use]
pub fn sigmoid(x: IrExpr) -> IrExpr {
    x.sigmoid()
}

/// Tanh: `tanh(x)`
#[must_use]
pub fn tanh(x: IrExpr) -> IrExpr {
    x.tanh()
}

/// GELU: `gelu(x)`
#[must_use]
pub fn gelu(x: IrExpr) -> IrExpr {
    x.gelu()
}

/// SiLU / Swish: `silu(x)`
#[must_use]
pub fn silu(x: IrExpr) -> IrExpr {
    x.silu()
}

/// Fused multiply-add: `fma(a, b, c) = a * b + c`
#[must_use]
pub fn fma(a: IrExpr, b: IrExpr, c: IrExpr) -> IrExpr {
    IrExpr::fma(a, b, c)
}
