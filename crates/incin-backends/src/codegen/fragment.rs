//! Scalar-fragment lowering: `IrExpr` into the body of an existing kernel template.
//!
//! `ir::render_cuda_expr` renders an expression against the IR's *own* kernel
//! signature, where `Arg(i)` becomes `in{i}[idx]`. That ties it to
//! `KernelDefinition::render_forward_cuda`, which emits a contiguous, unstrided,
//! unvectorised, uncached kernel and therefore cannot replace anything in
//! `crate::kernel` without losing strided layouts, packed loads, launch
//! autotuning and the kernel cache key.
//!
//! This module renders the same IR against *caller-named operands* instead, so
//! the result drops into the `{OP}` slot of the templates in
//! `crate::kernel::scalar`, where `x` (unary) or `a` and `b` (binary) are already
//! loaded as compute-typed scalars. Everything those templates provide is
//! inherited unchanged; only the producer of the expression text changes.
//!
//! Output is in static single assignment form: every interior node is bound to
//! its own `const` temporary and referred to by name. That is not cosmetic. The
//! direct renderer duplicates a subexpression's *text* once per syntactic use,
//! and `Square`, `Silu` and `Gelu` each use their operand more than once, so
//! nesting them grows the emitted source exponentially in the depth of the tree.
//! Binding every node once makes emission linear in the node count, and
//! memoising on structural identity makes common subexpression elimination fall
//! out of the same pass: two occurrences of the same subtree resolve to one
//! temporary and are evaluated once.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write as _;

use incin_core::error::{Error, Result};
use incin_core::tensor::dtype::DTypeId;

use super::ir::{IrBinaryOp, IrExpr, IrTernaryOp, IrUnaryOp};

/// A lowered expression: `prologue` statements followed by a `value` naming the result.
///
/// `value` is either a bare temporary name, an operand name or a literal, so it
/// is always safe to substitute into an rvalue position without extra
/// parentheses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarFragment {
    /// `const T t0 = ...;` bindings, in evaluation order.
    pub prologue: Vec<String>,
    /// The expression naming the fragment's result.
    pub value: String,
}

impl ScalarFragment {
    /// Wraps a hand-written CUDA C expression as a fragment with no bindings.
    ///
    /// This is the adapter that lets the literal and IR paths share one
    /// rendering pipeline: a literal is just a fragment whose prologue is empty.
    #[must_use]
    pub fn literal(expr: &str) -> Self {
        Self {
            prologue: Vec::new(),
            value: expr.to_string(),
        }
    }

    /// Renders the prologue as one indented block, or the empty string when
    /// there are no bindings.
    #[must_use]
    pub fn prologue_block(&self, indent: &str) -> String {
        let mut out = String::new();
        for statement in &self.prologue {
            let _ = writeln!(out, "{indent}{statement}");
        }
        out
    }
}

/// The C scalar type an expression is evaluated in.
///
/// This is the kernel's *compute* type, not its storage type. `F16` and `BF16`
/// tensors compute in `float`, so callers lowering a half-precision kernel pass
/// `F32` here and the fragment uses single-precision math functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComputeType {
    F32,
    F64,
}

impl ComputeType {
    fn resolve(dtype: DTypeId) -> Result<Self> {
        match dtype {
            DTypeId::F16 | DTypeId::BF16 | DTypeId::F32 => Ok(Self::F32),
            DTypeId::F64 => Ok(Self::F64),
            other => Err(Error::Msg(format!(
                "IR fragments require a float compute type, got {other:?}"
            ))),
        }
    }

    const fn c_name(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    const fn is_f64(self) -> bool {
        matches!(self, Self::F64)
    }

    /// Picks the double- or single-precision spelling of a libdevice function.
    const fn pick(self, f64_name: &'static str, f32_name: &'static str) -> &'static str {
        match self {
            Self::F64 => f64_name,
            Self::F32 => f32_name,
        }
    }

    /// Formats a literal with the suffix its precision requires.
    fn literal(self, value: f64) -> String {
        // `{:?}` on f64 round-trips and keeps short values short, so 0.5 stays
        // "0.5" rather than "0.50000000". Integral values need an explicit
        // fractional part or the suffix would attach to an integer literal.
        let mut text = format!("{value:?}");
        if !text.contains('.')
            && !text.contains('e')
            && !text.contains("inf")
            && !text.contains("NaN")
        {
            text.push_str(".0");
        }
        if self.is_f64() {
            text
        } else {
            text.push('f');
            text
        }
    }
}

/// Lowers `expr` to SSA over `operands`, evaluated in `dtype`'s compute type.
///
/// `operands[i]` supplies the C name substituted for `IrExpr::Arg(i)`; an
/// argument index past the end of the slice is an error rather than a silently
/// mis-bound read. `IrExpr::Var` names pass through verbatim, which is how a
/// caller injects a value the template already has in scope.
///
/// # Errors
///
/// Returns an error when `dtype` is not a float type, or when the expression
/// references an argument index `operands` does not supply.
pub fn lower_scalar(expr: &IrExpr, operands: &[&str], dtype: DTypeId) -> Result<ScalarFragment> {
    let compute = ComputeType::resolve(dtype)?;
    let mut lowering = Lowering {
        compute,
        operands,
        prologue: Vec::new(),
        memo: BTreeMap::new(),
        next_temp: 0,
    };
    let value = lowering.lower(expr)?;
    Ok(ScalarFragment {
        prologue: lowering.prologue,
        value,
    })
}

struct Lowering<'a> {
    compute: ComputeType,
    operands: &'a [&'a str],
    prologue: Vec<String>,
    /// Structural key of an already-lowered node to the name holding its value.
    /// This is what makes repeated subtrees share one evaluation.
    memo: BTreeMap<String, String>,
    next_temp: usize,
}

impl Lowering<'_> {
    /// Binds `body` to a fresh temporary and returns its name.
    fn bind(&mut self, body: &str) -> String {
        let name = format!("t{}", self.next_temp);
        self.next_temp += 1;
        self.prologue
            .push(format!("const {} {name} = {body};", self.compute.c_name()));
        name
    }

    fn lower(&mut self, expr: &IrExpr) -> Result<String> {
        // Leaves are already names or literals. Binding them would only add
        // indirection, and they cannot be the source of duplicated work.
        match expr {
            IrExpr::Arg(index) => {
                return self.operands.get(*index).map(ToString::to_string).ok_or_else(|| {
                    Error::Msg(format!(
                        "IR fragment references argument {index} but only {} operand(s) were supplied",
                        self.operands.len()
                    ))
                });
            }
            IrExpr::Const(value) => return Ok(self.compute.literal(*value)),
            IrExpr::Var(name) => return Ok(name.clone()),
            _ => {}
        }

        let key = structural_key(expr);
        if let Some(existing) = self.memo.get(&key) {
            return Ok(existing.clone());
        }

        let body = self.emit_interior(expr)?;
        let name = self.bind(&body);
        self.memo.insert(key, name.clone());
        Ok(name)
    }

    /// Emits the right-hand side for an interior node, with every operand
    /// already lowered to a name.
    fn emit_interior(&mut self, expr: &IrExpr) -> Result<String> {
        let compute = self.compute;
        let one = compute.literal(1.0);
        let zero = compute.literal(0.0);
        Ok(match expr {
            IrExpr::Arg(_) | IrExpr::Const(_) | IrExpr::Var(_) => {
                unreachable!("leaves are lowered before emit_interior")
            }
            IrExpr::Unary(op, inner) => {
                let v = self.lower(inner)?;
                match op {
                    IrUnaryOp::Neg => format!("-{v}"),
                    IrUnaryOp::Abs => format!("{}({v})", compute.pick("fabs", "fabsf")),
                    IrUnaryOp::Exp => format!("{}({v})", compute.pick("exp", "expf")),
                    IrUnaryOp::Log => format!("{}({v})", compute.pick("log", "logf")),
                    IrUnaryOp::Sin => format!("{}({v})", compute.pick("sin", "sinf")),
                    IrUnaryOp::Cos => format!("{}({v})", compute.pick("cos", "cosf")),
                    IrUnaryOp::Sqrt => format!("{}({v})", compute.pick("sqrt", "sqrtf")),
                    IrUnaryOp::Rsqrt => match compute {
                        ComputeType::F64 => format!("({one} / sqrt({v}))"),
                        ComputeType::F32 => format!("rsqrtf({v})"),
                    },
                    IrUnaryOp::Relu => {
                        format!("{}({zero}, {v})", compute.pick("fmax", "fmaxf"))
                    }
                    IrUnaryOp::Step => format!("({v} > {zero} ? {one} : {zero})"),
                    // SSA is what makes these safe: `v` is a name, so using it
                    // twice costs one extra read of a register, not a second
                    // evaluation of the whole subtree.
                    IrUnaryOp::Square => format!("({v} * {v})"),
                    IrUnaryOp::Reciprocal => format!("({one} / {v})"),
                    IrUnaryOp::Tanh => format!("{}({v})", compute.pick("tanh", "tanhf")),
                    IrUnaryOp::Sigmoid => {
                        let exp = compute.pick("exp", "expf");
                        format!("({one} / ({one} + {exp}(-{v})))")
                    }
                    IrUnaryOp::Silu => {
                        let exp = compute.pick("exp", "expf");
                        format!("({v} / ({one} + {exp}(-{v})))")
                    }
                    IrUnaryOp::Gelu => {
                        let tanh = compute.pick("tanh", "tanhf");
                        let k = compute.literal(GELU_K);
                        let c = compute.literal(GELU_C);
                        let half = compute.literal(0.5);
                        format!(
                            "({half} * {v} * ({one} + {tanh}({k} * ({v} + {c} * {v} * {v} * {v}))))"
                        )
                    }
                }
            }
            IrExpr::Binary(op, lhs, rhs) => {
                let l = self.lower(lhs)?;
                let r = self.lower(rhs)?;
                match op {
                    IrBinaryOp::Add => format!("({l} + {r})"),
                    IrBinaryOp::Sub => format!("({l} - {r})"),
                    IrBinaryOp::Mul => format!("({l} * {r})"),
                    IrBinaryOp::Div => format!("({l} / {r})"),
                    IrBinaryOp::Pow => format!("{}({l}, {r})", compute.pick("pow", "powf")),
                    IrBinaryOp::Max => format!("{}({l}, {r})", compute.pick("fmax", "fmaxf")),
                    IrBinaryOp::Min => format!("{}({l}, {r})", compute.pick("fmin", "fminf")),
                }
            }
            IrExpr::Ternary(op, a, b, c) => {
                let va = self.lower(a)?;
                let vb = self.lower(b)?;
                let vc = self.lower(c)?;
                match op {
                    IrTernaryOp::Fma => {
                        format!("{}({va}, {vb}, {vc})", compute.pick("fma", "fmaf"))
                    }
                    IrTernaryOp::Clamp => {
                        let min = compute.pick("fmin", "fminf");
                        let max = compute.pick("fmax", "fmaxf");
                        format!("{min}({max}({va}, {vb}), {vc})")
                    }
                    IrTernaryOp::Select => format!("({va} > {zero} ? {vb} : {vc})"),
                }
            }
        })
    }
}

/// The tanh-approximation constants GELU is defined by, shared with `ir.rs` so
/// the forward expression and its derivative cannot drift apart.
pub(crate) const GELU_K: f64 = 0.797_884_560_802_865_4;
pub(crate) const GELU_C: f64 = 0.044_715;

/// A canonical prefix-form key identifying a subtree up to structural equality.
///
/// `IrExpr` holds an `f64` and so cannot derive `Hash` or `Eq`; this gives an
/// ordered key instead. Constants are keyed on their bit pattern so that `-0.0`
/// and `0.0` stay distinct and `NaN` keys compare equal to itself, neither of
/// which is true of `f64` comparison.
fn structural_key(expr: &IrExpr) -> String {
    let mut out = String::new();
    write_key(expr, &mut out);
    out
}

fn write_key(expr: &IrExpr, out: &mut String) {
    match expr {
        IrExpr::Arg(index) => {
            let _ = write!(out, "a{index};");
        }
        IrExpr::Const(value) => {
            let _ = write!(out, "c{:x};", value.to_bits());
        }
        IrExpr::Var(name) => {
            let _ = write!(out, "v{name};");
        }
        IrExpr::Unary(op, inner) => {
            let _ = write!(out, "u{op:?}(");
            write_key(inner, out);
            out.push(')');
        }
        IrExpr::Binary(op, lhs, rhs) => {
            let _ = write!(out, "b{op:?}(");
            write_key(lhs, out);
            write_key(rhs, out);
            out.push(')');
        }
        IrExpr::Ternary(op, a, b, c) => {
            let _ = write!(out, "t{op:?}(");
            write_key(a, out);
            write_key(b, out);
            write_key(c, out);
            out.push(')');
        }
    }
}
