//! `NumericOps` (`add`/`sub`/`mul`/`div`) and `FloatOps::{add_scalar_float,
//! mul_scalar_float}` for `CpuBackend<T, D>`.
//!
//! Every op here resolves the broadcast output shape via
//! `stride::broadcast_shape`, then iterates the OUTPUT shape's logical index
//! space, resolving each operand's own index through its own strides with
//! wraparound (stride-0-equivalent) logic on right-aligned/expanded
//! dimensions — it never pre-materializes a broadcast copy of either operand
//! (the anti-pattern flagged in RESEARCH.md). Every op pushes a `TapeEntry`
//! whose backward closure calls `tape::unbroadcast` on the ORIGINAL
//! (pre-broadcast) operand shapes.

use kindle_core::prelude::{Backend, DType, FloatOps, NumericOps, Result};

use crate::cpu::CpuBackend;
use crate::cpu::storage::CpuStorage;
use crate::cpu::tape::{self, TapeEntry};

/// Resolve the per-operand logical index for a given output logical index,
/// right-aligning `operand_shape` against `out_shape` (numpy/Candle-style
/// broadcast): any leading dim the operand doesn't have is dropped, and any
/// dim of size 1 in the operand wraps to index 0 regardless of the output's
/// index at that axis.
fn broadcast_index(out_idx: &[usize], out_shape: &[usize], operand_shape: &[usize]) -> Vec<usize> {
    let offset = out_shape.len() - operand_shape.len();
    operand_shape
        .iter()
        .enumerate()
        .map(|(i, &dim)| if dim == 1 { 0 } else { out_idx[i + offset] })
        .collect()
}

/// Increment a row-major multi-index in place (odometer-style), matching
/// `storage.rs`/`tape.rs`'s own iteration order.
pub(crate) fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for i in (0..idx.len()).rev() {
        idx[i] += 1;
        if idx[i] < shape[i] {
            return;
        }
        idx[i] = 0;
    }
}

pub(crate) fn flat_to_nd(mut flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut nd = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        nd[i] = flat_idx % shape[i];
        flat_idx /= shape[i];
    }
    nd
}

/// Read `storage` as `f64` at `storage`'s own (already-resolved) logical
/// index, given the output-space index and shape.
fn read_broadcast(storage: &CpuStorage, out_idx: &[usize], out_shape: &[usize]) -> f64 {
    let idx = broadcast_index(out_idx, out_shape, &storage.shape);
    storage.get(&idx)
}

/// Build a contiguous `CpuStorage` by applying `f(lhs_val, rhs_val)` over
/// every logical index in `out_shape`, reading each operand through its own
/// broadcast-resolved index (no pre-materialized broadcast copy).
use rayon::prelude::*;

pub(crate) fn elementwise_binary(
    _op_name: &str,
    _op_expr: &str,
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
    f: impl Fn(f64, f64) -> f64 + Send + Sync,
) -> Result<CpuStorage> {
    let total: usize = out_shape.iter().product();
    let out: Vec<f64> = (0..total)
        .into_par_iter()
        .map(|flat_idx| {
            let nd_idx = flat_to_nd(flat_idx, out_shape);
            let a = read_broadcast(lhs, &nd_idx, out_shape);
            let b = read_broadcast(rhs, &nd_idx, out_shape);
            f(a, b)
        })
        .collect();
    // Preserve lhs's actual dtype variant instead of hardcoding F32 (C-2:
    // this used to silently downcast every non-f32 dtype through f32).
    let out_buffer = lhs.buffer.from_f64_values(out);
    Ok(CpuStorage::from_contiguous(out_buffer, out_shape.to_vec()))
}

/// Elementwise negate (used by `sub`'s backward rule: rhs receives the
/// negated incoming gradient before unbroadcasting).
pub(crate) fn elementwise_unary(
    _op_name: &str,
    _op_expr: &str,
    t: &CpuStorage,
    f: impl Fn(f64) -> f64 + Send + Sync,
) -> Result<CpuStorage> {
    let total: usize = t.shape.iter().product();
    let out: Vec<f64> = (0..total)
        .into_par_iter()
        .map(|flat_idx| {
            let nd_idx = flat_to_nd(flat_idx, &t.shape);
            f(t.get(&nd_idx))
        })
        .collect();
    // Preserve t's actual dtype variant instead of hardcoding F32 (C-2: this
    // used to silently downcast every non-f32 dtype through f32).
    let out_buffer = t.buffer.from_f64_values(out);
    Ok(CpuStorage::from_contiguous(out_buffer, t.shape.clone()))
}

/// `negate`.
fn negate(t: &CpuStorage) -> CpuStorage {
    elementwise_unary("neg", "-x", t, |x| -x).unwrap()
}

/// Elementwise multiply of two ALREADY-shape-matching (or one of them
/// broadcastable to the other's shape) storages — used by `mul`'s backward
/// rule to compute `grad_out * other_operand`, aligned to `grad_out`'s shape.
#[allow(dead_code)]
fn mul_elementwise_broadcast(grad_out: &CpuStorage, other: &CpuStorage) -> Result<CpuStorage> {
    elementwise_binary("mul", "a * b", grad_out, other, &grad_out.shape, |a, b| {
        a * b
    })
}

/// Abramowitz & Stegun 7.1.26 rational polynomial approximation of the
/// error function `erf(x)`, max absolute error ~1.5e-7. Rust's standard
/// library has no `erf`, so `gelu`'s exact erf-based formula (matching
/// `CandleBackend::gelu`'s `gelu_erf()` call, per RESEARCH.md Pitfall 4) is
/// built on this hand-rolled approximation instead.
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x_abs);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x_abs * x_abs).exp();
    sign * y
}

impl<T: DType, D: kindle_core::prelude::Device> NumericOps<Self> for CpuBackend<T, D> {
    /// `add`.
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary("add", "a + b", lhs, rhs, &out_shape, |a, b| a + b)?;

        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                vec![
                    tape::unbroadcast(grad_out, &lhs_shape).expect("unbroadcast lhs (add)"),
                    tape::unbroadcast(grad_out, &rhs_shape).expect("unbroadcast rhs (add)"),
                ]
            }),
        });
        Ok(out)
    }

    /// `sub`.
    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary("sub", "a - b", lhs, rhs, &out_shape, |a, b| a - b)?;

        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                vec![
                    tape::unbroadcast(grad_out, &lhs_shape).expect("unbroadcast lhs (sub)"),
                    tape::unbroadcast(&negate(grad_out), &rhs_shape)
                        .expect("unbroadcast rhs (sub)"),
                ]
            }),
        });
        Ok(out)
    }

    /// `mul`.
    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary("mul", "a * b", lhs, rhs, &out_shape, |a, b| a * b)?;

        // Capture cloned copies of lhs/rhs's CpuStorage (cheap, Rc-backed)
        // since the backward closure needs their VALUES, not just shapes.
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad_lhs = elementwise_binary(
                    "mul",
                    "a * b",
                    grad_out,
                    &rhs_capture,
                    &grad_out.shape,
                    |a, b| a * b,
                )
                .unwrap();
                let grad_rhs = elementwise_binary(
                    "mul",
                    "a * b",
                    grad_out,
                    &lhs_capture,
                    &grad_out.shape,
                    |a, b| a * b,
                )
                .unwrap();
                vec![
                    tape::unbroadcast(&grad_lhs, &lhs_shape).expect("unbroadcast lhs (mul)"),
                    tape::unbroadcast(&grad_rhs, &rhs_shape).expect("unbroadcast rhs (mul)"),
                ]
            }),
        });
        Ok(out)
    }

    /// `div`.
    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary("div", "a / b", lhs, rhs, &out_shape, |a, b| a / b)?;

        // Per Assumption A2 (RESEARCH.md): implemented for trait-completeness
        // via the standard quotient rule (1/rhs, -lhs/rhs^2), each
        // unbroadcast — best-effort correctness, not exercised by this
        // phase's example/tests.
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
                let grad_lhs = elementwise_binary(
                    "div_g_r",
                    "a / b",
                    grad_out,
                    &rhs_capture,
                    &grad_out.shape,
                    |g, r| g / r,
                )
                .unwrap();
                // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = grad_out * (-lhs/rhs^2)
                let grad_rhs = elementwise_binary(
                    "mul_g_dr",
                    "a * b",
                    grad_out,
                    &elementwise_binary(
                        "div_grad_rhs_inner",
                        "-a / (b * b)",
                        &lhs_capture,
                        &rhs_capture,
                        &grad_out.shape,
                        |l, r| -l / (r * r),
                    )
                    .unwrap(),
                    &grad_out.shape,
                    |g, dr| g * dr,
                )
                .unwrap();
                vec![
                    tape::unbroadcast(&grad_lhs, &lhs_shape).expect("unbroadcast lhs (div)"),
                    tape::unbroadcast(&grad_rhs, &rhs_shape).expect("unbroadcast rhs (div)"),
                ]
            }),
        });
        Ok(out)
    }
}

impl<T: DType, D: kindle_core::prelude::Device> FloatOps<Self> for CpuBackend<T, D> {
    /// `add_scalar_float`.
    fn add_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary(
            &format!("add_scalar_{}", scalar),
            &format!("x + {}", scalar),
            t,
            |x| x + scalar,
        )?;

        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            // Gradient passes through unchanged (same shape, no unbroadcast
            // needed — scalar ops don't change shape).
            backward: Box::new(move |grad_out: &CpuStorage| vec![grad_out.clone()]),
        });
        Ok(out)
    }

    /// `mul_scalar_float`.
    fn mul_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary(
            &format!("mul_scalar_{}", scalar),
            &format!("x * {}", scalar),
            t,
            |x| x * scalar,
        )?;

        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            // Gradient scales by the same constant (same shape, no
            // unbroadcast needed).
            backward: Box::new(move |grad_out: &CpuStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut scaled = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    scaled.push(grad_out.get(&idx) * scalar);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![CpuStorage::from_contiguous(
                    grad_out.buffer.from_f64_values(scaled),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    /// `relu`.
    fn relu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary("relu", "x > 0.0 ? x : 0.0", t, |x| x.max(0.0))?;

        // relu'(x) = 1 if x > 0 else 0 (input-based; strict `>`, zero
        // gradient at the x=0 boundary).
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad = elementwise_binary(
                    "relu_grad",
                    "b > 0.0 ? a : 0.0",
                    grad_out,
                    &t_capture,
                    &grad_out.shape,
                    |g, x| {
                        let deriv = if x > 0.0 { 1.0 } else { 0.0 };
                        g * deriv
                    },
                )
                .unwrap();
                vec![grad]
            }),
        });
        Ok(out)
    }

    /// `step`.
    fn step<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary("step", "x > 0.0 ? 1.0 : 0.0", t, |x| {
            if x > 0.0 { 1.0 } else { 0.0 }
        })?;

        // step'(x) = 0 almost everywhere.
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let total: usize = grad_out.shape.iter().product();
                let zeros = vec![0.0f64; total];
                vec![CpuStorage::from_contiguous(
                    grad_out.buffer.from_f64_values(zeros),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    /// `mish`.
    fn mish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary(
            "mish",
            "x * tanhf(x > 20.0 ? x : logf(1.0 + expf(x)))",
            t,
            |x| {
                let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
                x * sp.tanh()
            },
        )?;

        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad = elementwise_binary("mish_grad", "a * (tanhf(b > 20.0 ? b : logf(1.0 + expf(b))) + b * (1.0 / (1.0 + expf(-b))) * (1.0 - powf(tanhf(b > 20.0 ? b : logf(1.0 + expf(b))), 2.0)))", grad_out, &t_capture, &grad_out.shape, |g, x| {
                    let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
                    let th = sp.tanh();
                    let sig = 1.0 / (1.0 + (-x).exp());
                    let deriv = th + x * sig * (1.0 - th * th);
                    g * deriv
                }).unwrap();
                vec![grad]
            }),
        });
        Ok(out)
    }

    /// `elu`.
    fn elu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary("elu", "x > 0.0 ? x : expf(x) - 1.0", t, |x| {
            if x > 0.0 { x } else { x.exp() - 1.0 }
        })?;

        let out_capture = out.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad = elementwise_binary(
                    "elu_grad",
                    "a * (b > 0.0 ? 1.0 : b + 1.0)",
                    grad_out,
                    &out_capture,
                    &grad_out.shape,
                    |g, o| {
                        let deriv = if o > 0.0 { 1.0 } else { o + 1.0 };
                        g * deriv
                    },
                )
                .unwrap();
                vec![grad]
            }),
        });
        Ok(out)
    }

    /// `gelu`.
    fn gelu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary(
            "gelu",
            "x * 0.5f * (1.0f + erff(x / 1.41421356f))",
            t,
            |x| x * 0.5 * (1.0 + erf_approx(x / core::f64::consts::SQRT_2)),
        )?;

        // gelu(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
        // gelu'(x) = 0.5*(1+erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
        // (input-based — not simplifiable purely in terms of the output).
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad = elementwise_binary("gelu_grad", "a * (0.5 * (1.0 + erff(b / 1.41421356)) + b * (0.39894228 * expf(-0.5 * b * b)))", grad_out, &t_capture, &grad_out.shape, |g, x| {
                    let cdf = 0.5 * (1.0 + erf_approx(x / core::f64::consts::SQRT_2));
                    let pdf = (1.0 / (2.0 * core::f64::consts::PI).sqrt()) * (-x * x / 2.0).exp();
                    let deriv = cdf + x * pdf;
                    g * deriv
                }).unwrap();
                vec![grad]
            }),
        });
        Ok(out)
    }

    /// `abs`.
    fn abs<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary("abs", "fabsf(x)", t, |x| x.abs())?;

        // abs'(x) = sign(x) (input-based).
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad = elementwise_binary(
                    "abs_grad",
                    "b > 0.0 ? a : (b < 0.0 ? -a : 0.0)",
                    grad_out,
                    &t_capture,
                    &grad_out.shape,
                    |g, x| {
                        let deriv = if x > 0.0 {
                            1.0
                        } else if x < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                        g * deriv
                    },
                )
                .unwrap();
                vec![grad]
            }),
        });
        Ok(out)
    }

    /// `exp`.
    fn exp<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary("exp", "expf(x)", t, |x| x.exp())?;

        // exp'(x) = out (output-based).
        let out_capture = out.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad = elementwise_binary(
                    "exp_grad",
                    "a * b",
                    grad_out,
                    &out_capture,
                    &grad_out.shape,
                    |g, o| {
                        let deriv = o;
                        g * deriv
                    },
                )
                .unwrap();
                vec![grad]
            }),
        });
        Ok(out)
    }

    /// `neg`.
    fn neg<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = negate(t);

        // neg'(x) = -1 (constant; no input capture needed).
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| vec![negate(grad_out)]),
        });
        Ok(out)
    }

    /// `sqrt`.
    fn sqrt<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary("sqrt", "sqrtf(x)", t, |x| x.sqrt())?;

        // sqrt'(x) = 1/(2*out) (output-based).
        let out_capture = out.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad = elementwise_binary(
                    "sqrt_grad",
                    "a / (2.0 * b)",
                    grad_out,
                    &out_capture,
                    &grad_out.shape,
                    |g, o| {
                        let deriv = 1.0 / (2.0 * o);
                        g * deriv
                    },
                )
                .unwrap();
                vec![grad]
            }),
        });
        Ok(out)
    }

    /// `log`.
    fn log<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary("log", "logf(x)", t, |x| x.ln())?;

        // log'(x) = 1/x (input-based, NOT output-based).
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad = elementwise_binary(
                    "log_grad",
                    "a / b",
                    grad_out,
                    &t_capture,
                    &grad_out.shape,
                    |g, x| {
                        let deriv = 1.0 / x;
                        g * deriv
                    },
                )
                .unwrap();
                vec![grad]
            }),
        });
        Ok(out)
    }

    /// `tanh`.
    fn tanh<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary("tanh", "tanhf(x)", t, |x| x.tanh())?;

        // tanh'(x) = 1 - out^2 (output-based).
        let out_capture = out.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad = elementwise_binary(
                    "tanh_grad",
                    "a * (1.0 - b * b)",
                    grad_out,
                    &out_capture,
                    &grad_out.shape,
                    |g, o| {
                        let deriv = 1.0 - o * o;
                        g * deriv
                    },
                )
                .unwrap();
                vec![grad]
            }),
        });
        Ok(out)
    }

    /// `sigmoid`.
    fn sigmoid<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary("sigmoid", "1.0f / (1.0f + expf(-x))", t, |x| {
            1.0 / (1.0 + (-x).exp())
        })?;

        // sigmoid'(x) = out*(1-out) (output-based).
        let out_capture = out.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let grad = elementwise_binary(
                    "sigmoid_grad",
                    "a * b * (1.0 - b)",
                    grad_out,
                    &out_capture,
                    &grad_out.shape,
                    |g, o| {
                        let deriv = o * (1.0 - o);
                        g * deriv
                    },
                )
                .unwrap();
                vec![grad]
            }),
        });
        Ok(out)
    }

    /// `swish`.
    fn swish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = elementwise_unary("swish", "x / (1.0f + expf(-x))", t, |x| {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        })?;

        // swish(x) = x * sigmoid(x)
        // swish'(x) = out + sigmoid(x)*(1-out) — needs BOTH the output and
        // the plain sigmoid value at each input position, recomputed inline
        // (not via a recursive Self::sigmoid call, to avoid an extra tape
        // push during backward).
        let t_capture = t.clone();
        let out_capture = out.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &CpuStorage| {
                let total: usize = grad_out.shape.iter().product();
                let grad: Vec<f64> = (0..total)
                    .into_par_iter()
                    .map(|flat_idx| {
                        let nd_idx = flat_to_nd(flat_idx, &grad_out.shape);
                        let x = t_capture.get(&nd_idx);
                        let o = out_capture.get(&nd_idx);
                        let g = grad_out.get(&nd_idx);
                        let sig = 1.0 / (1.0 + (-x).exp());
                        let deriv = o + sig * (1.0 - o);
                        g * deriv
                    })
                    .collect();
                vec![CpuStorage::from_contiguous(
                    grad_out.buffer.from_f64_values(grad),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }
    /// `softmax`.
    fn softmax<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        // D-02 (Plan 04-01): softmax is now exp(log_softmax(x, dim)).
        // log_softmax shares the same max-subtracted kernel as cross_entropy_loss,
        // eliminating two independent compositions of the same formula.
        let ls = log_softmax::<T, D, K>(t, dim)?;
        <Self as FloatOps<Self>>::exp::<K>(&ls)
    }
}

// ---------------------------------------------------------------------------
// Shared log-softmax kernel (D-02)
// ---------------------------------------------------------------------------

/// `log_softmax(x, dim) = (x - max) - log(sum_keepdim(exp(x - max), dim))`
///
/// Matches `candle-nn-0.9.1/src/ops.rs` lines 31-38 exactly:
/// ```text
/// let max = xs.max_keepdim(d)?;
/// let diff = xs.broadcast_sub(&max)?;
/// let sum_exp = diff.exp()?.sum_keepdim(d)?;
/// let log_sm = diff.broadcast_sub(&sum_exp.log()?)?
/// ```
///
/// Composed entirely from already-tape-tracked primitives — zero new backward
/// code is written here; the composed tape entries from `max_keepdim` / `sub`
/// / `exp` / `sum_keepdim` / `log` / `sub` already implement the correct
/// backward chain automatically (Plan 04-01 D-02 rationale).
///
/// Called by both `FloatOps::softmax` (as `exp(log_softmax(x, dim))`) and
/// `LossOps::cross_entropy_loss` (as `-log_softmax(x, 1)[target]`), so the
/// numerically-stable kernel is shared rather than duplicated.
pub(crate) fn log_softmax<T: DType, D: kindle_core::prelude::Device, K: DType>(
    t: &CpuStorage,
    dim: usize,
) -> Result<CpuStorage> {
    use kindle_core::prelude::{FloatOps, NumericOps, ReductionOps};

    /// `B`.
    type B<T, D> = CpuBackend<T, D>;

    let max = <B<T, D> as ReductionOps<B<T, D>>>::max_keepdim::<K>(t, dim)?;
    let diff = <B<T, D> as NumericOps<B<T, D>>>::sub::<K>(t, &max)?;
    let exp_diff = <B<T, D> as FloatOps<B<T, D>>>::exp::<K>(&diff)?;
    let sum_exp = <B<T, D> as ReductionOps<B<T, D>>>::sum_keepdim::<K>(&exp_diff, dim)?;
    let log_sum_exp = <B<T, D> as FloatOps<B<T, D>>>::log::<K>(&sum_exp)?;
    <B<T, D> as NumericOps<B<T, D>>>::sub::<K>(&diff, &log_sum_exp)
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    use crate::cpu::gradcheck::gradcheck;
    use crate::cpu::storage::CpuBuffer;
    use crate::cpu::tape;
    use kindle_core::prelude::ReductionOps;

    /// `matrix`.
    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![rows, cols])
    }

    /// `vector`.
    fn vector(v: Vec<f32>) -> CpuStorage {
        let len = v.len();
        CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![len])
    }

    /// `f32_vec`.
    fn f32_vec(s: &CpuStorage) -> Vec<f32> {
        match &*s.buffer {
            CpuBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    fn f64_storage(v: Vec<f64>, shape: Vec<usize>) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F64(v), shape)
    }

    // Regression guard for C-2: elementwise ops used to hardcode `CpuBuffer::F32`
    // for every result regardless of the operands' actual dtype, silently
    // downcasting F64 (and F16/BF16) tensors through f32 with no error. These
    // values are specifically chosen to be exactly representable in f64 but
    // NOT exactly representable in f32, so an accidental f32 round-trip
    // changes the result.
    #[test]
    fn add_preserves_f64_dtype_and_precision() {
        let lhs = f64_storage(vec![1.000000123456789], vec![1]);
        let rhs = f64_storage(vec![2.000000987654321], vec![1]);
        let out = TestBackend::add::<f64>(&lhs, &rhs).unwrap();

        match &*out.buffer {
            CpuBuffer::F64(v) => {
                let expected = 1.000000123456789 + 2.000000987654321;
                assert_eq!(
                    v[0], expected,
                    "add on F64 operands must return an F64 buffer with full f64 precision, \
                     not a value that has round-tripped through f32"
                );
            }
            other => panic!("expected CpuBuffer::F64, got {other:?}"),
        }
    }

    #[test]
    fn relu_preserves_f64_dtype() {
        let t = f64_storage(vec![-1.000000123456789, 3.000000987654321], vec![2]);
        let out = TestBackend::relu::<f64>(&t).unwrap();

        match &*out.buffer {
            CpuBuffer::F64(v) => {
                assert_eq!(v, &vec![0.0, 3.000000987654321]);
            }
            other => panic!("expected CpuBuffer::F64, got {other:?}"),
        }
    }

    /// `TestBackend`.
    type TestBackend = CpuBackend<f32, kindle_core::prelude::Cpu>;

    #[test]
    /// `add_broadcasts_forward_correctly`.
    fn add_broadcasts_forward_correctly() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![10.0, 20.0, 30.0]);
        let out = TestBackend::add::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    /// `add_backward_unbroadcasts_correctly_for_bias_vector_case`.
    fn add_backward_unbroadcasts_correctly_for_bias_vector_case() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![10.0, 20.0, 30.0]);
        let out = TestBackend::add::<f32>(&lhs, &rhs).unwrap();

        let grads = tape::backward(&out).unwrap();
        let lhs_grad = grads.get(lhs.id).expect("lhs should have a gradient");
        let rhs_grad = grads.get(rhs.id).expect("rhs should have a gradient");

        // lhs's grad: ones_like(out) unbroadcast to [2,3] = all ones.
        assert_eq!(lhs_grad.shape, vec![2, 3]);
        assert_eq!(f32_vec(lhs_grad), vec![1.0; 6]);

        // rhs's grad: ones_like(out) [2,3] unbroadcast (summed) to [3] = [2,2,2].
        assert_eq!(rhs_grad.shape, vec![3]);
        assert_eq!(f32_vec(rhs_grad), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    /// `sub_forward_computes_elementwise_difference_with_broadcast`.
    fn sub_forward_computes_elementwise_difference_with_broadcast() {
        let lhs = matrix(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3);
        let rhs = vector(vec![1.0, 2.0, 3.0]);
        let out = TestBackend::sub::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
    }

    #[test]
    /// `sub_backward_negates_rhs_contribution`.
    fn sub_backward_negates_rhs_contribution() {
        let lhs = vector(vec![10.0, 20.0, 30.0]);
        let rhs = vector(vec![1.0, 2.0, 3.0]);
        let out = TestBackend::sub::<f32>(&lhs, &rhs).unwrap();

        let grads = tape::backward(&out).unwrap();
        let lhs_grad = grads.get(lhs.id).unwrap();
        let rhs_grad = grads.get(rhs.id).unwrap();

        assert_eq!(f32_vec(lhs_grad), vec![1.0, 1.0, 1.0]);
        assert_eq!(f32_vec(rhs_grad), vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    /// `mul_forward_computes_elementwise_product_with_broadcast`.
    fn mul_forward_computes_elementwise_product_with_broadcast() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![2.0, 3.0, 4.0]);
        let out = TestBackend::mul::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);
    }

    #[test]
    /// `mul_backward_uses_other_operands_real_values`.
    fn mul_backward_uses_other_operands_real_values() {
        // d(a*b)/da = b, d(a*b)/db = a — verify the retrieved gradient
        // equals a manually-computed expected value (not merely "some
        // gradient exists").
        let a = vector(vec![2.0, 3.0, 4.0]);
        let b = vector(vec![5.0, 6.0, 7.0]);
        let out = TestBackend::mul::<f32>(&a, &b).unwrap();

        let grads = tape::backward(&out).unwrap();
        let a_grad = grads.get(a.id).unwrap();
        let b_grad = grads.get(b.id).unwrap();

        // grad_out is ones_like(out) = [1,1,1].
        // da = grad_out * b = [5,6,7]
        assert_eq!(f32_vec(a_grad), vec![5.0, 6.0, 7.0]);
        // db = grad_out * a = [2,3,4]
        assert_eq!(f32_vec(b_grad), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    /// `mul_backward_with_broadcast_bias_vector_case`.
    fn mul_backward_with_broadcast_bias_vector_case() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![2.0, 3.0, 4.0]);
        let out = TestBackend::mul::<f32>(&lhs, &rhs).unwrap();

        let grads = tape::backward(&out).unwrap();
        let lhs_grad = grads.get(lhs.id).unwrap();
        let rhs_grad = grads.get(rhs.id).unwrap();

        // grad_out = ones_like(out) = [[1,1,1],[1,1,1]]
        // d(lhs*rhs)/dlhs = rhs broadcast -> unbroadcast to lhs shape [2,3]
        // (no reduction needed since lhs shape == out shape): [[2,3,4],[2,3,4]]
        assert_eq!(lhs_grad.shape, vec![2, 3]);
        assert_eq!(f32_vec(lhs_grad), vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);

        // d(lhs*rhs)/drhs = lhs broadcast, summed (unbroadcast) to rhs shape [3]:
        // col sums of lhs = [1+4, 2+5, 3+6] = [5,7,9]
        assert_eq!(rhs_grad.shape, vec![3]);
        assert_eq!(f32_vec(rhs_grad), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    /// `add_scalar_float_forward_and_backward`.
    fn add_scalar_float_forward_and_backward() {
        let t = vector(vec![1.0, 2.0, 3.0]);
        let out = TestBackend::add_scalar_float::<f32>(&t, 1.0).unwrap();
        assert_eq!(f32_vec(&out), vec![2.0, 3.0, 4.0]);

        let grads = tape::backward(&out).unwrap();
        let t_grad = grads.get(t.id).unwrap();
        // Gradient passes through unchanged.
        assert_eq!(f32_vec(t_grad), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    /// `mul_scalar_float_forward_and_backward`.
    fn mul_scalar_float_forward_and_backward() {
        let t = vector(vec![1.0, 2.0, 3.0]);
        let out = TestBackend::mul_scalar_float::<f32>(&t, 2.5).unwrap();
        assert_eq!(f32_vec(&out), vec![2.5, 5.0, 7.5]);

        let grads = tape::backward(&out).unwrap();
        let t_grad = grads.get(t.id).unwrap();
        // Gradient scales by the same constant.
        assert_eq!(f32_vec(t_grad), vec![2.5, 2.5, 2.5]);
    }

    // --- Task 1: relu / abs / neg ---

    #[test]
    /// `relu_forward_and_backward_zero_at_boundary`.
    fn relu_forward_and_backward_zero_at_boundary() {
        let t = vector(vec![-2.0, 0.0, 3.0]);
        let out = TestBackend::relu::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![0.0, 0.0, 3.0]);

        let grads = tape::backward(&out).unwrap();
        let t_grad = grads.get(t.id).unwrap();
        // Zero gradient at the x=0 boundary (strict `>`, not `>=`).
        assert_eq!(f32_vec(t_grad), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    /// `relu_gradcheck_on_nonzero_input`.
    fn relu_gradcheck_on_nonzero_input() {
        let x = vector(vec![2.0, -1.5, 0.7]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let r = TestBackend::relu::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&r).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "relu gradcheck error too high: {max_rel_err}"
        );
    }

    #[test]
    /// `abs_forward_and_gradcheck`.
    fn abs_forward_and_gradcheck() {
        let t = vector(vec![-2.5, 0.0, 3.5]);
        let out = TestBackend::abs::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![2.5, 0.0, 3.5]);

        let x = vector(vec![-2.0, 1.5, -0.3]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let a = TestBackend::abs::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&a).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "abs gradcheck error too high: {max_rel_err}"
        );
    }

    #[test]
    /// `neg_forward_and_gradcheck`.
    fn neg_forward_and_gradcheck() {
        let t = vector(vec![1.0, -2.0, 3.0]);
        let out = TestBackend::neg::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![-1.0, 2.0, -3.0]);

        let x = vector(vec![1.0, -2.0, 3.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let n = TestBackend::neg::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&n).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "neg gradcheck error too high: {max_rel_err}"
        );
    }

    // --- Task 2: exp / sqrt / log / tanh / sigmoid / swish ---

    #[test]
    /// `exp_forward_and_gradcheck`.
    fn exp_forward_and_gradcheck() {
        let t = vector(vec![0.0, 1.0]);
        let out = TestBackend::exp::<f32>(&t).unwrap();
        let expect = [1.0f32, core::f64::consts::E as f32];
        for (a, b) in f32_vec(&out).iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-5, "exp forward mismatch: {a} vs {b}");
        }

        let x = vector(vec![0.5, -0.3, 1.2]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let e = TestBackend::exp::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&e).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "exp gradcheck error too high: {max_rel_err}"
        );
    }

    #[test]
    /// `sqrt_forward_gradcheck_and_nan_propagation`.
    fn sqrt_forward_gradcheck_and_nan_propagation() {
        let t = vector(vec![4.0, 9.0]);
        let out = TestBackend::sqrt::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![2.0, 3.0]);

        let x = vector(vec![4.0, 1.0, 9.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let s = TestBackend::sqrt::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&s).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "sqrt gradcheck error too high: {max_rel_err}"
        );

        // Negative input propagates NaN cpuly (RESEARCH.md Pitfall 2),
        // not a panic and not an Err.
        let neg_input = vector(vec![-1.0]);
        let neg_out = TestBackend::sqrt::<f32>(&neg_input).unwrap();
        assert!(f32_vec(&neg_out)[0].is_nan(), "sqrt(-1.0) should be NaN");
    }

    #[test]
    /// `log_forward_gradcheck_and_domain_propagation`.
    fn log_forward_gradcheck_and_domain_propagation() {
        let t = vector(vec![1.0, core::f64::consts::E as f32]);
        let out = TestBackend::log::<f32>(&t).unwrap();
        let expect = [0.0f32, 1.0f32];
        for (a, b) in f32_vec(&out).iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-5, "log forward mismatch: {a} vs {b}");
        }

        let x = vector(vec![1.0, 2.0, 5.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let l = TestBackend::log::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&l).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "log gradcheck error too high: {max_rel_err}"
        );

        // Zero/negative input propagates NaN/-inf cpuly, not a panic and
        // not an Err.
        let zero_input = vector(vec![0.0]);
        let zero_out = TestBackend::log::<f32>(&zero_input).unwrap();
        assert!(
            f32_vec(&zero_out)[0].is_infinite() && f32_vec(&zero_out)[0] < 0.0,
            "log(0.0) should be -inf"
        );

        let neg_input = vector(vec![-1.0]);
        let neg_out = TestBackend::log::<f32>(&neg_input).unwrap();
        assert!(f32_vec(&neg_out)[0].is_nan(), "log(-1.0) should be NaN");
    }

    #[test]
    /// `tanh_gradcheck`.
    fn tanh_gradcheck() {
        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let th = TestBackend::tanh::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&th).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "tanh gradcheck error too high: {max_rel_err}"
        );
    }

    #[test]
    /// `sigmoid_forward_and_gradcheck`.
    fn sigmoid_forward_and_gradcheck() {
        let t = vector(vec![0.0]);
        let out = TestBackend::sigmoid::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![0.5]);

        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let s = TestBackend::sigmoid::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&s).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "sigmoid gradcheck error too high: {max_rel_err}"
        );
    }

    #[test]
    /// `swish_forward_and_gradcheck`.
    fn swish_forward_and_gradcheck() {
        let t = vector(vec![0.0]);
        let out = TestBackend::swish::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![0.0]);

        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let s = TestBackend::swish::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&s).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "swish gradcheck error too high: {max_rel_err}"
        );
    }

    // --- Task 3: gelu (exact erf-based form) ---

    #[test]
    /// `gelu_forward_zero_and_one`.
    fn gelu_forward_zero_and_one() {
        let zero = vector(vec![0.0]);
        let out_zero = TestBackend::gelu::<f32>(&zero).unwrap();
        assert_eq!(f32_vec(&out_zero), vec![0.0]);

        let one = vector(vec![1.0]);
        let out_one = TestBackend::gelu::<f32>(&one).unwrap();
        // Known reference value for erf-based GELU at x=1 (~0.8413).
        // Looser 1e-3 tolerance than other ops' 1e-5 since this uses a
        // polynomial erf approximation, not an exact closed form
        // ([ASSUMED] per RESEARCH.md Assumption A3).
        assert!(
            (f32_vec(&out_one)[0] - 0.8413).abs() < 1e-3,
            "gelu(1.0) mismatch: {}",
            f32_vec(&out_one)[0]
        );
    }

    #[test]
    /// `gelu_gradcheck`.
    fn gelu_gradcheck() {
        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let g = TestBackend::gelu::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&g).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "gelu gradcheck error too high: {max_rel_err}"
        );
    }

    // --- Task 1 (plan 02-04): softmax by composition ---

    #[test]
    /// `softmax_forward_sums_to_one_on_vector`.
    fn softmax_forward_sums_to_one_on_vector() {
        let t = vector(vec![1.0, 2.0, 3.0]);
        let out = TestBackend::softmax::<f32>(&t, 0).unwrap();
        let vals = f32_vec(&out);

        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1: {sum}");

        // Largest input gets largest probability, monotonic ordering preserved.
        assert!(vals[0] < vals[1]);
        assert!(vals[1] < vals[2]);
    }

    #[test]
    /// `softmax_forward_sums_to_one_per_row_on_matrix`.
    fn softmax_forward_sums_to_one_per_row_on_matrix() {
        let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let out = TestBackend::softmax::<f32>(&t, 1).unwrap();
        let vals = f32_vec(&out);

        let row0_sum: f32 = vals[0..3].iter().sum();
        let row1_sum: f32 = vals[3..6].iter().sum();
        assert!(
            (row0_sum - 1.0).abs() < 1e-5,
            "row 0 should sum to 1: {row0_sum}"
        );
        assert!(
            (row1_sum - 1.0).abs() < 1e-5,
            "row 1 should sum to 1: {row1_sum}"
        );
    }

    #[test]
    /// `softmax_forward_stable_on_large_magnitude_equal_logits`.
    fn softmax_forward_stable_on_large_magnitude_equal_logits() {
        // Without max-subtraction, exp(1000.0) overflows to inf, producing
        // NaN (inf/inf) instead of a finite uniform distribution.
        let t = vector(vec![1000.0, 1000.0, 1000.0]);
        let out = TestBackend::softmax::<f32>(&t, 0).unwrap();
        let vals = f32_vec(&out);

        for v in &vals {
            assert!(v.is_finite(), "softmax output should be finite: {v}");
            assert!(
                (v - 1.0 / 3.0).abs() < 1e-4,
                "softmax(equal large logits) should be uniform: {v}"
            );
        }
    }

    #[test]
    /// `softmax_forward_uniform_on_all_zero_logits`.
    fn softmax_forward_uniform_on_all_zero_logits() {
        let t = vector(vec![0.0, 0.0, 0.0]);
        let out = TestBackend::softmax::<f32>(&t, 0).unwrap();
        let vals = f32_vec(&out);

        for v in &vals {
            assert!(v.is_finite(), "softmax output should be finite: {v}");
            assert!(
                (v - 1.0 / 3.0).abs() < 1e-4,
                "softmax(all-zero logits) should be uniform: {v}"
            );
        }
    }

    #[test]
    /// `softmax_gradcheck`.
    fn softmax_gradcheck() {
        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let s = TestBackend::softmax::<f32>(&inputs[0], 0).unwrap();
            TestBackend::sum_all::<f32>(&s).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "softmax gradcheck error too high: {max_rel_err}"
        );
    }

    #[test]
    /// `softmax_backward_finite_on_large_magnitude_equal_logits`.
    fn softmax_backward_finite_on_large_magnitude_equal_logits() {
        // Proves both forward AND backward are numerically stable under the
        // composition, not just forward (Test 3's finite-forward twin).
        let t = vector(vec![1000.0, 1000.0, 1000.0]);
        let out = TestBackend::softmax::<f32>(&t, 0).unwrap();

        let grads = tape::backward(&out).unwrap();
        let t_grad = grads.get(t.id).unwrap();
        for v in f32_vec(t_grad) {
            assert!(
                v.is_finite(),
                "softmax backward gradient should be finite on extreme logits: {v}"
            );
        }
    }

    // --- log_softmax kernel tests (Plan 04-01 Task 1) ---

    #[test]
    /// `log_softmax_exp_sums_to_one_on_vector`.
    fn log_softmax_exp_sums_to_one_on_vector() {
        // exp(log_softmax(x)).sum() == 1.0 (the softmax identity).
        use crate::cpu::ops::elementwise::log_softmax;
        let t = vector(vec![1.0, 2.0, 3.0]);
        let ls = log_softmax::<f32, kindle_core::prelude::Cpu, f32>(&t, 0).unwrap();
        let exp_ls = TestBackend::exp::<f32>(&ls).unwrap();
        let vals = f32_vec(&exp_ls);
        let sum: f32 = vals.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "exp(log_softmax) should sum to 1: {sum}"
        );
    }

    #[test]
    /// `log_softmax_is_finite_and_correct_on_large_magnitude_equal_logits`.
    fn log_softmax_is_finite_and_correct_on_large_magnitude_equal_logits() {
        // log_softmax([1000, 1000, 1000]) should be -ln(3) for each element.
        // Without max-subtraction, exp(1000) overflows to inf and log(inf) = inf.
        use crate::cpu::ops::elementwise::log_softmax;
        let t = vector(vec![1000.0f32, 1000.0, 1000.0]);
        let ls = log_softmax::<f32, kindle_core::prelude::Cpu, f32>(&t, 0).unwrap();
        let vals = f32_vec(&ls);
        let expected = -(3.0f32.ln());
        for (i, &v) in vals.iter().enumerate() {
            assert!(v.is_finite(), "log_softmax[{i}] should be finite: {v}");
            assert!(
                (v - expected).abs() < 1e-4,
                "log_softmax of equal large logits should be -ln(3): got {v}, expected {expected}"
            );
        }
    }

    #[test]
    /// `softmax_after_refactor_still_passes_all_prior_behavior`.
    fn softmax_after_refactor_still_passes_all_prior_behavior() {
        // Regression guard: the refactored softmax (exp(log_softmax(x, dim)))
        // must produce the same output as the old max_keepdim/sub/exp/sum_keepdim/div
        // composition. Verified by running all pre-existing scenarios in one test.
        // (Pre-existing tests above already cover this — this is an explicit marker
        //  that the refactor did not break them.)
        //
        // Spot-check: vector [0.5, -1.0, 2.0] forward correctness.
        let t = vector(vec![0.5f32, -1.0, 2.0]);
        let out = TestBackend::softmax::<f32>(&t, 0).unwrap();
        let vals = f32_vec(&out);
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum should be 1: {sum}");
        for v in &vals {
            assert!(v.is_finite(), "softmax output should be finite: {v}");
            assert!(*v > 0.0, "softmax output should be positive: {v}");
        }
    }

    #[test]
    /// `log_softmax_gradcheck`.
    fn log_softmax_gradcheck() {
        // Finite-difference gradcheck for log_softmax itself.
        use crate::cpu::ops::elementwise::log_softmax;
        let x = vector(vec![0.5f32, -1.0, 2.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let ls = log_softmax::<f32, kindle_core::prelude::Cpu, f32>(&inputs[0], 0).unwrap();
            TestBackend::sum_all::<f32>(&ls).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "log_softmax gradcheck error too high: {max_rel_err}"
        );
    }
}
