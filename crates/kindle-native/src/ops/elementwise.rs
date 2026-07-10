//! `NumericOps` (`add`/`sub`/`mul`/`div`) and `FloatOps::{add_scalar_float,
//! mul_scalar_float}` for `NativeBackend<T, D>`.
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

use crate::NativeBackend;
use crate::storage::{NativeBuffer, NativeStorage};
use crate::tape::{self, TapeEntry};

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

/// Read `storage` as `f64` at `storage`'s own (already-resolved) logical
/// index, given the output-space index and shape.
fn read_broadcast(storage: &NativeStorage, out_idx: &[usize], out_shape: &[usize]) -> f64 {
    let idx = broadcast_index(out_idx, out_shape, &storage.shape);
    storage.get(&idx)
}

/// Build a contiguous `NativeStorage` by applying `f(lhs_val, rhs_val)` over
/// every logical index in `out_shape`, reading each operand through its own
/// broadcast-resolved index (no pre-materialized broadcast copy).
fn elementwise_binary(
    lhs: &NativeStorage,
    rhs: &NativeStorage,
    out_shape: &[usize],
    f: impl Fn(f64, f64) -> f64,
) -> NativeStorage {
    let total: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        let a = read_broadcast(lhs, &idx, out_shape);
        let b = read_broadcast(rhs, &idx, out_shape);
        out.push(f(a, b) as f32);
        if !out_shape.is_empty() {
            increment_index(&mut idx, out_shape);
        }
    }
    NativeStorage::from_contiguous(NativeBuffer::F32(out), out_shape.to_vec())
}

/// Elementwise negate (used by `sub`'s backward rule: rhs receives the
/// negated incoming gradient before unbroadcasting).
fn negate(t: &NativeStorage) -> NativeStorage {
    let total: usize = t.shape.iter().product();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; t.shape.len()];
    for _ in 0..total {
        out.push(-t.get(&idx) as f32);
        if !t.shape.is_empty() {
            increment_index(&mut idx, &t.shape);
        }
    }
    NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone())
}

/// Elementwise multiply of two ALREADY-shape-matching (or one of them
/// broadcastable to the other's shape) storages — used by `mul`'s backward
/// rule to compute `grad_out * other_operand`, aligned to `grad_out`'s shape.
fn mul_elementwise_broadcast(grad_out: &NativeStorage, other: &NativeStorage) -> NativeStorage {
    elementwise_binary(grad_out, other, &grad_out.shape, |a, b| a * b)
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

impl<T: DType, D: kindle_core::prelude::Device> NumericOps<Self> for NativeBackend<T, D> {
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary(lhs, rhs, &out_shape, |a, b| a + b);

        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![
                    tape::unbroadcast(grad_out, &lhs_shape).expect("unbroadcast lhs (add)"),
                    tape::unbroadcast(grad_out, &rhs_shape).expect("unbroadcast rhs (add)"),
                ]
            }),
        });
        Ok(out)
    }

    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary(lhs, rhs, &out_shape, |a, b| a - b);

        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                vec![
                    tape::unbroadcast(grad_out, &lhs_shape).expect("unbroadcast lhs (sub)"),
                    tape::unbroadcast(&negate(grad_out), &rhs_shape)
                        .expect("unbroadcast rhs (sub)"),
                ]
            }),
        });
        Ok(out)
    }

    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary(lhs, rhs, &out_shape, |a, b| a * b);

        // Capture cloned copies of lhs/rhs's NativeStorage (cheap, Rc-backed)
        // since the backward closure needs their VALUES, not just shapes.
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let grad_lhs = mul_elementwise_broadcast(grad_out, &rhs_capture);
                let grad_rhs = mul_elementwise_broadcast(grad_out, &lhs_capture);
                vec![
                    tape::unbroadcast(&grad_lhs, &lhs_shape).expect("unbroadcast lhs (mul)"),
                    tape::unbroadcast(&grad_rhs, &rhs_shape).expect("unbroadcast rhs (mul)"),
                ]
            }),
        });
        Ok(out)
    }

    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out = elementwise_binary(lhs, rhs, &out_shape, |a, b| a / b);

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
            backward: Box::new(move |grad_out: &NativeStorage| {
                // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
                let grad_lhs =
                    elementwise_binary(grad_out, &rhs_capture, &grad_out.shape, |g, r| g / r);
                // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = grad_out * (-lhs/rhs^2)
                let grad_rhs = elementwise_binary(
                    grad_out,
                    &elementwise_binary(&lhs_capture, &rhs_capture, &grad_out.shape, |l, r| {
                        -l / (r * r)
                    }),
                    &grad_out.shape,
                    |g, dr| g * dr,
                );
                vec![
                    tape::unbroadcast(&grad_lhs, &lhs_shape).expect("unbroadcast lhs (div)"),
                    tape::unbroadcast(&grad_rhs, &rhs_shape).expect("unbroadcast rhs (div)"),
                ]
            }),
        });
        Ok(out)
    }
}

impl<T: DType, D: kindle_core::prelude::Device> FloatOps<Self> for NativeBackend<T, D> {
    fn add_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            out.push((t.get(&idx) + scalar) as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            // Gradient passes through unchanged (same shape, no unbroadcast
            // needed — scalar ops don't change shape).
            backward: Box::new(move |grad_out: &NativeStorage| vec![grad_out.clone()]),
        });
        Ok(out)
    }

    fn mul_scalar_float<K: DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            out.push((t.get(&idx) * scalar) as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            // Gradient scales by the same constant (same shape, no
            // unbroadcast needed).
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut scaled = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    scaled.push((grad_out.get(&idx) * scalar) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(scaled),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    fn relu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            out.push(t.get(&idx).max(0.0) as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        // relu'(x) = 1 if x > 0 else 0 (input-based; strict `>`, zero
        // gradient at the x=0 boundary).
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut grad = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    let x = t_capture.get(&idx);
                    let deriv = if x > 0.0 { 1.0 } else { 0.0 };
                    grad.push((grad_out.get(&idx) * deriv) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(grad),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    fn gelu<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            let x = t.get(&idx);
            out.push((x * 0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))) as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        // gelu(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
        // gelu'(x) = 0.5*(1+erf(x/sqrt(2))) + x * (1/sqrt(2*pi)) * exp(-x^2/2)
        // (input-based — not simplifiable purely in terms of the output).
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut grad = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    let x = t_capture.get(&idx);
                    let cdf = 0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2));
                    let pdf = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-x * x / 2.0).exp();
                    let deriv = cdf + x * pdf;
                    grad.push((grad_out.get(&idx) * deriv) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(grad),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    fn abs<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            out.push(t.get(&idx).abs() as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        // abs'(x) = sign(x) (input-based).
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut grad = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    let x = t_capture.get(&idx);
                    let deriv = if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    grad.push((grad_out.get(&idx) * deriv) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(grad),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    fn exp<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            out.push(t.get(&idx).exp() as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        // exp'(x) = out (output-based).
        let out_capture = out.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut grad = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    let deriv = out_capture.get(&idx);
                    grad.push((grad_out.get(&idx) * deriv) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(grad),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    fn neg<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let out = negate(t);

        // neg'(x) = -1 (constant; no input capture needed).
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| vec![negate(grad_out)]),
        });
        Ok(out)
    }

    fn sqrt<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            // No domain guard: `f64::sqrt`'s native NaN propagation on
            // negative input flows through unchanged (RESEARCH.md Pitfall 2).
            out.push(t.get(&idx).sqrt() as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        // sqrt'(x) = 1/(2*out) (output-based).
        let out_capture = out.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut grad = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    let deriv = 1.0 / (2.0 * out_capture.get(&idx));
                    grad.push((grad_out.get(&idx) * deriv) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(grad),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    fn log<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            // No domain guard: `f64::ln`'s native NaN/-inf propagation on
            // zero/negative input flows through unchanged (RESEARCH.md
            // Pitfall 2).
            out.push(t.get(&idx).ln() as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        // log'(x) = 1/x (input-based, NOT output-based).
        let t_capture = t.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut grad = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    let deriv = 1.0 / t_capture.get(&idx);
                    grad.push((grad_out.get(&idx) * deriv) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(grad),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    fn tanh<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            out.push(t.get(&idx).tanh() as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        // tanh'(x) = 1 - out^2 (output-based).
        let out_capture = out.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut grad = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    let o = out_capture.get(&idx);
                    let deriv = 1.0 - o * o;
                    grad.push((grad_out.get(&idx) * deriv) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(grad),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    fn sigmoid<K: DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            let x = t.get(&idx);
            out.push((1.0 / (1.0 + (-x).exp())) as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

        // sigmoid'(x) = out*(1-out) (output-based).
        let out_capture = out.clone();
        let (t_id, out_id) = (t.id, out.id);
        tape::push(TapeEntry {
            output_id: out_id,
            input_ids: vec![t_id],
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut grad = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    let o = out_capture.get(&idx);
                    let deriv = o * (1.0 - o);
                    grad.push((grad_out.get(&idx) * deriv) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(grad),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }

    fn swish<K: DType>(t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<K>> {
        let total: usize = t.shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; t.shape.len()];
        for _ in 0..total {
            let x = t.get(&idx);
            let sig = 1.0 / (1.0 + (-x).exp());
            out.push((x * sig) as f32);
            if !t.shape.is_empty() {
                increment_index(&mut idx, &t.shape);
            }
        }
        let out = NativeStorage::from_contiguous(NativeBuffer::F32(out), t.shape.clone());

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
            backward: Box::new(move |grad_out: &NativeStorage| {
                let total: usize = grad_out.shape.iter().product();
                let mut grad = Vec::with_capacity(total);
                let mut idx = vec![0usize; grad_out.shape.len()];
                for _ in 0..total {
                    let x = t_capture.get(&idx);
                    let o = out_capture.get(&idx);
                    let sig = 1.0 / (1.0 + (-x).exp());
                    let deriv = o + sig * (1.0 - o);
                    grad.push((grad_out.get(&idx) * deriv) as f32);
                    if !grad_out.shape.is_empty() {
                        increment_index(&mut idx, &grad_out.shape);
                    }
                }
                vec![NativeStorage::from_contiguous(
                    NativeBuffer::F32(grad),
                    grad_out.shape.clone(),
                )]
            }),
        });
        Ok(out)
    }
    fn softmax<K: DType>(
        _t: &<Self as Backend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        unimplemented!("softmax not implemented for NativeBackend")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape;
    use crate::testutil::gradcheck;
    use kindle_core::prelude::ReductionOps;

    fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> NativeStorage {
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![rows, cols])
    }

    fn vector(v: Vec<f32>) -> NativeStorage {
        let len = v.len();
        NativeStorage::from_contiguous(NativeBuffer::F32(v), vec![len])
    }

    fn f32_vec(s: &NativeStorage) -> Vec<f32> {
        match &*s.buffer {
            NativeBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    type TestBackend = NativeBackend<f32, kindle_core::prelude::Cpu>;

    #[test]
    fn add_broadcasts_forward_correctly() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![10.0, 20.0, 30.0]);
        let out = TestBackend::add::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
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
    fn sub_forward_computes_elementwise_difference_with_broadcast() {
        let lhs = matrix(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3);
        let rhs = vector(vec![1.0, 2.0, 3.0]);
        let out = TestBackend::sub::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
    }

    #[test]
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
    fn mul_forward_computes_elementwise_product_with_broadcast() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![2.0, 3.0, 4.0]);
        let out = TestBackend::mul::<f32>(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);
    }

    #[test]
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
    fn relu_gradcheck_on_nonzero_input() {
        let x = vector(vec![2.0, -1.5, 0.7]);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
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
    fn abs_forward_and_gradcheck() {
        let t = vector(vec![-2.5, 0.0, 3.5]);
        let out = TestBackend::abs::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![2.5, 0.0, 3.5]);

        let x = vector(vec![-2.0, 1.5, -0.3]);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
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
    fn neg_forward_and_gradcheck() {
        let t = vector(vec![1.0, -2.0, 3.0]);
        let out = TestBackend::neg::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![-1.0, 2.0, -3.0]);

        let x = vector(vec![1.0, -2.0, 3.0]);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
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
    fn exp_forward_and_gradcheck() {
        let t = vector(vec![0.0, 1.0]);
        let out = TestBackend::exp::<f32>(&t).unwrap();
        let expect = [1.0f32, std::f64::consts::E as f32];
        for (a, b) in f32_vec(&out).iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-5, "exp forward mismatch: {a} vs {b}");
        }

        let x = vector(vec![0.5, -0.3, 1.2]);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
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
    fn sqrt_forward_gradcheck_and_nan_propagation() {
        let t = vector(vec![4.0, 9.0]);
        let out = TestBackend::sqrt::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![2.0, 3.0]);

        let x = vector(vec![4.0, 1.0, 9.0]);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
            let s = TestBackend::sqrt::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&s).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "sqrt gradcheck error too high: {max_rel_err}"
        );

        // Negative input propagates NaN natively (RESEARCH.md Pitfall 2),
        // not a panic and not an Err.
        let neg_input = vector(vec![-1.0]);
        let neg_out = TestBackend::sqrt::<f32>(&neg_input).unwrap();
        assert!(f32_vec(&neg_out)[0].is_nan(), "sqrt(-1.0) should be NaN");
    }

    #[test]
    fn log_forward_gradcheck_and_domain_propagation() {
        let t = vector(vec![1.0, std::f64::consts::E as f32]);
        let out = TestBackend::log::<f32>(&t).unwrap();
        let expect = [0.0f32, 1.0f32];
        for (a, b) in f32_vec(&out).iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-5, "log forward mismatch: {a} vs {b}");
        }

        let x = vector(vec![1.0, 2.0, 5.0]);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
            let l = TestBackend::log::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&l).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "log gradcheck error too high: {max_rel_err}"
        );

        // Zero/negative input propagates NaN/-inf natively, not a panic and
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
    fn tanh_gradcheck() {
        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
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
    fn sigmoid_forward_and_gradcheck() {
        let t = vector(vec![0.0]);
        let out = TestBackend::sigmoid::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![0.5]);

        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
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
    fn swish_forward_and_gradcheck() {
        let t = vector(vec![0.0]);
        let out = TestBackend::swish::<f32>(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![0.0]);

        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
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
    fn gelu_gradcheck() {
        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[NativeStorage]| -> NativeStorage {
            let g = TestBackend::gelu::<f32>(&inputs[0]).unwrap();
            TestBackend::sum_all::<f32>(&g).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "gelu gradcheck error too high: {max_rel_err}"
        );
    }
}
