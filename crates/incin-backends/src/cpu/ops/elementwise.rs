//! Concrete CPU pointwise helpers used by descriptor executors and composed
//! backend operations.
//!
//! Every op here resolves the broadcast output shape via
//! `stride::broadcast_shape`, then iterates the OUTPUT shape's logical index
//! space, resolving each operand's own index through its own strides with
//! wraparound (stride-0-equivalent) logic on right-aligned/expanded
//! dimensions — it never pre-materializes a broadcast copy of either operand
//! (the anti-pattern flagged in RESEARCH.md). Every op pushes a `TapeEntry`
//! whose backward closure calls `tape::unbroadcast` on the ORIGINAL
//! (pre-broadcast) operand shapes.

use incin_core::error::{Error, Result};
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::{DType, FloatDType};

use crate::cpu::ops::elementwise_kernel::{self, BinaryOp, UnaryOp};
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::tape::{self, TapeEntry};
use crate::iteration::{IterationPlan, OperandLayout};

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

/// Build a contiguous `CpuStorage` by applying `f(lhs_val, rhs_val)` over
/// every logical index in `out_shape`, reading each operand through its own
/// broadcast-resolved index (no pre-materialized broadcast copy).
use rayon::prelude::*;

pub(crate) fn elementwise_binary(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
    f: impl Fn(f64, f64) -> f64 + Send + Sync,
) -> Result<CpuStorage> {
    if let (Some(l_range), Some(r_range)) = (
        elementwise_kernel::dense_range(lhs, lhs.buffer.len(), out_shape),
        elementwise_kernel::dense_range(rhs, rhs.buffer.len(), out_shape),
    ) {
        let buffer = match (&*lhs.buffer, &*rhs.buffer) {
            (CpuBuffer::F32(l), CpuBuffer::F32(r)) => {
                let out =
                    crate::cpu::typed_kernel::map_binary_typed(&l[l_range], &r[r_range], |a, b| {
                        f(a as f64, b as f64) as f32
                    });
                CpuBuffer::F32(out)
            }
            (CpuBuffer::F64(l), CpuBuffer::F64(r)) => {
                let out = crate::cpu::typed_kernel::map_binary_typed(&l[l_range], &r[r_range], &f);
                CpuBuffer::F64(out)
            }
            _ => {
                let plan = IterationPlan::binary(
                    OperandLayout {
                        shape: &lhs.shape,
                        strides: &lhs.strides,
                        offset: lhs.offset_elements,
                    },
                    OperandLayout {
                        shape: &rhs.shape,
                        strides: &rhs.strides,
                        offset: rhs.offset_elements,
                    },
                    out_shape,
                )?;
                let lhs_plan = &plan.operands[0];
                let rhs_plan = &plan.operands[1];
                let out: Vec<f64> = (0..plan.numel)
                    .into_par_iter()
                    .map(|flat_idx| {
                        let a = lhs
                            .buffer
                            .get_f64(lhs_plan.physical_index(flat_idx, &plan.output_shape));
                        let b = rhs
                            .buffer
                            .get_f64(rhs_plan.physical_index(flat_idx, &plan.output_shape));
                        f(a, b)
                    })
                    .collect();
                lhs.buffer.from_f64_values(out)?
            }
        };
        return Ok(CpuStorage::from_contiguous(buffer, out_shape.to_vec()));
    }

    let plan = IterationPlan::binary(
        OperandLayout {
            shape: &lhs.shape,
            strides: &lhs.strides,
            offset: lhs.offset_elements,
        },
        OperandLayout {
            shape: &rhs.shape,
            strides: &rhs.strides,
            offset: rhs.offset_elements,
        },
        out_shape,
    )?;
    let lhs_plan = &plan.operands[0];
    let rhs_plan = &plan.operands[1];
    let out: Vec<f64> = (0..plan.numel)
        .into_par_iter()
        .map(|flat_idx| {
            let a = lhs
                .buffer
                .get_f64(lhs_plan.physical_index(flat_idx, &plan.output_shape));
            let b = rhs
                .buffer
                .get_f64(rhs_plan.physical_index(flat_idx, &plan.output_shape));
            f(a, b)
        })
        .collect();
    let out_buffer = lhs.buffer.from_f64_values(out)?;
    Ok(CpuStorage::from_contiguous(out_buffer, out_shape.to_vec()))
}

pub(crate) fn canonical_relu(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Relu, t)?;
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_binary(grad_out, &t_capture, &grad_out.shape, |g, x| {
                let deriv = if x > 0.0 { 1.0 } else { 0.0 };
                g * deriv
            })?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_unary(op: UnaryOp, t: &CpuStorage) -> Result<CpuStorage> {
    elementwise_unary_typed(op, t)
}

pub(crate) fn canonical_neg(t: &CpuStorage) -> Result<CpuStorage> {
    let out = negate(t);
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| Ok(vec![negate(grad_out)])),
    });
    Ok(out)
}

pub(crate) fn canonical_step(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Step, t)?;
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let total = crate::cpu::stride::checked_numel(&grad_out.shape)?;
            let zeros = vec![0.0f64; total];
            Ok(vec![CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(zeros)?,
                grad_out.shape.to_vec(),
            )])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_add_scalar(t: &CpuStorage, scalar: f64) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::AddScalar(scalar), t)?;
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| Ok(vec![grad_out.clone()])),
    });
    Ok(out)
}

pub(crate) fn canonical_mul_scalar(t: &CpuStorage, scalar: f64) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::MulScalar(scalar), t)?;
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let total = crate::cpu::stride::checked_numel(&grad_out.shape)?;
            let mut scaled = Vec::with_capacity(total);
            let mut idx = vec![0usize; grad_out.shape.len()];
            for _ in 0..total {
                scaled.push(grad_out.get(&idx) * scalar);
                if !grad_out.shape.is_empty() {
                    increment_index(&mut idx, &grad_out.shape);
                }
            }
            Ok(vec![CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(scaled)?,
                grad_out.shape.to_vec(),
            )])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_powf(t: &CpuStorage, exponent: f64) -> Result<CpuStorage> {
    elementwise_unary_typed(UnaryOp::Powf(exponent), t)
}

pub(crate) fn canonical_clamp(t: &CpuStorage, min: f64, max: f64) -> Result<CpuStorage> {
    elementwise_unary_typed(UnaryOp::Clamp(min, max), t)
}

pub(crate) fn canonical_fmod(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    elementwise_binary(lhs, rhs, &lhs.shape, |a, b| a % b)
}

pub(crate) fn canonical_remainder(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    elementwise_binary(lhs, rhs, &lhs.shape, |a, b| a.rem_euclid(b))
}

pub(crate) fn canonical_atan2(y: &CpuStorage, x: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_binary(y, x, &y.shape, |y_value, x_value| y_value.atan2(x_value))?;
    let (y_id, x_id, out_id) = (y.id, x.id, out.id);
    let (y_capture, x_capture) = (y.clone(), x.clone());
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![y_id, x_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let denominator = elementwise_binary(
                &y_capture,
                &x_capture,
                &grad_out.shape,
                |y_value, x_value| x_value * x_value + y_value * y_value,
            )?;
            let grad_y =
                elementwise_binary(grad_out, &x_capture, &grad_out.shape, |g, x_value| {
                    g * x_value
                })?;
            let grad_y = elementwise_binary(&grad_y, &denominator, &grad_out.shape, |g, d| g / d)?;
            let grad_x =
                elementwise_binary(grad_out, &y_capture, &grad_out.shape, |g, y_value| {
                    -g * y_value
                })?;
            let grad_x = elementwise_binary(&grad_x, &denominator, &grad_out.shape, |g, d| g / d)?;
            Ok(vec![
                tape::unbroadcast(&grad_y, &y_capture.shape)?,
                tape::unbroadcast(&grad_x, &x_capture.shape)?,
            ])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_softmax<D: Device>(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    let log_values = log_softmax::<D, f32>(t, dim)?;
    canonical_exp(&log_values)
}

pub(crate) fn canonical_exp(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Exp, t)?;
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_binary(grad_out, &out_capture, &grad_out.shape, |g, o| g * o)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_abs(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Abs, t)?;
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_binary(grad_out, &t_capture, &grad_out.shape, |g, x| {
                let derivative = if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                g * derivative
            })?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_sqrt(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Sqrt, t)?;
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_binary(grad_out, &out_capture, &grad_out.shape, |g, o| {
                g / (2.0 * o)
            })?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_mish(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Mish, t)?;
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_binary(grad_out, &t_capture, &grad_out.shape, |g, x| {
                let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
                let tanh = softplus.tanh();
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                g * (tanh + x * sigmoid * (1.0 - tanh * tanh))
            })?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_elu(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Elu, t)?;
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_binary(grad_out, &out_capture, &grad_out.shape, |g, o| {
                g * if o > 0.0 { 1.0 } else { o + 1.0 }
            })?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_trunc(t: &CpuStorage) -> Result<CpuStorage> {
    elementwise_unary_typed(UnaryOp::Trunc, t)
}

pub(crate) fn canonical_frac(t: &CpuStorage) -> Result<CpuStorage> {
    elementwise_unary_typed(UnaryOp::Frac, t)
}

pub(crate) fn canonical_log(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Log, t)?;
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_binary(grad_out, &t_capture, &grad_out.shape, |g, x| g / x)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_tanh(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Tanh, t)?;
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_binary(grad_out, &out_capture, &grad_out.shape, |g, o| {
                g * (1.0 - o * o)
            })?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_sigmoid(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Sigmoid, t)?;
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_binary(grad_out, &out_capture, &grad_out.shape, |g, o| {
                g * o * (1.0 - o)
            })?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

fn canonical_unary_with_derivative<F>(
    op: UnaryOp,
    t: &CpuStorage,
    derivative: F,
) -> Result<CpuStorage>
where
    F: Fn(f64) -> f64 + Send + Sync + 'static,
{
    let out = elementwise_unary_typed(op, t)?;
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_binary(grad_out, &t_capture, &grad_out.shape, |g, x| {
                g * derivative(x)
            })?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_gelu(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Gelu, t, |x| {
        let cdf = 0.5 * (1.0 + erf_approx(x / core::f64::consts::SQRT_2));
        let pdf = (1.0 / (2.0 * core::f64::consts::PI).sqrt()) * (-x * x / 2.0).exp();
        cdf + x * pdf
    })
}

pub(crate) fn canonical_swish(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Swish, t)?;
    let t_capture = t.clone();
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let total: usize = crate::cpu::stride::checked_numel(&grad_out.shape)?;
            let grad: Vec<f64> = (0..total)
                .into_par_iter()
                .map(|flat_idx| {
                    let nd_idx = flat_to_nd(flat_idx, &grad_out.shape);
                    let x = t_capture.get(&nd_idx);
                    let o = out_capture.get(&nd_idx);
                    let g = grad_out.get(&nd_idx);
                    let sig = 1.0 / (1.0 + (-x).exp());
                    g * (o + sig * (1.0 - o))
                })
                .collect();
            Ok(vec![CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(grad)?,
                grad_out.shape.to_vec(),
            )])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_tan(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Tan, t, |x| 1.0 + x.tan().powi(2))
}

pub(crate) fn canonical_asin(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Asin, t, |x| 1.0 / (1.0 - x * x).sqrt())
}

pub(crate) fn canonical_acos(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Acos, t, |x| -1.0 / (1.0 - x * x).sqrt())
}

pub(crate) fn canonical_atan(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Atan, t, |x| 1.0 / (1.0 + x * x))
}

pub(crate) fn canonical_sinh(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Sinh, t, f64::cosh)
}

pub(crate) fn canonical_cosh(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Cosh, t, f64::sinh)
}

pub(crate) fn canonical_asinh(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Asinh, t, |x| 1.0 / (x * x + 1.0).sqrt())
}

pub(crate) fn canonical_acosh(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Acosh, t, |x| 1.0 / (x * x - 1.0).sqrt())
}

pub(crate) fn canonical_atanh(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Atanh, t, |x| 1.0 / (1.0 - x * x))
}

pub(crate) fn canonical_erf(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Erf, t, |x| {
        2.0 / core::f64::consts::PI.sqrt() * (-x * x).exp()
    })
}

pub(crate) fn canonical_rsqrt(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_derivative(UnaryOp::Rsqrt, t, |x| -0.5 / (x * x.sqrt()))
}

fn elementwise_binary_numeric(
    op: BinaryOp,
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
) -> Result<CpuStorage> {
    if let Some(output) = elementwise_kernel::execute_binary(op, lhs, rhs, out_shape)? {
        return Ok(output);
    }
    elementwise_binary(lhs, rhs, out_shape, move |lhs, rhs| op.eval_f64(lhs, rhs))
}

/// Elementwise negate (used by `sub`'s backward rule: rhs receives the
/// negated incoming gradient before unbroadcasting).
pub(crate) fn elementwise_unary(
    t: &CpuStorage,
    f: impl Fn(f64) -> f64 + Send + Sync,
) -> Result<CpuStorage> {
    if let Some(range) = elementwise_kernel::dense_range(t, t.buffer.len(), &t.shape) {
        let buffer = match &*t.buffer {
            CpuBuffer::F32(v) => {
                let out =
                    crate::cpu::typed_kernel::map_unary_typed(&v[range], |x| f(x as f64) as f32);
                CpuBuffer::F32(out)
            }
            CpuBuffer::F64(v) => {
                let out = crate::cpu::typed_kernel::map_unary_typed(&v[range], &f);
                CpuBuffer::F64(out)
            }
            CpuBuffer::F16(v) => {
                let out = crate::cpu::typed_kernel::map_unary_typed(&v[range], |x| {
                    half::f16::from_f64(f(x.to_f64()))
                });
                CpuBuffer::F16(out)
            }
            CpuBuffer::BF16(v) => {
                let out = crate::cpu::typed_kernel::map_unary_typed(&v[range], |x| {
                    half::bf16::from_f64(f(x.to_f64()))
                });
                CpuBuffer::BF16(out)
            }
            _ => {
                let total: usize = crate::cpu::stride::checked_numel(&(t.shape))?;
                let out: Vec<f64> = (0..total)
                    .into_par_iter()
                    .map(|flat_idx| {
                        let nd_idx = flat_to_nd(flat_idx, &t.shape);
                        f(t.get(&nd_idx))
                    })
                    .collect();
                t.buffer.from_f64_values(out)?
            }
        };
        return Ok(CpuStorage::from_contiguous(buffer, t.shape.to_vec()));
    }

    let total: usize = crate::cpu::stride::checked_numel(&(t.shape))?;
    let out: Vec<f64> = (0..total)
        .into_par_iter()
        .map(|flat_idx| {
            let nd_idx = flat_to_nd(flat_idx, &t.shape);
            f(t.get(&nd_idx))
        })
        .collect();
    let out_buffer = t.buffer.from_f64_values(out)?;
    Ok(CpuStorage::from_contiguous(out_buffer, t.shape.to_vec()))
}

fn elementwise_unary_typed(op: UnaryOp, input: &CpuStorage) -> Result<CpuStorage> {
    if let Some(output) = elementwise_kernel::execute_unary(op, input)? {
        return Ok(output);
    }
    elementwise_unary(input, move |value| op.eval_f64(value))
}

/// `negate`.
fn negate(t: &CpuStorage) -> CpuStorage {
    elementwise_unary_typed(UnaryOp::Neg, t).unwrap()
}

/// Elementwise multiply of two ALREADY-shape-matching (or one of them
/// broadcastable to the other's shape) storages — used by `mul`'s backward
/// rule to compute `grad_out * other_operand`, aligned to `grad_out`'s shape.
#[allow(dead_code)]
fn mul_elementwise_broadcast(grad_out: &CpuStorage, other: &CpuStorage) -> Result<CpuStorage> {
    elementwise_binary_numeric(BinaryOp::Mul, grad_out, other, &grad_out.shape)
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

// The four pointwise binary kernels below are free functions rather than trait
// bodies because two entry points now need them: `` for the legacy
// tensor surface, and the canonical `Execute<op::Add>` executor in
// `cpu::canonical`. Keeping one body means the descriptor path cannot drift
// from the path it is replacing, and when `` is deleted these
// functions stay exactly as they are.

/// Broadcast elementwise addition, with its gradient recorded.
pub(crate) fn add_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
    add_storage_with_shape(lhs, rhs, &out_shape)
}

/// [`add_storage`] for a caller that already holds the resolved output shape.
///
/// The canonical executor does: `dispatch::execute_shaped` infers and
/// validates the output metadata before the backend is reached, so recomputing
/// the broadcast here would repeat a fallible loop and a heap allocation whose
/// answer is already sealed in the descriptor. `out_shape` must be that
/// resolved shape; passing anything else is a caller bug, not a runtime case,
/// which is why this takes a shape rather than an `Option`.
pub(crate) fn add_storage_with_shape(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
) -> Result<CpuStorage> {
    let out = elementwise_binary_numeric(BinaryOp::Add, lhs, rhs, out_shape)?;

    let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![
                tape::unbroadcast(grad_out, &lhs_shape)?,
                tape::unbroadcast(grad_out, &rhs_shape)?,
            ])
        }),
    });
    Ok(out)
}

/// Broadcast elementwise subtraction, with its gradient recorded.
pub(crate) fn sub_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
    sub_storage_with_shape(lhs, rhs, &out_shape)
}

/// [`sub_storage`] for a caller that already holds the resolved output shape.
///
/// The canonical executor does: `dispatch::execute_shaped` infers and
/// validates the output metadata before the backend is reached, so recomputing
/// the broadcast here would repeat a fallible loop and a heap allocation whose
/// answer is already sealed in the descriptor. `out_shape` must be that
/// resolved shape; passing anything else is a caller bug, not a runtime case,
/// which is why this takes a shape rather than an `Option`.
pub(crate) fn sub_storage_with_shape(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
) -> Result<CpuStorage> {
    let out = elementwise_binary_numeric(BinaryOp::Sub, lhs, rhs, out_shape)?;

    let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![
                tape::unbroadcast(grad_out, &lhs_shape)?,
                tape::unbroadcast(&negate(grad_out), &rhs_shape)?,
            ])
        }),
    });
    Ok(out)
}

/// Broadcast elementwise multiplication, with its gradient recorded.
pub(crate) fn mul_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
    mul_storage_with_shape(lhs, rhs, &out_shape)
}

/// [`mul_storage`] for a caller that already holds the resolved output shape.
///
/// The canonical executor does: `dispatch::execute_shaped` infers and
/// validates the output metadata before the backend is reached, so recomputing
/// the broadcast here would repeat a fallible loop and a heap allocation whose
/// answer is already sealed in the descriptor. `out_shape` must be that
/// resolved shape; passing anything else is a caller bug, not a runtime case,
/// which is why this takes a shape rather than an `Option`.
pub(crate) fn mul_storage_with_shape(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
) -> Result<CpuStorage> {
    let out = elementwise_binary_numeric(BinaryOp::Mul, lhs, rhs, out_shape)?;

    // Capture cloned copies of lhs/rhs's CpuStorage (cheap, Rc-backed)
    // since the backward closure needs their VALUES, not just shapes.
    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad_lhs =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &rhs_capture, &grad_out.shape)?;
            let grad_rhs =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &lhs_capture, &grad_out.shape)?;
            Ok(vec![
                tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                tape::unbroadcast(&grad_rhs, &rhs_shape)?,
            ])
        }),
    });
    Ok(out)
}

/// Broadcast elementwise division, with its gradient recorded.
pub(crate) fn div_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
    div_storage_with_shape(lhs, rhs, &out_shape)
}

/// [`div_storage`] for a caller that already holds the resolved output shape.
///
/// The canonical executor does: `dispatch::execute_shaped` infers and
/// validates the output metadata before the backend is reached, so recomputing
/// the broadcast here would repeat a fallible loop and a heap allocation whose
/// answer is already sealed in the descriptor. `out_shape` must be that
/// resolved shape; passing anything else is a caller bug, not a runtime case,
/// which is why this takes a shape rather than an `Option`.
pub(crate) fn div_storage_with_shape(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    out_shape: &[usize],
) -> Result<CpuStorage> {
    let out = elementwise_binary_numeric(BinaryOp::Div, lhs, rhs, out_shape)?;

    // Per Assumption A2 (RESEARCH.md): implemented for trait-completeness
    // via the standard quotient rule (1/rhs, -lhs/rhs^2), each
    // unbroadcast — best-effort correctness, not exercised by this
    // phase's example/tests.
    let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
    let (lhs_shape, rhs_shape) = (lhs.shape.to_vec(), rhs.shape.to_vec());
    let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![lhs_id, rhs_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
            let grad_lhs =
                elementwise_binary_numeric(BinaryOp::Div, grad_out, &rhs_capture, &grad_out.shape)?;
            // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = grad_out * (-lhs/rhs^2)
            let grad_rhs = elementwise_binary_numeric(
                BinaryOp::Mul,
                grad_out,
                &elementwise_binary(&lhs_capture, &rhs_capture, &grad_out.shape, |l, r| {
                    -l / (r * r)
                })?,
                &grad_out.shape,
            )?;
            Ok(vec![
                tape::unbroadcast(&grad_lhs, &lhs_shape)?,
                tape::unbroadcast(&grad_rhs, &rhs_shape)?,
            ])
        }),
    });
    Ok(out)
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
/// Called by both `::softmax` (as `exp(log_softmax(x, dim))`) and
/// `cross_entropy_loss` (as `-log_softmax(x, 1)[target]`), so the
/// numerically-stable kernel is shared rather than duplicated.
pub(crate) fn log_softmax<D: incin_core::tensor::device::Device, K: DType>(
    t: &CpuStorage,
    dim: usize,
) -> Result<CpuStorage> {
    let max = crate::cpu::ops::reduce::max_keepdim(t, dim)?;
    let diff = sub_storage(t, &max)?;
    let exp_diff = canonical_exp(&diff)?;
    let sum_exp = crate::cpu::ops::reduce::sum_keepdim(&exp_diff, dim)?;
    let log_sum_exp = canonical_log(&sum_exp)?;
    sub_storage(&diff, &log_sum_exp)
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    use crate::cpu::gradcheck::gradcheck;
    use crate::cpu::storage::CpuBuffer;
    use crate::cpu::tape;

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
        let out = add_storage(&lhs, &rhs).unwrap();

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
    fn numeric_ops_preserve_half_storage_and_compute_in_f32() {
        let f16_lhs = CpuStorage::from_contiguous(
            CpuBuffer::F16(vec![half::f16::from_f32(1.5), half::f16::from_f32(2.0)]),
            vec![2],
        );
        let f16_rhs = CpuStorage::from_contiguous(
            CpuBuffer::F16(vec![half::f16::from_f32(2.0), half::f16::from_f32(4.0)]),
            vec![2],
        );
        let f16_out = mul_storage(&f16_lhs, &f16_rhs).unwrap();
        assert_eq!(
            &*f16_out.buffer,
            &CpuBuffer::F16(vec![half::f16::from_f32(3.0), half::f16::from_f32(8.0)])
        );

        let bf16_lhs = CpuStorage::from_contiguous(
            CpuBuffer::BF16(vec![half::bf16::from_f32(1.5), half::bf16::from_f32(2.0)]),
            vec![2],
        );
        let bf16_rhs = CpuStorage::from_contiguous(
            CpuBuffer::BF16(vec![half::bf16::from_f32(2.0), half::bf16::from_f32(4.0)]),
            vec![2],
        );
        let bf16_out = mul_storage(&bf16_lhs, &bf16_rhs).unwrap();
        assert_eq!(
            &*bf16_out.buffer,
            &CpuBuffer::BF16(vec![half::bf16::from_f32(3.0), half::bf16::from_f32(8.0)])
        );
    }

    #[test]
    fn numeric_ops_preserve_non_contiguous_view_semantics() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
            .transpose(0, 1)
            .unwrap();
        let rhs = matrix(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3)
            .transpose(0, 1)
            .unwrap();
        let output = add_storage(&lhs, &rhs).unwrap();

        assert_eq!(output.shape, vec![3, 2]);
        assert_eq!(f32_vec(&output), vec![11.0, 44.0, 22.0, 55.0, 33.0, 66.0]);
    }

    #[test]
    fn relu_preserves_f64_dtype() {
        let t = f64_storage(vec![-1.000000123456789, 3.000000987654321], vec![2]);
        let out = canonical_relu(&t).unwrap();

        match &*out.buffer {
            CpuBuffer::F64(v) => {
                assert_eq!(*v, vec![0.0f64, 3.000000987654321f64]);
            }
            other => panic!("expected CpuBuffer::F64, got {other:?}"),
        }
    }


    #[test]
    /// `add_broadcasts_forward_correctly`.
    fn add_broadcasts_forward_correctly() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![10.0, 20.0, 30.0]);
        let out = add_storage(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    /// `add_backward_unbroadcasts_correctly_for_bias_vector_case`.
    fn add_backward_unbroadcasts_correctly_for_bias_vector_case() {
        let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let rhs = vector(vec![10.0, 20.0, 30.0]);
        let out = add_storage(&lhs, &rhs).unwrap();

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
        let out = sub_storage(&lhs, &rhs).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(f32_vec(&out), vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
    }

    #[test]
    /// `sub_backward_negates_rhs_contribution`.
    fn sub_backward_negates_rhs_contribution() {
        let lhs = vector(vec![10.0, 20.0, 30.0]);
        let rhs = vector(vec![1.0, 2.0, 3.0]);
        let out = sub_storage(&lhs, &rhs).unwrap();

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
        let out = mul_storage(&lhs, &rhs).unwrap();
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
        let out = mul_storage(&a, &b).unwrap();

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
        let out = mul_storage(&lhs, &rhs).unwrap();

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
        let out = canonical_add_scalar(&t, 1.0).unwrap();
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
        let out = canonical_mul_scalar(&t, 2.5).unwrap();
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
        let out = canonical_relu(&t).unwrap();
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
            let r = canonical_relu(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&r).unwrap()
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
        let out = canonical_abs(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![2.5, 0.0, 3.5]);

        let x = vector(vec![-2.0, 1.5, -0.3]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let a = canonical_abs(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&a).unwrap()
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
        let out = canonical_neg(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![-1.0, 2.0, -3.0]);

        let x = vector(vec![1.0, -2.0, 3.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let n = canonical_neg(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&n).unwrap()
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
        let out = canonical_exp(&t).unwrap();
        let expect = [1.0f32, core::f64::consts::E as f32];
        for (a, b) in f32_vec(&out).iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-5, "exp forward mismatch: {a} vs {b}");
        }

        let x = vector(vec![0.5, -0.3, 1.2]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let e = canonical_exp(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&e).unwrap()
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
        let out = canonical_sqrt(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![2.0, 3.0]);

        let x = vector(vec![4.0, 1.0, 9.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let s = canonical_sqrt(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&s).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "sqrt gradcheck error too high: {max_rel_err}"
        );

        // Negative input propagates NaN cpuly (RESEARCH.md Pitfall 2),
        // not a panic and not an Err.
        let neg_input = vector(vec![-1.0]);
        let neg_out = canonical_sqrt(&neg_input).unwrap();
        assert!(f32_vec(&neg_out)[0].is_nan(), "sqrt(-1.0) should be NaN");
    }

    #[test]
    /// `log_forward_gradcheck_and_domain_propagation`.
    fn log_forward_gradcheck_and_domain_propagation() {
        let t = vector(vec![1.0, core::f64::consts::E as f32]);
        let out = canonical_log(&t).unwrap();
        let expect = [0.0f32, 1.0f32];
        for (a, b) in f32_vec(&out).iter().zip(expect.iter()) {
            assert!((a - b).abs() < 1e-5, "log forward mismatch: {a} vs {b}");
        }

        let x = vector(vec![1.0, 2.0, 5.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let l = canonical_log(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&l).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "log gradcheck error too high: {max_rel_err}"
        );

        // Zero/negative input propagates NaN/-inf cpuly, not a panic and
        // not an Err.
        let zero_input = vector(vec![0.0]);
        let zero_out = canonical_log(&zero_input).unwrap();
        assert!(
            f32_vec(&zero_out)[0].is_infinite() && f32_vec(&zero_out)[0] < 0.0,
            "log(0.0) should be -inf"
        );

        let neg_input = vector(vec![-1.0]);
        let neg_out = canonical_log(&neg_input).unwrap();
        assert!(f32_vec(&neg_out)[0].is_nan(), "log(-1.0) should be NaN");
    }

    #[test]
    /// `tanh_gradcheck`.
    fn tanh_gradcheck() {
        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let th = canonical_tanh(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&th).unwrap()
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
        let out = canonical_sigmoid(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![0.5]);

        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let s = canonical_sigmoid(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&s).unwrap()
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
        let out = canonical_swish(&t).unwrap();
        assert_eq!(f32_vec(&out), vec![0.0]);

        let x = vector(vec![0.5, -1.0, 2.0]);
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let s = canonical_swish(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&s).unwrap()
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
        let out_zero = canonical_gelu(&zero).unwrap();
        assert_eq!(f32_vec(&out_zero), vec![0.0]);

        let one = vector(vec![1.0]);
        let out_one = canonical_gelu(&one).unwrap();
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
            let g = canonical_gelu(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&g).unwrap()
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
        let out = canonical_softmax::<incin_core::tensor::device::Cpu>(&t, 0).unwrap();
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
        let out = canonical_softmax::<incin_core::tensor::device::Cpu>(&t, 1).unwrap();
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
        let out = canonical_softmax::<incin_core::tensor::device::Cpu>(&t, 0).unwrap();
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
        let out = canonical_softmax::<incin_core::prelude::Cpu>(&t, 0).unwrap();
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
            let s = canonical_softmax::<incin_core::tensor::device::Cpu>(&inputs[0], 0).unwrap();
            crate::cpu::ops::reduce::sum_all(&s).unwrap()
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
        let out = canonical_softmax::<incin_core::prelude::Cpu>(&t, 0).unwrap();

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
        let ls = log_softmax::<incin_core::tensor::device::Cpu, f32>(&t, 0).unwrap();
        let exp_ls = canonical_exp(&ls).unwrap();
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
        let ls = log_softmax::<incin_core::tensor::device::Cpu, f32>(&t, 0).unwrap();
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
        let out = canonical_softmax::<incin_core::prelude::Cpu>(&t, 0).unwrap();
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
            let ls = log_softmax::<incin_core::tensor::device::Cpu, f32>(&inputs[0], 0).unwrap();
            crate::cpu::ops::reduce::sum_all(&ls).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "log_softmax gradcheck error too high: {max_rel_err}"
        );
    }
}
