use super::*;

pub(crate) fn canonical_relu(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Relu, t)?;
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let step_mask = elementwise_unary_typed(UnaryOp::Step, &t_capture)?;
            let grad =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &step_mask, &grad_out.shape)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_neg(t: &CpuStorage) -> Result<CpuStorage> {
    let out = negate(t);
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| Ok(vec![negate(grad_out)])),
    });
    Ok(out)
}

pub(crate) fn canonical_step(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Step, t)?;
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![CpuStorage::zeros_like(grad_out)?])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_add_scalar(t: &CpuStorage, scalar: f64) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::AddScalar(scalar), t)?;
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| Ok(vec![grad_out.clone()])),
    });
    Ok(out)
}

pub(crate) fn canonical_mul_scalar(t: &CpuStorage, scalar: f64) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::MulScalar(scalar), t)?;
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad = elementwise_unary_typed(UnaryOp::MulScalar(scalar), grad_out)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_powf(t: &CpuStorage, exponent: f64) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Powf(exponent), t)?;

    // d x^p / dx = p * x^(p-1), evaluated at the captured input.
    let t_cap = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(move || TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let derivative = elementwise_unary_typed(UnaryOp::Powf(exponent - 1.0), &t_cap)?;
            let scaled = elementwise_unary_typed(UnaryOp::MulScalar(exponent), &derivative)?;
            Ok(vec![elementwise_binary_numeric(
                BinaryOp::Mul,
                grad_out,
                &scaled,
                &grad_out.shape,
            )?])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_clamp(t: &CpuStorage, min: f64, max: f64) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Clamp(min, max), t)?;

    // The cotangent passes through the interior and stops at both clamped
    // regions; on a boundary the subgradient convention is zero, matching
    // torch.clamp.
    let t_cap = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(move || TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let total = crate::cpu::stride::checked_numel(&grad_out.shape)?;
            let mut vals = Vec::with_capacity(total);
            let mut idx = vec![0usize; grad_out.shape.len()];
            for _ in 0..total {
                let value = t_cap.get(&idx);
                vals.push(if value < min || value > max {
                    0.0
                } else {
                    grad_out.get(&idx)
                });
                if !grad_out.shape.is_empty() {
                    crate::cpu::storage::increment_index(&mut idx, &grad_out.shape);
                }
            }
            Ok(vec![CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(vals)?,
                &grad_out.shape,
            )])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_exp(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Exp, t)?;
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &out_capture, &grad_out.shape)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_abs(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Abs, t)?;
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let sign_mask = elementwise_unary_typed(UnaryOp::Sign, &t_capture)?;
            let grad =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &sign_mask, &grad_out.shape)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_sqrt(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Sqrt, t)?;
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let div =
                elementwise_binary_numeric(BinaryOp::Div, grad_out, &out_capture, &grad_out.shape)?;
            let grad = elementwise_unary_typed(UnaryOp::MulScalar(0.5), &div)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_mish(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Mish, UnaryOp::MishBackward, t)
}

pub(crate) fn canonical_elu(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Elu, t)?;
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let deriv = elementwise_unary_typed(UnaryOp::EluBackward, &out_capture)?;
            let grad =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &deriv, &grad_out.shape)?;
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
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let grad =
                elementwise_binary_numeric(BinaryOp::Div, grad_out, &t_capture, &grad_out.shape)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_tanh(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Tanh, t)?;
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let deriv = elementwise_unary_typed(UnaryOp::TanhBackward, &out_capture)?;
            let grad =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &deriv, &grad_out.shape)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_sigmoid(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Sigmoid, t)?;
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let deriv = elementwise_unary_typed(UnaryOp::SigmoidBackward, &out_capture)?;
            let grad =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &deriv, &grad_out.shape)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_gelu(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Gelu, UnaryOp::GeluBackward, t)
}

// SiLU/Swish backward combines three operands per element (input t, forward out,
// and incoming grad_out). Because there is no 3-operand typed kernel layout to
// reuse, this gradient remains on the f64 index-walk path until a generic 3-way
// typed kernel is introduced in a future refactor.
pub(crate) fn canonical_swish(t: &CpuStorage) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(UnaryOp::Swish, t)?;
    let t_capture = t.clone();
    let out_capture = out.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
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
                &grad_out.shape,
            )])
        }),
    });
    Ok(out)
}

pub(crate) fn canonical_tan(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Tan, UnaryOp::TanBackward, t)
}

pub(crate) fn canonical_asin(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Asin, UnaryOp::AsinBackward, t)
}

pub(crate) fn canonical_acos(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Acos, UnaryOp::AcosBackward, t)
}

pub(crate) fn canonical_atan(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Atan, UnaryOp::AtanBackward, t)
}

pub(crate) fn canonical_sinh(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Sinh, UnaryOp::Cosh, t)
}

pub(crate) fn canonical_cosh(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Cosh, UnaryOp::Sinh, t)
}

pub(crate) fn canonical_asinh(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Asinh, UnaryOp::AsinhBackward, t)
}

pub(crate) fn canonical_acosh(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Acosh, UnaryOp::AcoshBackward, t)
}

pub(crate) fn canonical_atanh(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Atanh, UnaryOp::AtanhBackward, t)
}

pub(crate) fn canonical_erf(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Erf, UnaryOp::ErfBackward, t)
}

pub(crate) fn canonical_rsqrt(t: &CpuStorage) -> Result<CpuStorage> {
    canonical_unary_with_deriv_op(UnaryOp::Rsqrt, UnaryOp::RsqrtBackward, t)
}
