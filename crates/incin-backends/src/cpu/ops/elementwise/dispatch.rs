use super::*;

/// Build a contiguous `CpuStorage` by applying `f(lhs_val, rhs_val)` over
/// every logical index in `out_shape`, reading each operand through its own
/// broadcast-resolved index (no pre-materialized broadcast copy).
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
        return Ok(CpuStorage::from_contiguous(buffer, out_shape));
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
    Ok(CpuStorage::from_contiguous(out_buffer, out_shape))
}

pub(crate) fn canonical_unary(op: UnaryOp, t: &CpuStorage) -> Result<CpuStorage> {
    elementwise_unary_typed(op, t)
}

pub(super) fn elementwise_binary_numeric(
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
        return Ok(CpuStorage::from_contiguous(buffer, &t.shape));
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
    Ok(CpuStorage::from_contiguous(out_buffer, &t.shape))
}

pub(super) fn elementwise_unary_typed(op: UnaryOp, input: &CpuStorage) -> Result<CpuStorage> {
    if let Some(output) = elementwise_kernel::execute_unary(op, input)? {
        return Ok(output);
    }
    elementwise_unary(input, move |value| op.eval_f64(value))
}

/// `negate`.
pub(super) fn negate(t: &CpuStorage) -> CpuStorage {
    elementwise_unary_typed(UnaryOp::Neg, t).unwrap()
}

pub(super) fn canonical_unary_with_deriv_op(
    op: UnaryOp,
    deriv_op: UnaryOp,
    t: &CpuStorage,
) -> Result<CpuStorage> {
    let out = elementwise_unary_typed(op, t)?;
    let t_capture = t.clone();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let deriv = elementwise_unary_typed(deriv_op, &t_capture)?;
            let grad =
                elementwise_binary_numeric(BinaryOp::Mul, grad_out, &deriv, &grad_out.shape)?;
            Ok(vec![grad])
        }),
    });
    Ok(out)
}
