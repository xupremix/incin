use super::*;

pub(crate) fn elementwise_cmp(
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    f: impl Fn(f64, f64) -> bool,
) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
    let total: usize = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    if lhs.shape == rhs.shape {
        let mut idx = vec![0usize; lhs.shape.len()];
        for _ in 0..total {
            let v = if f(lhs.get(&idx), rhs.get(&idx)) {
                1u8
            } else {
                0u8
            };
            out.push(v);
            if !lhs.shape.is_empty() {
                crate::cpu::storage::increment_index(&mut idx, &lhs.shape);
            }
        }
    } else {
        let plan = crate::iteration::IterationPlan::binary(
            crate::iteration::OperandLayout {
                shape: &lhs.shape,
                strides: &lhs.strides,
                offset: lhs.offset_elements,
            },
            crate::iteration::OperandLayout {
                shape: &rhs.shape,
                strides: &rhs.strides,
                offset: rhs.offset_elements,
            },
            &out_shape,
        )?;
        let l_plan = &plan.operands[0];
        let r_plan = &plan.operands[1];
        for flat_idx in 0..plan.numel {
            let a = lhs
                .buffer
                .get_f64(l_plan.physical_index(flat_idx, &plan.output_shape));
            let b = rhs
                .buffer
                .get_f64(r_plan.physical_index(flat_idx, &plan.output_shape));
            out.push(if f(a, b) { 1u8 } else { 0u8 });
        }
    }
    Ok(CpuStorage::from_contiguous(CpuBuffer::Bool(out), out_shape))
}

pub(crate) fn sub_scalar_storage(t: &CpuStorage, val: f64) -> Result<CpuStorage> {
    elementwise_unary(t, |value| value - val)
}

pub(crate) fn div_scalar_storage(t: &CpuStorage, val: f64) -> Result<CpuStorage> {
    elementwise_unary(t, |value| value / val)
}
