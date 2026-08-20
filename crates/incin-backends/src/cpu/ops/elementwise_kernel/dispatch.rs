use super::*;

pub(crate) fn execute_binary(
    op: BinaryOp,
    lhs: &CpuStorage,
    rhs: &CpuStorage,
    output_shape: &[usize],
) -> Result<Option<CpuStorage>> {
    if lhs.buffer.dtype_id() != rhs.buffer.dtype_id() {
        return Err(Error::DTypeStorageMismatch {
            expected: lhs.buffer.descriptor(),
            got: rhs.buffer.descriptor(),
        });
    }
    if crate::cpu::validate_cpu_dtype(lhs.buffer.descriptor(), "pointwise").is_err() {
        return Ok(None);
    }

    let output: Option<Result<CpuBuffer>> = match (&*lhs.buffer, &*rhs.buffer) {
        (CpuBuffer::F32(lhs_values), CpuBuffer::F32(rhs_values)) => Some(
            execute_layout_f32(op, lhs, lhs_values, rhs, rhs_values, output_shape)
                .map(CpuBuffer::F32),
        ),
        (CpuBuffer::F64(lhs_values), CpuBuffer::F64(rhs_values)) => Some(
            execute_layout_f64(op, lhs, lhs_values, rhs, rhs_values, output_shape)
                .map(CpuBuffer::F64),
        ),
        (CpuBuffer::F16(lhs_values), CpuBuffer::F16(rhs_values)) => Some(
            execute_layout(lhs, lhs_values, rhs, rhs_values, output_shape, |a, b| {
                f16::from_f32(op.eval_f32(a.to_f32(), b.to_f32()))
            })
            .map(CpuBuffer::F16),
        ),
        (CpuBuffer::BF16(lhs_values), CpuBuffer::BF16(rhs_values)) => Some(
            execute_layout(lhs, lhs_values, rhs, rhs_values, output_shape, |a, b| {
                bf16::from_f32(op.eval_f32(a.to_f32(), b.to_f32()))
            })
            .map(CpuBuffer::BF16),
        ),
        _ => None,
    };

    Ok(output
        .transpose()?
        .map(|buffer| CpuStorage::from_contiguous(buffer, output_shape)))
}

fn execute_layout_f32(
    op: BinaryOp,
    lhs: &CpuStorage,
    lhs_values: &[f32],
    rhs: &CpuStorage,
    rhs_values: &[f32],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    if let (Some(lhs_range), Some(rhs_range)) = (
        dense_range(lhs, lhs_values.len(), output_shape),
        dense_range(rhs, rhs_values.len(), output_shape),
    ) {
        return Ok(map_binary_f32(
            op,
            &lhs_values[lhs_range],
            &rhs_values[rhs_range],
        ));
    }
    if let Some(lhs_range) = dense_range(lhs, lhs_values.len(), output_shape)
        && let Some(rhs_scalar) = scalar_value(rhs, rhs_values)
    {
        return Ok(map_scalar_f32(
            op,
            &lhs_values[lhs_range],
            rhs_scalar,
            false,
        ));
    }
    if let Some(rhs_range) = dense_range(rhs, rhs_values.len(), output_shape)
        && let Some(lhs_scalar) = scalar_value(lhs, lhs_values)
    {
        return Ok(map_scalar_f32(op, &rhs_values[rhs_range], lhs_scalar, true));
    }
    execute_strided_f32(op, lhs, lhs_values, rhs, rhs_values, output_shape)
}

fn execute_layout_f64(
    op: BinaryOp,
    lhs: &CpuStorage,
    lhs_values: &[f64],
    rhs: &CpuStorage,
    rhs_values: &[f64],
    output_shape: &[usize],
) -> Result<Vec<f64>> {
    if let (Some(lhs_range), Some(rhs_range)) = (
        dense_range(lhs, lhs_values.len(), output_shape),
        dense_range(rhs, rhs_values.len(), output_shape),
    ) {
        return Ok(map_binary_f64(
            op,
            &lhs_values[lhs_range],
            &rhs_values[rhs_range],
        ));
    }
    if let Some(lhs_range) = dense_range(lhs, lhs_values.len(), output_shape)
        && let Some(rhs_scalar) = scalar_value(rhs, rhs_values)
    {
        return Ok(map_scalar_f64(
            op,
            &lhs_values[lhs_range],
            rhs_scalar,
            false,
        ));
    }
    if let Some(rhs_range) = dense_range(rhs, rhs_values.len(), output_shape)
        && let Some(lhs_scalar) = scalar_value(lhs, lhs_values)
    {
        return Ok(map_scalar_f64(op, &rhs_values[rhs_range], lhs_scalar, true));
    }
    execute_strided_f64(op, lhs, lhs_values, rhs, rhs_values, output_shape)
}

fn execute_strided_f32(
    op: BinaryOp,
    lhs: &CpuStorage,
    lhs_values: &[f32],
    rhs: &CpuStorage,
    rhs_values: &[f32],
    output_shape: &[usize],
) -> Result<Vec<f32>> {
    let plan = binary_iteration_plan(lhs, lhs_values.len(), rhs, rhs_values.len(), output_shape)?;
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    if let Some(output) = map_iteration_avx2_f32(op, lhs_values, rhs_values, &plan) {
        return Ok(output);
    }
    Ok(map_binary_strided(
        lhs_values,
        rhs_values,
        &plan,
        &|lhs, rhs| op.eval_f32(lhs, rhs),
    ))
}

fn execute_strided_f64(
    op: BinaryOp,
    lhs: &CpuStorage,
    lhs_values: &[f64],
    rhs: &CpuStorage,
    rhs_values: &[f64],
    output_shape: &[usize],
) -> Result<Vec<f64>> {
    let plan = binary_iteration_plan(lhs, lhs_values.len(), rhs, rhs_values.len(), output_shape)?;
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    if let Some(output) = map_iteration_avx2_f64(op, lhs_values, rhs_values, &plan) {
        return Ok(output);
    }
    Ok(map_binary_strided(
        lhs_values,
        rhs_values,
        &plan,
        &|lhs, rhs| op.eval_f64(lhs, rhs),
    ))
}

pub(crate) fn execute_unary(op: UnaryOp, input: &CpuStorage) -> Result<Option<CpuStorage>> {
    if crate::cpu::validate_cpu_dtype(input.buffer.descriptor(), "pointwise").is_err() {
        return Ok(None);
    }
    let output: Option<Result<CpuBuffer>> = match &*input.buffer {
        CpuBuffer::F32(values) => Some(
            execute_unary_layout(input, values, |value| op.eval_f32(value)).map(CpuBuffer::F32),
        ),
        CpuBuffer::F64(values) => Some(
            execute_unary_layout(input, values, |value| op.eval_f64(value)).map(CpuBuffer::F64),
        ),
        CpuBuffer::F16(values) => Some(
            execute_unary_layout(input, values, |value| {
                f16::from_f32(op.eval_f32(value.to_f32()))
            })
            .map(CpuBuffer::F16),
        ),
        CpuBuffer::BF16(values) => Some(
            execute_unary_layout(input, values, |value| {
                bf16::from_f32(op.eval_f32(value.to_f32()))
            })
            .map(CpuBuffer::BF16),
        ),
        _ => None,
    };

    Ok(output
        .transpose()?
        .map(|buffer| CpuStorage::from_contiguous(buffer, &input.shape)))
}

fn execute_unary_layout<T>(
    input: &CpuStorage,
    values: &[T],
    op: impl Fn(T) -> T + Send + Sync,
) -> Result<Vec<T>>
where
    T: TypedKernel,
{
    if let Some(range) = dense_range(input, values.len(), &input.shape) {
        return Ok(map_unary(&values[range], &op));
    }

    let plan = UnaryIterationPlan::new(OperandLayout {
        shape: &input.shape,
        strides: &input.strides,
        offset: input.offset_elements,
    })?;
    validate_bounds(&plan.operand, &plan.output_shape, values.len())?;
    Ok(map_unary_strided(values, &plan, &op))
}

fn execute_layout<T>(
    lhs: &CpuStorage,
    lhs_values: &[T],
    rhs: &CpuStorage,
    rhs_values: &[T],
    output_shape: &[usize],
    op: impl Fn(T, T) -> T + Send + Sync,
) -> Result<Vec<T>>
where
    T: TypedKernel,
{
    if let (Some(lhs_range), Some(rhs_range)) = (
        dense_range(lhs, lhs_values.len(), output_shape),
        dense_range(rhs, rhs_values.len(), output_shape),
    ) {
        return Ok(map_binary(
            &lhs_values[lhs_range],
            &rhs_values[rhs_range],
            &op,
        ));
    }

    if let Some(lhs_range) = dense_range(lhs, lhs_values.len(), output_shape)
        && let Some(rhs_scalar) = scalar_value(rhs, rhs_values)
    {
        return Ok(map_scalar_right(&lhs_values[lhs_range], rhs_scalar, &op));
    }

    if let Some(rhs_range) = dense_range(rhs, rhs_values.len(), output_shape)
        && let Some(lhs_scalar) = scalar_value(lhs, lhs_values)
    {
        return Ok(map_scalar_left(lhs_scalar, &rhs_values[rhs_range], &op));
    }

    let plan = binary_iteration_plan(lhs, lhs_values.len(), rhs, rhs_values.len(), output_shape)?;
    Ok(map_binary_strided(lhs_values, rhs_values, &plan, &op))
}

pub(super) fn binary_iteration_plan(
    lhs: &CpuStorage,
    lhs_len: usize,
    rhs: &CpuStorage,
    rhs_len: usize,
    output_shape: &[usize],
) -> Result<IterationPlan> {
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
        output_shape,
    )?;
    validate_bounds(&plan.operands[0], &plan.output_shape, lhs_len)?;
    validate_bounds(&plan.operands[1], &plan.output_shape, rhs_len)?;
    Ok(plan)
}
