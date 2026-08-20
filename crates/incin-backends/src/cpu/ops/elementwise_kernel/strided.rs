use super::*;

pub(super) fn map_unary<T: TypedKernel, F>(input: &[T], op: &F) -> Vec<T>
where
    F: Fn(T) -> T + Send + Sync,
{
    map_unary_typed(input, op)
}

pub(super) fn map_binary_strided<T, F>(lhs: &[T], rhs: &[T], plan: &IterationPlan, op: &F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    if plan.numel < PARALLEL_GRAIN {
        return map_binary_strided_serial(lhs, rhs, plan, op);
    }
    let evaluate = |flat_index| {
        let lhs_index = plan.operands[0].physical_index(flat_index, &plan.output_shape);
        let rhs_index = plan.operands[1].physical_index(flat_index, &plan.output_shape);
        op(lhs[lhs_index], rhs[rhs_index])
    };
    (0..plan.numel).into_par_iter().map(evaluate).collect()
}

fn map_binary_strided_serial<T, F>(lhs: &[T], rhs: &[T], plan: &IterationPlan, op: &F) -> Vec<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    let mut output = Vec::with_capacity(plan.numel);
    if plan.numel == 0 {
        return output;
    }
    if plan.output_shape.is_empty() {
        output.push(op(
            lhs[plan.operands[0].offset],
            rhs[plan.operands[1].offset],
        ));
        return output;
    }

    let outer_rank = plan.output_shape.len() - 1;
    let inner_len = plan.output_shape[outer_rank];
    let outer_count = plan.numel / inner_len;
    let mut coordinates = vec![0usize; outer_rank];
    let mut lhs_index = plan.operands[0].offset;
    let mut rhs_index = plan.operands[1].offset;
    let lhs_inner_stride = plan.operands[0].strides[outer_rank];
    let rhs_inner_stride = plan.operands[1].strides[outer_rank];

    for outer_index in 0..outer_count {
        for inner_index in 0..inner_len {
            output.push(op(
                lhs[lhs_index + inner_index * lhs_inner_stride],
                rhs[rhs_index + inner_index * rhs_inner_stride],
            ));
        }
        if outer_index + 1 == outer_count {
            break;
        }
        advance_binary(
            &mut coordinates,
            &plan.output_shape[..outer_rank],
            &mut lhs_index,
            &plan.operands[0].strides[..outer_rank],
            &mut rhs_index,
            &plan.operands[1].strides[..outer_rank],
        );
    }
    output
}

pub(super) fn map_unary_strided<T, F>(input: &[T], plan: &UnaryIterationPlan, op: &F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T) -> T + Send + Sync,
{
    if plan.numel < PARALLEL_GRAIN {
        return map_unary_strided_serial(input, plan, op);
    }
    let evaluate = |flat_index| {
        let input_index = plan.operand.physical_index(flat_index, &plan.output_shape);
        op(input[input_index])
    };
    (0..plan.numel).into_par_iter().map(evaluate).collect()
}

fn map_unary_strided_serial<T, F>(input: &[T], plan: &UnaryIterationPlan, op: &F) -> Vec<T>
where
    T: Copy,
    F: Fn(T) -> T,
{
    let mut output = Vec::with_capacity(plan.numel);
    if plan.numel == 0 {
        return output;
    }
    if plan.output_shape.is_empty() {
        output.push(op(input[plan.operand.offset]));
        return output;
    }

    let outer_rank = plan.output_shape.len() - 1;
    let inner_len = plan.output_shape[outer_rank];
    let outer_count = plan.numel / inner_len;
    let mut coordinates = vec![0usize; outer_rank];
    let mut input_index = plan.operand.offset;
    let inner_stride = plan.operand.strides[outer_rank];

    for outer_index in 0..outer_count {
        for inner_index in 0..inner_len {
            output.push(op(input[input_index + inner_index * inner_stride]));
        }
        if outer_index + 1 == outer_count {
            break;
        }
        advance_unary(
            &mut coordinates,
            &plan.output_shape[..outer_rank],
            &mut input_index,
            &plan.operand.strides[..outer_rank],
        );
    }
    output
}

fn advance_binary(
    coordinates: &mut [usize],
    shape: &[usize],
    lhs_index: &mut usize,
    lhs_strides: &[usize],
    rhs_index: &mut usize,
    rhs_strides: &[usize],
) {
    for axis in (0..shape.len()).rev() {
        if coordinates[axis] + 1 < shape[axis] {
            coordinates[axis] += 1;
            *lhs_index += lhs_strides[axis];
            *rhs_index += rhs_strides[axis];
            return;
        }
        coordinates[axis] = 0;
        *lhs_index -= lhs_strides[axis] * (shape[axis] - 1);
        *rhs_index -= rhs_strides[axis] * (shape[axis] - 1);
    }
}

fn advance_unary(
    coordinates: &mut [usize],
    shape: &[usize],
    input_index: &mut usize,
    strides: &[usize],
) {
    for axis in (0..shape.len()).rev() {
        if coordinates[axis] + 1 < shape[axis] {
            coordinates[axis] += 1;
            *input_index += strides[axis];
            return;
        }
        coordinates[axis] = 0;
        *input_index -= strides[axis] * (shape[axis] - 1);
    }
}

pub(super) fn map_scalar_right<T, F>(lhs: &[T], rhs: T, op: &F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    if lhs.len() < PARALLEL_GRAIN {
        lhs.iter().map(|&lhs| op(lhs, rhs)).collect()
    } else {
        lhs.par_iter().map(|&lhs| op(lhs, rhs)).collect()
    }
}

pub(super) fn map_scalar_left<T, F>(lhs: T, rhs: &[T], op: &F) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    if rhs.len() < PARALLEL_GRAIN {
        rhs.iter().map(|&rhs| op(lhs, rhs)).collect()
    } else {
        rhs.par_iter().map(|&rhs| op(lhs, rhs)).collect()
    }
}
