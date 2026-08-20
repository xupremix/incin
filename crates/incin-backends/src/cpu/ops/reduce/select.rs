use super::*;

/// Index of the maximum element. `Some(d)`: per-axis, axis removed from
/// the output shape (mirrors `max_dim`'s squeeze shape). `None`: fully
/// flattened, returns a scalar (shape `[]`) holding the single winning
/// flat index. Forward-only - `incin-core`'s `Tensor::argmax`
/// structurally forces `G = NoGrad` on the output regardless of the
/// input's own `G`, so this deliberately never calls `tape::push`
/// (T-02-09 mitigation; the one exception to this file's
/// every-other-method unconditional-push convention).
pub(crate) fn argmax<KInt: DType>(t: &CpuStorage, dim: Option<usize>) -> Result<CpuStorage> {
    match dim {
        Some(d) => {
            if d >= t.shape.len() {
                return Err(Error::ShapeMismatch {
                    op: "argmax",
                    expected: t.shape.to_vec(),
                    got: vec![d],
                    msg: format!("argmax: axis {d} out of range for shape {:?}", t.shape),
                });
            }
            let (_, winning_flat_src_idx) = max_axis_with_indices(t, d)?;
            let mut out_shape = t.shape.to_vec();
            out_shape[d] = 1;
            // Convert each winning FLAT source index into its coordinate
            // along `d` (the axis-position the winner occupied), not the
            // flat index itself.
            let idx_vals: Vec<i64> = winning_flat_src_idx
                .iter()
                .map(|&flat_src| {
                    let multi = unflatten_index(flat_src, &t.shape);
                    multi[d] as i64
                })
                .collect();
            let keepdim_out =
                CpuStorage::from_contiguous(index_buffer::<KInt>("argmax", &idx_vals)?, &out_shape);
            let mut squeeze_shape = keepdim_out.shape.to_vec();
            squeeze_shape.remove(d);
            Ok(keepdim_out
                .reshape(&squeeze_shape)
                .expect("argmax: squeeze reshape of size-1 keepdim result cannot fail"))
        }
        None => {
            let total: usize = crate::cpu::stride::validated_numel(&(t.shape));
            let mut idx = vec![0usize; t.shape.len()];
            let mut best_val = f64::NEG_INFINITY;
            let mut best_flat_idx = 0i64;
            for flat in 0..total {
                let v = t.get(&idx);
                if v > best_val {
                    best_val = v;
                    best_flat_idx = flat as i64;
                }
                if !t.shape.is_empty() {
                    increment_index(&mut idx, &t.shape);
                }
            }
            Ok(CpuStorage::from_contiguous(
                index_buffer::<KInt>("argmax", &[best_flat_idx])?,
                vec![],
            ))
        }
    }
}

/// Index of the minimum element. Mirror of `argmax` using
/// `min_axis_with_indices`. Forward-only, no `tape::push` (T-02-09).
pub(crate) fn argmin<KInt: DType>(t: &CpuStorage, dim: Option<usize>) -> Result<CpuStorage> {
    match dim {
        Some(d) => {
            if d >= t.shape.len() {
                return Err(Error::ShapeMismatch {
                    op: "argmin",
                    expected: t.shape.to_vec(),
                    got: vec![d],
                    msg: format!("argmin: axis {d} out of range for shape {:?}", t.shape),
                });
            }
            let (_, winning_flat_src_idx) = min_axis_with_indices(t, d)?;
            let mut out_shape = t.shape.to_vec();
            out_shape[d] = 1;
            let idx_vals: Vec<i64> = winning_flat_src_idx
                .iter()
                .map(|&flat_src| {
                    let multi = unflatten_index(flat_src, &t.shape);
                    multi[d] as i64
                })
                .collect();
            let keepdim_out =
                CpuStorage::from_contiguous(index_buffer::<KInt>("argmin", &idx_vals)?, &out_shape);
            let mut squeeze_shape = keepdim_out.shape.to_vec();
            squeeze_shape.remove(d);
            Ok(keepdim_out
                .reshape(&squeeze_shape)
                .expect("argmin: squeeze reshape of size-1 keepdim result cannot fail"))
        }
        None => {
            let total: usize = crate::cpu::stride::validated_numel(&(t.shape));
            let mut idx = vec![0usize; t.shape.len()];
            let mut best_val = f64::INFINITY;
            let mut best_flat_idx = 0i64;
            for flat in 0..total {
                let v = t.get(&idx);
                if v < best_val {
                    best_val = v;
                    best_flat_idx = flat as i64;
                }
                if !t.shape.is_empty() {
                    increment_index(&mut idx, &t.shape);
                }
            }
            Ok(CpuStorage::from_contiguous(
                index_buffer::<KInt>("argmin", &[best_flat_idx])?,
                vec![],
            ))
        }
    }
}

/// `topk`.
pub(crate) fn topk<KInt: DType>(
    t: &CpuStorage,
    k: usize,
    dim: usize,
    largest: bool,
) -> Result<(CpuStorage, CpuStorage)> {
    let shape = t.shape.to_vec();
    if dim >= shape.len() {
        return Err(Error::ShapeMismatch {
            op: "topk",
            expected: shape.to_vec(),
            got: vec![dim],
            msg: format!("topk: axis {} out of range", dim),
        });
    }
    let k = k.min(shape[dim]);
    let mut out_shape = shape.clone();
    out_shape[dim] = k;

    let mut base_shape = shape.clone();
    base_shape[dim] = 1;
    let n_slices = crate::cpu::stride::checked_numel(&base_shape)?;

    let out_len = crate::cpu::stride::checked_numel(&out_shape)?;
    // The values keep the operand's dtype. Accumulating them as `f64` and
    // converting through the operand's own buffer at the end is what makes
    // that true: the buffer used to be built as `F32` whatever was read,
    // so a `f64` or `f16` operand came back relabelled and narrowed.
    let mut out_vals = vec![0.0f64; out_len];
    let mut out_indices = vec![0i64; out_len];

    for i in 0..n_slices {
        let mut rem = i;
        let mut coords = vec![0usize; shape.len()];
        for dd in (0..shape.len()).rev() {
            coords[dd] = rem % base_shape[dd];
            rem /= base_shape[dd];
        }

        let mut slice_vals = Vec::with_capacity(shape[dim]);
        for j in 0..shape[dim] {
            coords[dim] = j;
            slice_vals.push((
                t.get(&coords),
                i64::try_from(j).map_err(|_| ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Reduction,
                    expression: "topk index does not fit i64",
                })?,
            ));
        }
        if largest {
            slice_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        } else {
            slice_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        }

        let mut out_coords = coords.clone();
        for (j, &(val, idx)) in slice_vals.iter().enumerate().take(k) {
            out_coords[dim] = j;
            let flat = flatten_index(&out_coords, &out_shape);
            out_vals[flat] = val;
            out_indices[flat] = idx;
        }
    }
    Ok((
        CpuStorage::from_contiguous(t.buffer.from_f64_values(out_vals)?, &out_shape),
        CpuStorage::from_contiguous(index_buffer::<KInt>("topk", &out_indices)?, &out_shape),
    ))
}

/// `argsort`.
pub(crate) fn argsort<KInt: DType>(
    t: &CpuStorage,
    dim: usize,
    descending: bool,
) -> Result<CpuStorage> {
    let shape = t.shape.to_vec();
    if dim >= shape.len() {
        return Err(Error::ShapeMismatch {
            op: "argsort",
            expected: shape.to_vec(),
            got: vec![dim],
            msg: format!("argsort: axis {} out of range", dim),
        });
    }
    let mut base_shape = shape.clone();
    base_shape[dim] = 1;
    let n_slices = crate::cpu::stride::checked_numel(&base_shape)?;
    let mut out = vec![0i64; crate::cpu::stride::checked_numel(&shape)?];

    for i in 0..n_slices {
        let mut rem = i;
        let mut coords = vec![0usize; shape.len()];
        for dd in (0..shape.len()).rev() {
            coords[dd] = rem % base_shape[dd];
            rem /= base_shape[dd];
        }

        let mut slice_vals = Vec::with_capacity(shape[dim]);
        for k in 0..shape[dim] {
            coords[dim] = k;
            slice_vals.push((
                t.get(&coords),
                i64::try_from(k).map_err(|_| ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Reduction,
                    expression: "argsort index does not fit i64",
                })?,
            ));
        }
        if descending {
            slice_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
        } else {
            slice_vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
        }
        for (k, &(_, idx)) in slice_vals.iter().enumerate() {
            coords[dim] = k;
            let flat = flatten_index(&coords, &shape);
            out[flat] = idx;
        }
    }
    Ok(CpuStorage::from_contiguous(
        index_buffer::<KInt>("argsort", &out)?,
        &shape,
    ))
}
