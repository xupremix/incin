use super::*;

/// Sum over `dim`, removing that axis from the output shape.
/// (e.g. `[2, 3]` over dim 0 → `[3]`)
pub(crate) fn sum_dim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "sum_dim",
            expected: t.shape.to_vec(),
            got: vec![dim],
            msg: format!("sum_dim: axis {dim} out of range for shape {:?}", t.shape),
        });
    }
    let out = sum_axis_squeeze(t, dim)?;

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            // Backward of sum_dim (squeeze): reinsert the axis with size 1,
            // then broadcast back to the original shape.
            let mut keepdim_shape = grad_out.shape.to_vec();
            keepdim_shape.insert(dim, 1);
            let keepdim = grad_out.reshape(&keepdim_shape)?;
            let expanded = keepdim.broadcast_as(&original_shape)?;
            // Materialize the broadcast view (walk all elements) so the
            // gradient is a concrete contiguous tensor, not a strided view
            // that upstream accumulation might mis-sum.
            let total: usize = crate::cpu::stride::validated_numel(&(original_shape));
            let mut idx = vec![0usize; original_shape.len()];
            let mut vals = Vec::with_capacity(total);
            for _ in 0..total {
                vals.push(expanded.get(&idx) as f32);
                increment_index(&mut idx, &original_shape);
            }
            Ok(vec![CpuStorage::from_contiguous(
                CpuBuffer::F32(vals),
                &original_shape,
            )])
        }),
    });

    Ok(out)
}

/// Sum over `dim`, keeping that axis as size 1.
/// (e.g. `[2, 3]` over dim 0 → `[1, 3]`)
pub(crate) fn sum_keepdim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "sum_keepdim",
            expected: t.shape.to_vec(),
            got: vec![dim],
            msg: format!(
                "sum_keepdim: axis {dim} out of range for shape {:?}",
                t.shape
            ),
        });
    }
    let out = sum_axis_keepdim(t, dim)?;

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            // Backward of sum_keepdim: broadcast the keepdim gradient
            // (which already has size 1 on `dim`) back to the original
            // shape, then materialize it.
            let expanded = grad_out.broadcast_as(&original_shape)?;
            let total: usize = crate::cpu::stride::validated_numel(&(original_shape));
            let mut idx = vec![0usize; original_shape.len()];
            let mut vals = Vec::with_capacity(total);
            for _ in 0..total {
                vals.push(expanded.get(&idx) as f32);
                increment_index(&mut idx, &original_shape);
            }
            Ok(vec![CpuStorage::from_contiguous(
                CpuBuffer::F32(vals),
                &original_shape,
            )])
        }),
    });

    Ok(out)
}

/// Mean over `dim`, removing that axis from the output shape.
/// Thin wrapper over `sum_axis_squeeze`, divided by the axis length.
/// (e.g. `[2, 3]` over dim 0 → `[3]`, each value = column sum / 2)
pub(crate) fn mean_dim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "mean_dim",
            expected: t.shape.to_vec(),
            got: vec![dim],
            msg: format!("mean_dim: axis {dim} out of range for shape {:?}", t.shape),
        });
    }
    let axis_len = t.shape[dim] as f64;
    let summed = sum_axis_squeeze(t, dim)?;
    let out_shape = summed.shape.to_vec();
    let total: usize = crate::cpu::stride::validated_numel(&(out_shape));
    let mut idx = vec![0usize; out_shape.len()];
    let mut vals = Vec::with_capacity(total);
    for _ in 0..total {
        vals.push((summed.get(&idx) / axis_len) as f32);
        increment_index(&mut idx, &out_shape);
    }
    let out = CpuStorage::from_contiguous(CpuBuffer::F32(vals), &out_shape);

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            // Backward of mean_dim (squeeze): reinsert the axis with size
            // 1, broadcast back to the original shape, then scale every
            // materialized value by 1/axis_len (mirrors mean_all's 1/n
            // relationship to sum_all).
            let mut keepdim_shape = grad_out.shape.to_vec();
            keepdim_shape.insert(dim, 1);
            let keepdim = grad_out.reshape(&keepdim_shape)?;
            let expanded = keepdim.broadcast_as(&original_shape)?;
            let total: usize = crate::cpu::stride::validated_numel(&(original_shape));
            let mut idx = vec![0usize; original_shape.len()];
            let mut vals = Vec::with_capacity(total);
            for _ in 0..total {
                vals.push((expanded.get(&idx) / axis_len) as f32);
                increment_index(&mut idx, &original_shape);
            }
            Ok(vec![CpuStorage::from_contiguous(
                CpuBuffer::F32(vals),
                &original_shape,
            )])
        }),
    });

    Ok(out)
}

/// Mean over `dim`, keeping that axis as size 1.
/// Thin wrapper over `sum_axis_keepdim`, divided by the axis length.
/// (e.g. `[2, 3]` over dim 0 → `[1, 3]`)
pub(crate) fn mean_keepdim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "mean_keepdim",
            expected: t.shape.to_vec(),
            got: vec![dim],
            msg: format!(
                "mean_keepdim: axis {dim} out of range for shape {:?}",
                t.shape
            ),
        });
    }
    let axis_len = t.shape[dim] as f64;
    let summed = sum_axis_keepdim(t, dim)?;
    let out_shape = summed.shape.to_vec();
    let total: usize = crate::cpu::stride::validated_numel(&(out_shape));
    let mut idx = vec![0usize; out_shape.len()];
    let mut vals = Vec::with_capacity(total);
    for _ in 0..total {
        vals.push((summed.get(&idx) / axis_len) as f32);
        increment_index(&mut idx, &out_shape);
    }
    let out = CpuStorage::from_contiguous(CpuBuffer::F32(vals), &out_shape);

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            // Backward of mean_keepdim: broadcast the keepdim gradient
            // (already size 1 on `dim`) back to the original shape, then
            // scale by 1/axis_len.
            let expanded = grad_out.broadcast_as(&original_shape)?;
            let total: usize = crate::cpu::stride::validated_numel(&(original_shape));
            let mut idx = vec![0usize; original_shape.len()];
            let mut vals = Vec::with_capacity(total);
            for _ in 0..total {
                vals.push((expanded.get(&idx) / axis_len) as f32);
                increment_index(&mut idx, &original_shape);
            }
            Ok(vec![CpuStorage::from_contiguous(
                CpuBuffer::F32(vals),
                &original_shape,
            )])
        }),
    });

    Ok(out)
}

/// Maximum over `dim`, removing that axis from the output shape.
/// Backward routes gradient to exactly one winning element per output
/// position (T-02-07/T-02-08 mitigations).
pub(crate) fn max_dim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "max_dim",
            expected: t.shape.to_vec(),
            got: vec![dim],
            msg: format!("max_dim: axis {dim} out of range for shape {:?}", t.shape),
        });
    }
    let (keepdim_out, winning_flat_src_idx) = max_axis_with_indices(t, dim)?;
    let mut squeeze_shape = keepdim_out.shape.to_vec();
    squeeze_shape.remove(dim);
    let out = keepdim_out
        .reshape(&squeeze_shape)
        .expect("max_dim: squeeze reshape of size-1 keepdim result cannot fail");

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![scatter_axis_grad(
                grad_out,
                &winning_flat_src_idx,
                &original_shape,
            )])
        }),
    });

    Ok(out)
}

/// Maximum over `dim`, keeping that axis as size 1.
pub(crate) fn max_keepdim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "max_keepdim",
            expected: t.shape.to_vec(),
            got: vec![dim],
            msg: format!(
                "max_keepdim: axis {dim} out of range for shape {:?}",
                t.shape
            ),
        });
    }
    let (out, winning_flat_src_idx) = max_axis_with_indices(t, dim)?;

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![scatter_axis_grad(
                grad_out,
                &winning_flat_src_idx,
                &original_shape,
            )])
        }),
    });

    Ok(out)
}

/// Minimum over `dim`, removing that axis from the output shape. Mirror
/// of `max_dim` using `min_axis_with_indices`.
pub(crate) fn min_dim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "min_dim",
            expected: t.shape.to_vec(),
            got: vec![dim],
            msg: format!("min_dim: axis {dim} out of range for shape {:?}", t.shape),
        });
    }
    let (keepdim_out, winning_flat_src_idx) = min_axis_with_indices(t, dim)?;
    let mut squeeze_shape = keepdim_out.shape.to_vec();
    squeeze_shape.remove(dim);
    let out = keepdim_out
        .reshape(&squeeze_shape)
        .expect("min_dim: squeeze reshape of size-1 keepdim result cannot fail");

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![scatter_axis_grad(
                grad_out,
                &winning_flat_src_idx,
                &original_shape,
            )])
        }),
    });

    Ok(out)
}

/// Minimum over `dim`, keeping that axis as size 1. Mirror of
/// `max_keepdim` using `min_axis_with_indices`.
pub(crate) fn min_keepdim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "min_keepdim",
            expected: t.shape.to_vec(),
            got: vec![dim],
            msg: format!(
                "min_keepdim: axis {dim} out of range for shape {:?}",
                t.shape
            ),
        });
    }
    let (out, winning_flat_src_idx) = min_axis_with_indices(t, dim)?;

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            Ok(vec![scatter_axis_grad(
                grad_out,
                &winning_flat_src_idx,
                &original_shape,
            )])
        }),
    });

    Ok(out)
}

pub(crate) fn prod_dim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    let mut out_shape = t.shape.to_vec();
    out_shape.remove(dim);
    let mut keep_shape = t.shape.to_vec();
    keep_shape[dim] = 1;
    let total: usize = crate::cpu::stride::validated_numel(&(keep_shape));
    let mut prods = vec![1.0f64; total];
    let src_total: usize = crate::cpu::stride::validated_numel(&(t.shape));
    let mut idx = vec![0usize; t.shape.len()];
    for _ in 0..src_total {
        let mut out_idx = idx.clone();
        out_idx[dim] = 0;
        let flat_out = flatten_index(&out_idx, &keep_shape);
        prods[flat_out] *= t.get(&idx);
        increment_index(&mut idx, &t.shape);
    }
    let buffer = t.buffer.from_f64_values(prods)?;
    let storage = CpuStorage::from_contiguous(buffer, keep_shape);
    storage.reshape(&out_shape)
}

pub(crate) fn cumsum(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    let total: usize = crate::cpu::stride::validated_numel(&(t.shape));
    let mut out_data = vec![0.0f64; total];
    let dim_len = t.shape[dim];
    let strides = contiguous_strides(&t.shape);
    let mut idx = vec![0usize; t.shape.len()];
    for _ in 0..total {
        if idx[dim] == 0 {
            let mut current = 0.0f64;
            for step in 0..dim_len {
                let mut step_idx = idx.clone();
                step_idx[dim] = step;
                current += t.get(&step_idx);
                let flat_dest: usize = step_idx
                    .iter()
                    .zip(strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum();
                out_data[flat_dest] = current;
            }
        }
        increment_index(&mut idx, &t.shape);
    }
    let buffer = t.buffer.from_f64_values(out_data)?;
    Ok(CpuStorage::from_contiguous(buffer, &t.shape))
}
