use super::*;

pub(crate) fn concat_storage(tensors: &[&CpuStorage], dim: usize) -> Result<CpuStorage> {
    if tensors.is_empty() {
        return Err(Error::ShapeMismatch {
            op: "concat",
            expected: vec![],
            got: vec![],
            msg: "concat requires at least one input tensor".into(),
        });
    }
    let rank = tensors[0].shape.len();
    if dim >= rank {
        return Err(Error::ShapeMismatch {
            op: "concat",
            expected: tensors[0].shape.to_vec(),
            got: vec![dim],
            msg: format!(
                "concat dim {dim} out of range for rank-{rank} shape {:?}",
                tensors[0].shape
            ),
        });
    }
    for tensor in tensors.iter().skip(1) {
        if tensor.shape.len() != rank {
            return Err(Error::ShapeMismatch {
                op: "concat",
                expected: tensors[0].shape.to_vec(),
                got: tensor.shape.to_vec(),
                msg: format!(
                    "concat requires every input to have the same rank; expected rank {rank}, got shape {:?}",
                    tensor.shape
                ),
            });
        }
        for (axis, (&expected, &actual)) in
            tensors[0].shape.iter().zip(tensor.shape.iter()).enumerate()
        {
            if axis != dim && expected != actual {
                return Err(Error::ShapeMismatch {
                    op: "concat",
                    expected: tensors[0].shape.to_vec(),
                    got: tensor.shape.to_vec(),
                    msg: format!(
                        "concat requires exact equality on every non-concat axis; axis {axis} has size {expected} vs {actual}"
                    ),
                });
            }
        }
    }

    let mut out_shape = tensors[0].shape.to_vec();
    out_shape[dim] = tensors.iter().try_fold(0usize, |total, tensor| {
        total
            .checked_add(tensor.shape[dim])
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Concat,
                expression: "sum of concatenated axis dimensions",
            })
    })?;
    let out_strides = crate::cpu::stride::contiguous_strides(&out_shape);
    let total = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = vec![0.0f64; total];
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut running = 0usize;
    for tensor in tensors {
        offsets.push(running);
        running = running
            .checked_add(tensor.shape[dim])
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Concat,
                expression: "cumulative concatenation offset",
            })?;
        let value_count = crate::cpu::stride::checked_numel(&tensor.shape)?;
        let mut index = vec![0usize; rank];
        for _ in 0..value_count {
            let mut flat_dest = 0usize;
            for (axis, &coordinate) in index.iter().enumerate() {
                let destination = if axis == dim {
                    coordinate + offsets.last().copied().unwrap_or(0)
                } else {
                    coordinate
                };
                flat_dest += destination * out_strides[axis];
            }
            out[flat_dest] = tensor.get(&index);
            crate::cpu::storage::increment_index(&mut index, &tensor.shape);
        }
    }
    let output = CpuStorage::from_contiguous(tensors[0].buffer.from_f64_values(out)?, out_shape);
    let output_id = output.id;
    let input_ids = tensors.iter().map(|tensor| tensor.id).collect();
    let input_dim_sizes = tensors
        .iter()
        .map(|tensor| tensor.shape[dim])
        .collect::<Vec<_>>();
    tape::push_with(|| TapeEntry {
        output_id,
        input_ids,
        backward: Box::new(move |grad_out: &CpuStorage| {
            offsets
                .iter()
                .zip(input_dim_sizes.iter())
                .map(|(&offset, &len)| grad_out.narrow(dim, offset, len))
                .collect()
        }),
    });
    Ok(output)
}

pub(crate) fn stack_storage(tensors: &[&CpuStorage], dim: usize) -> Result<CpuStorage> {
    if tensors.is_empty() {
        return Err(Error::ShapeMismatch {
            op: "stack",
            expected: vec![],
            got: vec![],
            msg: "stack requires at least one input tensor".into(),
        });
    }
    let rank = tensors[0].shape.len();
    if dim > rank {
        return Err(Error::ShapeMismatch {
            op: "stack",
            expected: tensors[0].shape.to_vec(),
            got: vec![dim],
            msg: format!(
                "stack dim {dim} out of range for rank-{rank} shape {:?} (dim may equal rank to append at the end)",
                tensors[0].shape
            ),
        });
    }
    for tensor in tensors.iter().skip(1) {
        if tensor.shape != tensors[0].shape {
            return Err(Error::ShapeMismatch {
                op: "stack",
                expected: tensors[0].shape.to_vec(),
                got: tensor.shape.to_vec(),
                msg: format!(
                    "stack requires every input to have an IDENTICAL shape; expected {:?}, got {:?}",
                    tensors[0].shape, tensor.shape
                ),
            });
        }
    }
    let mut unsqueezed = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let mut target_shape = tensor.shape.to_vec();
        target_shape.insert(dim, 1);
        unsqueezed.push(reshape_storage(tensor, &target_shape)?);
    }
    let refs = unsqueezed.iter().collect::<Vec<_>>();
    concat_storage(&refs, dim)
}

pub(crate) fn unfold_storage(
    t: &CpuStorage,
    dim: usize,
    size: usize,
    step: usize,
) -> Result<CpuStorage> {
    let dim_len = t.shape[dim];
    if size > dim_len {
        return Err(Error::Msg(
            "unfold size cannot exceed dimension length".into(),
        ));
    }
    let n_windows = (dim_len - size) / step + 1;
    let mut out_shape = t.shape.to_vec();
    out_shape[dim] = n_windows;
    out_shape.push(size);
    let total: usize = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        let win_idx = idx[dim];
        let offset_idx = idx[out_shape.len() - 1];
        let mut src_idx = idx[..t.shape.len()].to_vec();
        src_idx[dim] = win_idx * step + offset_idx;
        out.push(t.get(&src_idx));
        crate::cpu::storage::increment_index(&mut idx, &out_shape);
    }
    Ok(CpuStorage::from_contiguous(
        t.buffer.from_f64_values(out)?,
        out_shape,
    ))
}

pub(crate) fn pixel_shuffle_storage(t: &CpuStorage, upscale_factor: usize) -> Result<CpuStorage> {
    if t.shape.len() != 4 {
        return Err(Error::Msg(
            "pixel_shuffle expects 4D tensor (N, C, H, W)".into(),
        ));
    }
    let (n, c, h, w) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let r = upscale_factor;
    let r_sq = r * r;
    if c % r_sq != 0 {
        return Err(Error::Msg(
            "pixel_shuffle channels must be divisible by upscale_factor^2".into(),
        ));
    }
    let out_c = c / r_sq;
    let out_h = h * r;
    let out_w = w * r;
    let out_shape = vec![n, out_c, out_h, out_w];
    let total: usize = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; 4];
    for _ in 0..total {
        let (b, c_out, h_out, w_out) = (idx[0], idx[1], idx[2], idx[3]);
        let h_in = h_out / r;
        let w_in = w_out / r;
        let r_h = h_out % r;
        let r_w = w_out % r;
        let c_in = c_out * r_sq + r_h * r + r_w;
        out.push(t.get(&[b, c_in, h_in, w_in]));
        crate::cpu::storage::increment_index(&mut idx, &out_shape);
    }
    Ok(CpuStorage::from_contiguous(
        t.buffer.from_f64_values(out)?,
        out_shape,
    ))
}

pub(crate) fn repeat_storage(t: &CpuStorage, repeats: &[usize]) -> Result<CpuStorage> {
    if repeats.len() != t.shape.len() {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::Repeat,
            reason: "repeat factors must match tensor rank",
        }));
    }
    let out_shape: Vec<usize> = t
        .shape
        .iter()
        .zip(repeats.iter())
        .map(|(size, repeat)| size * repeat)
        .collect();
    let total = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        let src_idx: Vec<usize> = idx
            .iter()
            .enumerate()
            .map(|(axis, &value)| value % t.shape[axis])
            .collect();
        out.push(t.get(&src_idx));
        if !out_shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &out_shape);
        }
    }
    let buffer = t.buffer.from_f64_values(out)?;
    Ok(CpuStorage::from_contiguous(buffer, out_shape))
}

pub(crate) fn pad_storage(
    t: &CpuStorage,
    padding: &[(usize, usize)],
    value: f64,
) -> Result<CpuStorage> {
    let out_shape: Vec<usize> = t
        .shape
        .iter()
        .zip(padding.iter())
        .map(|(size, &(before, after))| size + before + after)
        .collect();
    let total = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        let mut inside = true;
        let mut src_idx = Vec::with_capacity(idx.len());
        for (axis, &position) in idx.iter().enumerate() {
            let (before, _) = padding[axis];
            if position < before || position >= before + t.shape[axis] {
                inside = false;
                break;
            }
            src_idx.push(position - before);
        }
        out.push(if inside { t.get(&src_idx) } else { value });
        if !out_shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &out_shape);
        }
    }
    let buffer = t.buffer.from_f64_values(out)?;
    Ok(CpuStorage::from_contiguous(buffer, out_shape))
}
