use super::*;

pub(crate) fn triu_storage(t: &CpuStorage, k: i64) -> Result<CpuStorage> {
    triangular_storage(t, k, true)
}

pub(crate) fn tril_storage(t: &CpuStorage, k: i64) -> Result<CpuStorage> {
    triangular_storage(t, k, false)
}

pub(crate) fn diag_storage(t: &CpuStorage, k: i64) -> Result<CpuStorage> {
    let rank = t.shape.len();
    if rank == 1 {
        return diagonal_matrix(t, k);
    }
    diagonal_vector(t, k)
}

/// Rank-one operand -> diagonal matrix.
fn diagonal_matrix(t: &CpuStorage, k: i64) -> Result<CpuStorage> {
    let n = t.shape[0];
    let k_abs = k.unsigned_abs() as usize;
    let out_dim = n.checked_add(k_abs).ok_or(ShapeError::ArithmeticOverflow {
        operation: OperationKind::Storage,
        expression: "diagonal output dimension",
    })?;
    let out_total =
        ShapeBuf::from_slice(&[out_dim, out_dim]).checked_numel(OperationKind::Storage)?;
    let mut out = vec![0.0f64; out_total];
    for i in 0..n {
        let row = if k >= 0 { i } else { i + k_abs };
        let col = if k >= 0 { i + k_abs } else { i };
        if row < out_dim && col < out_dim {
            out[row * out_dim + col] = t.get(&[i]);
        }
    }
    let out_storage =
        CpuStorage::from_contiguous(t.buffer.from_f64_values(out)?, vec![out_dim, out_dim]);

    // Vector -> diagonal matrix: the cotangent is the output cotangent's own
    // diagonal, read back through the same offset the forward wrote.
    let (t_id, out_id) = (t.id, out_storage.id);
    tape::push_with(move || TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let mut vals = Vec::with_capacity(n);
            for i in 0..n {
                let row = if k >= 0 { i } else { i + k_abs };
                let col = if k >= 0 { i + k_abs } else { i };
                vals.push(if row < out_dim && col < out_dim {
                    grad_out.get(&[row, col])
                } else {
                    0.0
                });
            }
            Ok(vec![CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(vals)?,
                [n],
            )])
        }),
    });
    Ok(out_storage)
}

/// Matrix-or-higher operand -> diagonal vector.
///
/// The forward reads the diagonal of the FIRST trailing matrix only (every
/// leading coordinate stays zero), so that is exactly where its backward
/// routes the incoming cotangent.
fn diagonal_vector(t: &CpuStorage, k: i64) -> Result<CpuStorage> {
    let rank = t.shape.len();
    let row_len = t.shape[rank - 2];
    let col_len = t.shape[rank - 1];
    let mut values = Vec::new();
    let mut diag_positions: Vec<(usize, usize)> = Vec::new();
    for row in 0..row_len {
        let Some(col) = (row as i64).checked_add(k).filter(|&col| col >= 0) else {
            continue;
        };
        let col = col as usize;
        if col < col_len {
            let mut index = vec![0; rank];
            index[rank - 2] = row;
            index[rank - 1] = col;
            values.push(t.get(&index));
            diag_positions.push((row, col));
        }
    }
    let len = values.len();
    let out_storage = CpuStorage::from_contiguous(t.buffer.from_f64_values(values)?, vec![len]);

    let original_shape = t.shape.to_vec();
    let strides = crate::cpu::stride::contiguous_strides(&original_shape);
    let (t_id, out_id) = (t.id, out_storage.id);
    tape::push_with(move || TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let mut grads = vec![0.0; crate::cpu::stride::checked_numel(&original_shape)?];
            for (i, &(row, col)) in diag_positions.iter().enumerate() {
                let mut index = vec![0usize; original_shape.len()];
                index[original_shape.len() - 2] = row;
                index[original_shape.len() - 1] = col;
                let flat: usize = index
                    .iter()
                    .zip(strides.iter())
                    .map(|(&coordinate, &stride)| coordinate * stride)
                    .sum();
                grads[flat] += grad_out.get(&[i]);
            }
            Ok(vec![CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(grads)?,
                &original_shape,
            )])
        }),
    });
    Ok(out_storage)
}

fn triangular_storage(t: &CpuStorage, k: i64, upper: bool) -> Result<CpuStorage> {
    let rank = t.shape.len();
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; rank];
    for _ in 0..total {
        let (row, col) = if rank >= 2 {
            (idx[rank - 2] as i64, idx[rank - 1] as i64)
        } else {
            (0, idx[0] as i64)
        };
        let keep = if upper {
            col >= row + k
        } else {
            col <= row + k
        };
        out.push(if keep { t.get(&idx) } else { 0.0 });
        if !t.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &t.shape);
        }
    }
    let out_storage = CpuStorage::from_contiguous(t.buffer.from_f64_values(out)?, &t.shape);

    // Zeroing is its own transpose: the mask that produced the output masks
    // the incoming cotangent identically.
    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out_storage.id);
    tape::push_with(move || TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let total = crate::cpu::stride::checked_numel(&original_shape)?;
            let mut vals = Vec::with_capacity(total);
            let mut idx = vec![0usize; original_shape.len()];
            let grad_rank = original_shape.len();
            for _ in 0..total {
                let (row, col) = if grad_rank >= 2 {
                    (idx[grad_rank - 2] as i64, idx[grad_rank - 1] as i64)
                } else {
                    (0, idx[0] as i64)
                };
                let keep = if upper {
                    col >= row + k
                } else {
                    col <= row + k
                };
                vals.push(if keep { grad_out.get(&idx) } else { 0.0 });
                if !original_shape.is_empty() {
                    crate::cpu::storage::increment_index(&mut idx, &original_shape);
                }
            }
            Ok(vec![CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(vals)?,
                &original_shape,
            )])
        }),
    });
    Ok(out_storage)
}
