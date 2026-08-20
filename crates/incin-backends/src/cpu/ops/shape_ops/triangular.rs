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
        return Ok(CpuStorage::from_contiguous(
            t.buffer.from_f64_values(out)?,
            vec![out_dim, out_dim],
        ));
    }
    let row_len = t.shape[rank - 2];
    let col_len = t.shape[rank - 1];
    let mut values = Vec::new();
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
        }
    }
    let len = values.len();
    Ok(CpuStorage::from_contiguous(
        t.buffer.from_f64_values(values)?,
        vec![len],
    ))
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
    Ok(CpuStorage::from_contiguous(
        t.buffer.from_f64_values(out)?,
        &t.shape,
    ))
}
