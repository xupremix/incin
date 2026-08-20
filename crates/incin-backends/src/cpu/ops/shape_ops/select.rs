use super::*;

pub(crate) fn masked_fill_storage(
    t: &CpuStorage,
    mask: &CpuStorage,
    value: f64,
) -> Result<CpuStorage> {
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; t.shape.len()];
    for _ in 0..total {
        out.push(if mask.get_bool(&idx) {
            value
        } else {
            t.get(&idx)
        });
        if !t.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &t.shape);
        }
    }
    let buffer = t.buffer.from_f64_values(out)?;
    Ok(CpuStorage::from_contiguous(buffer, &t.shape))
}

pub(crate) fn where_storage(
    mask: &CpuStorage,
    on_true: &CpuStorage,
    on_false: &CpuStorage,
) -> Result<CpuStorage> {
    let out_shape = crate::cpu::stride::broadcast_shape(&on_true.shape, &on_false.shape)?;
    let total = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        out.push(if mask.get_bool(&idx) {
            on_true.get(&idx)
        } else {
            on_false.get(&idx)
        });
        if !out_shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &out_shape);
        }
    }
    let out_storage = CpuStorage::from_contiguous(on_true.buffer.from_f64_values(out)?, out_shape);
    let (mask_cap, on_true_cap, on_false_cap) = (mask.clone(), on_true.clone(), on_false.clone());
    let (true_id, false_id, out_id) = (on_true.id, on_false.id, out_storage.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![true_id, false_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let total = crate::cpu::stride::checked_numel(&grad_out.shape)?;
            let mut grad_true = Vec::with_capacity(total);
            let mut grad_false = Vec::with_capacity(total);
            let mut idx = vec![0usize; grad_out.shape.len()];
            for _ in 0..total {
                let gradient = grad_out.get(&idx);
                if mask_cap.get_bool(&idx) {
                    grad_true.push(gradient);
                    grad_false.push(0.0);
                } else {
                    grad_true.push(0.0);
                    grad_false.push(gradient);
                }
                if !grad_out.shape.is_empty() {
                    crate::cpu::storage::increment_index(&mut idx, &grad_out.shape);
                }
            }
            let grad_true = CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(grad_true)?,
                &grad_out.shape,
            );
            let grad_false = CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(grad_false)?,
                &grad_out.shape,
            );
            Ok(vec![
                tape::unbroadcast(&grad_true, &on_true_cap.shape)?,
                tape::unbroadcast(&grad_false, &on_false_cap.shape)?,
            ])
        }),
    });
    Ok(out_storage)
}

pub(crate) fn gather_storage(t: &CpuStorage, dim: usize, index: &CpuStorage) -> Result<CpuStorage> {
    let out_shape = index.shape.to_vec();
    let total = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        let target_i = index.get(&idx) as usize;
        let mut src_idx = idx.clone();
        src_idx[dim] = target_i;
        out.push(t.get(&src_idx));
        if !out_shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &out_shape);
        }
    }
    let out_storage = CpuStorage::from_contiguous(t.buffer.from_f64_values(out)?, out_shape);
    let (t_cap, index_cap) = (t.clone(), index.clone());
    let (t_id, out_id) = (t.id, out_storage.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let t_total = crate::cpu::stride::checked_numel(&t_cap.shape)?;
            let mut grad_t_data = vec![0.0; t_total];
            let index_total = crate::cpu::stride::checked_numel(&index_cap.shape)?;
            let mut idx = vec![0usize; index_cap.shape.len()];
            for _ in 0..index_total {
                let target_i = index_cap.get(&idx) as usize;
                let mut src_idx = idx.clone();
                src_idx[dim] = target_i;
                let strides = crate::cpu::stride::contiguous_strides(&t_cap.shape);
                let flat_dst: usize = src_idx
                    .iter()
                    .zip(strides.iter())
                    .map(|(&i, &stride)| i * stride)
                    .sum();
                grad_t_data[flat_dst] += grad_out.get(&idx);
                if !index_cap.shape.is_empty() {
                    crate::cpu::storage::increment_index(&mut idx, &index_cap.shape);
                }
            }
            Ok(vec![CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(grad_t_data)?,
                &t_cap.shape,
            )])
        }),
    });
    Ok(out_storage)
}

pub(crate) fn index_select_storage(
    t: &CpuStorage,
    dim: usize,
    index: &CpuStorage,
) -> Result<CpuStorage> {
    let index_total = crate::cpu::stride::checked_numel(&index.shape)?;
    let index_values: Vec<f64> = (0..index_total)
        .map(|i| index.get(&crate::cpu::ops::elementwise::flat_to_nd(i, &index.shape)))
        .collect();
    let mut out_shape = t.shape.to_vec();
    out_shape[dim] = index_values.len();
    let total = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    let mut out_idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        let selected_pos = index_values[out_idx[dim]] as usize;
        let mut src_idx = out_idx.clone();
        src_idx[dim] = selected_pos;
        out.push(t.get(&src_idx));
        if !out_shape.is_empty() {
            crate::cpu::storage::increment_index(&mut out_idx, &out_shape);
        }
    }
    Ok(CpuStorage::from_contiguous(
        t.buffer.from_f64_values(out)?,
        out_shape,
    ))
}

pub(crate) fn scatter_storage(
    t: &CpuStorage,
    dim: usize,
    index: &CpuStorage,
    source: &CpuStorage,
) -> Result<CpuStorage> {
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut out_data: Vec<f64> = (0..total)
        .map(|i| t.get(&crate::cpu::ops::elementwise::flat_to_nd(i, &t.shape)))
        .collect();
    let index_total = crate::cpu::stride::checked_numel(&index.shape)?;
    let mut idx = vec![0usize; index.shape.len()];
    for _ in 0..index_total {
        let target_i = index.get(&idx) as usize;
        let mut dest_idx = idx.clone();
        dest_idx[dim] = target_i;
        let strides = crate::cpu::stride::contiguous_strides(&t.shape);
        let flat_dest: usize = dest_idx
            .iter()
            .zip(strides.iter())
            .map(|(&i, &stride)| i * stride)
            .sum();
        if flat_dest < out_data.len() {
            out_data[flat_dest] = source.get(&idx);
        }
        if !index.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &index.shape);
        }
    }
    Ok(CpuStorage::from_contiguous(
        t.buffer.from_f64_values(out_data)?,
        &t.shape,
    ))
}
