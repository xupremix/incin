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
    let out_storage = CpuStorage::from_contiguous(buffer, &t.shape);

    // The filled value is a constant, so the Selection profile's piecewise
    // gradient reaches only the input: positions under a true mask receive
    // nothing.
    let mask_cap = mask.clone();
    let (t_id, out_id) = (t.id, out_storage.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let total = crate::cpu::stride::checked_numel(&grad_out.shape)?;
            let mut vals = Vec::with_capacity(total);
            let mut idx = vec![0usize; grad_out.shape.len()];
            for _ in 0..total {
                vals.push(if mask_cap.get_bool(&idx) {
                    0.0
                } else {
                    grad_out.get(&idx)
                });
                if !grad_out.shape.is_empty() {
                    crate::cpu::storage::increment_index(&mut idx, &grad_out.shape);
                }
            }
            Ok(vec![CpuStorage::from_contiguous(
                grad_out.buffer.from_f64_values(vals)?,
                &grad_out.shape,
            )])
        }),
    });
    Ok(out_storage)
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
    let out_storage = CpuStorage::from_contiguous(t.buffer.from_f64_values(out)?, out_shape);

    // Same cotangent gather already used by `gather_storage`: every output
    // position routes its gradient back to the source position its index
    // named, accumulating where an index selects the same row twice. The
    // integer index operand is off the tape by construction.
    let t_cap = t.clone();
    let (t_id, out_id) = (t.id, out_storage.id);
    tape::push_with(move || TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let t_total = crate::cpu::stride::checked_numel(&t_cap.shape)?;
            let mut grad_t_data = vec![0.0; t_total];
            let strides = crate::cpu::stride::contiguous_strides(&t_cap.shape);
            let grad_total = crate::cpu::stride::checked_numel(&grad_out.shape)?;
            let mut grad_idx = vec![0usize; grad_out.shape.len()];
            for _ in 0..grad_total {
                let selected_pos = index_values[grad_idx[dim].min(index_values.len() - 1)] as usize;
                let mut src_idx = grad_idx.clone();
                src_idx[dim] = selected_pos;
                let flat_dst: usize = src_idx
                    .iter()
                    .zip(strides.iter())
                    .map(|(&i, &stride)| i * stride)
                    .sum();
                grad_t_data[flat_dst] += grad_out.get(&grad_idx);
                if !grad_out.shape.is_empty() {
                    crate::cpu::storage::increment_index(&mut grad_idx, &grad_out.shape);
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
    let mut written_flat_dest = Vec::with_capacity(index_total);
    let mut written_src_idx: Vec<Vec<usize>> = Vec::with_capacity(index_total);
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
        written_flat_dest.push(flat_dest);
        written_src_idx.push(idx.clone());
        if !index.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &index.shape);
        }
    }
    let out_storage = CpuStorage::from_contiguous(t.buffer.from_f64_values(out_data)?, &t.shape);

    // The input keeps its cotangent everywhere EXCEPT the positions a write
    // overwrote. The source receives the output cotangent only through the
    // LAST write to each destination - the forward's last-write-wins rule
    // means earlier writes to the same position contributed nothing. The
    // integer index operand is off the tape by construction.
    let t_cap = t.clone();
    let source_cap = source.clone();
    let mut last_write_of_dest: std::collections::BTreeMap<usize, usize> =
        std::collections::BTreeMap::new();
    for (position, &flat_dest) in written_flat_dest.iter().enumerate() {
        last_write_of_dest.insert(flat_dest, position);
    }
    let surviving_writes: Vec<(usize, Vec<usize>)> = last_write_of_dest
        .into_iter()
        .filter_map(|(flat_dest, position)| {
            written_src_idx
                .get(position)
                .cloned()
                .map(|src_idx| (flat_dest, src_idx))
        })
        .collect();
    let (t_id, source_id, out_id) = (t.id, source.id, out_storage.id);
    tape::push_with(move || TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id, source_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let t_total = crate::cpu::stride::checked_numel(&t_cap.shape)?;
            let mut grad_source = vec![0.0; crate::cpu::stride::checked_numel(&source_cap.shape)?];
            let written_positions: Vec<usize> =
                surviving_writes.iter().map(|&(flat, _)| flat).collect();
            let mut grad_t = Vec::with_capacity(t_total);
            for i in 0..t_total {
                grad_t.push(if written_positions.contains(&i) {
                    0.0
                } else {
                    grad_out.get(&crate::cpu::ops::elementwise::flat_to_nd(
                        i,
                        &grad_out.shape,
                    ))
                });
            }
            for (flat_dest, src_idx) in &surviving_writes {
                if *flat_dest < t_total {
                    let flat_src = flatten_index_checked(src_idx, &source_cap.shape);
                    grad_source[flat_src] += grad_out.get(
                        &crate::cpu::ops::elementwise::flat_to_nd(*flat_dest, &grad_out.shape),
                    );
                }
            }
            Ok(vec![
                CpuStorage::from_contiguous(grad_out.buffer.from_f64_values(grad_t)?, &t_cap.shape),
                CpuStorage::from_contiguous(
                    grad_out.buffer.from_f64_values(grad_source)?,
                    &source_cap.shape,
                ),
            ])
        }),
    });
    Ok(out_storage)
}

/// Flat row-major index of `idx` within `shape`, saturating each coordinate
/// into range so an out-of-bounds write target cannot panic in backward.
fn flatten_index_checked(idx: &[usize], shape: &[usize]) -> usize {
    let strides = crate::cpu::stride::contiguous_strides(shape);
    let mut flat = 0usize;
    for (axis, (&coordinate, &stride)) in idx.iter().zip(strides.iter()).enumerate() {
        let bound = shape.get(axis).copied().unwrap_or(1).max(1);
        flat += coordinate.min(bound - 1) * stride;
    }
    flat
}
