use super::*;

/// Fail-closed admission of `operand` against the geometry the kernel will
/// iterate (`out_shape`): the right-aligned broadcast of the pair must
/// resolve *back* to `out_shape` exactly. A mask that would enlarge the
/// output, or that disagrees with it on an axis, is refused here rather than
/// reaching the element loop, whose coordinates are `out_shape`'s (#100).
/// The message matches `validated.rs`'s MaskedFill rule verbatim, so a
/// caller sees the same sentence whichever layer rejects first.
fn admit_broadcast_operand(
    operation: &'static str,
    operand_shape: &[usize],
    out_shape: &[usize],
    msg: &'static str,
) -> Result<()> {
    // Incompatible axes and axes that would *enlarge* `out_shape` both land
    // in the same refusal: neither resolves back to the geometry the loop is
    // about to index with, and `validated.rs` words them identically.
    let resolves_to_out = crate::layout::broadcast_shape(out_shape, operand_shape)
        .is_ok_and(|resolved| resolved.as_slice() == out_shape);
    if resolves_to_out {
        return Ok(());
    }
    Err(Error::ShapeMismatch {
        op: operation,
        expected: out_shape.to_vec(),
        got: operand_shape.to_vec(),
        msg: msg.into(),
    })
}

/// Read the bool operand at output coordinate `idx`, right-aligned per
/// NumPy broadcasting (#100): axes the operand does not reach are implicit
/// size-1, and a size-1 axis clamps the coordinate to 0. The caller admits
/// the operand into `out_shape` first (`admit_broadcast_operand`, or the
/// checked broadcast fold that *built* `out_shape`), so the operand's rank
/// never exceeds `out_shape`'s and every clamped coordinate is in range.
/// The equal-shape fast path keeps the dominant same-rank case exactly what
/// it was before #100: a direct `get_bool` with no intermediate index.
fn mask_at(mask: &CpuStorage, out_shape: &[usize], idx: &[usize]) -> bool {
    if mask.shape.dims() == out_shape {
        return mask.get_bool(idx);
    }
    let start = out_shape.len() - mask.shape.len();
    let mut mask_idx = Vec::with_capacity(mask.shape.len());
    for (axis, &extent) in mask.shape.iter().enumerate() {
        let coordinate = idx[start + axis];
        mask_idx.push(if extent == 1 { 0 } else { coordinate });
    }
    mask.get_bool(&mask_idx)
}

/// [`mask_at`] for a value operand: `where_cond`'s `on_true`/`on_false` may
/// broadcast among themselves (`Dyn` operands of one type can differ at run
/// time), so the f64 read right-aligns the same way the mask read does.
fn operand_at(operand: &CpuStorage, out_shape: &[usize], idx: &[usize]) -> f64 {
    if operand.shape.dims() == out_shape {
        return operand.get(idx);
    }
    let start = out_shape.len() - operand.shape.len();
    let mut operand_idx = Vec::with_capacity(operand.shape.len());
    for (axis, &extent) in operand.shape.iter().enumerate() {
        let coordinate = idx[start + axis];
        operand_idx.push(if extent == 1 { 0 } else { coordinate });
    }
    operand.get(&operand_idx)
}

pub(crate) fn masked_fill_storage(
    t: &CpuStorage,
    mask: &CpuStorage,
    value: f64,
) -> Result<CpuStorage> {
    // #100's frontend admits masks that broadcast *into* the input; the
    // output geometry is the input's own, so the mask must resolve back to
    // it before the loop indexes with input coordinates.
    admit_broadcast_operand(
        "masked_fill",
        &mask.shape,
        &t.shape,
        "mask must broadcast to the input shape",
    )?;
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; t.shape.len()];
    for _ in 0..total {
        out.push(if mask_at(mask, &t.shape, &idx) {
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
                vals.push(if mask_at(&mask_cap, &grad_out.shape, &idx) {
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
    // Fold the mask into the output shape the way CUDA's `Execute<WhereCond>`
    // does: base = broadcast(on_true, on_false), out = broadcast(mask, base).
    // Under #100's directional pin the public API already guarantees the mask
    // broadcasts into the data, so the fold is the data's shape in every
    // reachable invocation; it exists so a rank-deficit mask cannot make the
    // element loop index with coordinates the mask does not have, and so a
    // direct storage-level call fails closed (via `broadcast_shape`) instead
    // of reading out of bounds.
    let data_shape = crate::cpu::stride::broadcast_shape(&on_true.shape, &on_false.shape)?;
    let out_shape = crate::cpu::stride::broadcast_shape(&mask.shape, &data_shape)?;
    let total = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..total {
        out.push(if mask_at(mask, &out_shape, &idx) {
            operand_at(on_true, &out_shape, &idx)
        } else {
            operand_at(on_false, &out_shape, &idx)
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
                if mask_at(&mask_cap, &grad_out.shape, &idx) {
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
            // Unbroadcast, not truncate: when the operands broadcast into the
            // output (rank-deficit or size-1 axes under #100), each cotangent
            // must be summed back to its own operand's shape before
            // accumulation sees it.
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

/// `scatter_add`: like [`scatter_storage`], except colliding writes sum rather
/// than the last one winning.
///
/// The difference is one operator in the forward loop and a much shorter
/// backward. `scatter_storage` has to work out which write to each destination
/// survived, because only that one earned the cotangent and the rest
/// contributed nothing to the result. Summing keeps every contribution, so
/// there is no survivor to identify: each write takes the output cotangent at
/// the destination it wrote, and the target keeps its own everywhere, since
/// adding to a value does not displace it.
///
/// That is what makes this the combine step for a top-`k` router. A token sent
/// to `k` experts writes `k` times to the same row, and under
/// [`scatter_storage`] that row would keep one expert's output and silently
/// discard the other `k - 1`, gradients included.
///
/// Accumulation runs in row-major order of `index`. Floating-point addition is
/// not associative, so that order is part of the contract rather than an
/// implementation detail, and it is why the catalog row claims determinism: a
/// backend summing with atomics would produce a different low bit run to run
/// and could not advertise this operation.
pub(crate) fn scatter_add_storage(
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
    let strides = crate::cpu::stride::contiguous_strides(&t.shape);
    // Every write is recorded, not just the last one to each destination, which
    // is the whole difference from `scatter_storage`'s bookkeeping.
    let mut writes: Vec<(usize, Vec<usize>)> = Vec::with_capacity(index_total);
    let mut idx = vec![0usize; index.shape.len()];
    for _ in 0..index_total {
        let target_i = index.get(&idx) as usize;
        let mut dest_idx = idx.clone();
        dest_idx[dim] = target_i;
        let flat_dest: usize = dest_idx
            .iter()
            .zip(strides.iter())
            .map(|(&i, &stride)| i * stride)
            .sum();
        // Out-of-range destinations are dropped rather than clamped, matching
        // `scatter_storage`: clamping would silently add a contribution to a
        // row the caller never named.
        if flat_dest < out_data.len() {
            out_data[flat_dest] += source.get(&idx);
            writes.push((flat_dest, idx.clone()));
        }
        if !index.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &index.shape);
        }
    }
    let out_storage = CpuStorage::from_contiguous(t.buffer.from_f64_values(out_data)?, &t.shape);

    let t_cap = t.clone();
    let source_cap = source.clone();
    let (t_id, source_id, out_id) = (t.id, source.id, out_storage.id);
    tape::push_with(move || TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id, source_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let t_total = crate::cpu::stride::checked_numel(&t_cap.shape)?;
            // The target is passed through by addition, so its cotangent is the
            // output's untouched. No position is zeroed the way an overwriting
            // scatter has to zero the ones it clobbered.
            let grad_t: Vec<f64> = (0..t_total)
                .map(|i| {
                    grad_out.get(&crate::cpu::ops::elementwise::flat_to_nd(
                        i,
                        &grad_out.shape,
                    ))
                })
                .collect();
            let mut grad_source = vec![0.0; crate::cpu::stride::checked_numel(&source_cap.shape)?];
            for (flat_dest, src_idx) in &writes {
                let flat_src = flatten_index_checked(src_idx, &source_cap.shape);
                grad_source[flat_src] += grad_out.get(&crate::cpu::ops::elementwise::flat_to_nd(
                    *flat_dest,
                    &grad_out.shape,
                ));
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

/// `one_hot`: encode each integer index as a row of `depth` booleans.
///
/// One operator per input element and no backward: the output is a boolean
/// function of which slot an integer names, so there is no cotangent to route
/// and no tape entry is pushed. The descriptor already refused a non-integer
/// operand and a zero depth before this runs, so neither is re-checked here.
///
/// An index outside `[0, depth)` encodes as an all-`false` row rather than an
/// error, matching ONNX `OneHot`: the value names no slot, so no slot is set,
/// and a router's padding index flows through instead of aborting the batch.
pub(crate) fn one_hot_storage(t: &CpuStorage, depth: usize) -> Result<CpuStorage> {
    let mut out_shape = t.shape.clone();
    out_shape.push(depth);
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut out = vec![0u8; total * depth];
    let mut idx = vec![0usize; t.shape.len()];
    for flat in 0..total {
        let value = t.get(&idx);
        if value >= 0.0 && value < depth as f64 {
            out[flat * depth + value as usize] = 1;
        }
        if !t.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &t.shape);
        }
    }
    Ok(CpuStorage::from_contiguous(CpuBuffer::Bool(out), out_shape))
}

/// Counts how many times each of `bins` slots appears in an index operand.
///
/// An index outside the range is an error rather than a skipped element: a
/// count that is quietly low makes every offset derived from it wrong, and
/// nothing downstream can tell that from a genuinely empty bin.
pub(crate) fn bincount_storage(t: &CpuStorage, bins: usize) -> Result<CpuStorage> {
    if bins == 0 {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::Bincount,
            reason: "bincount needs at least one bin to count into",
        }));
    }
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut counts = alloc::vec![0i64; bins];
    let mut idx = alloc::vec![0usize; t.shape.len()];
    for _ in 0..total {
        let value = t.get(&idx);
        // `is_finite` first so a NaN index is refused rather than slipping
        // through comparisons that are all false for it.
        let addressable =
            value.is_finite() && value >= 0.0 && value < bins as f64 && value.fract() == 0.0;
        if !addressable {
            return Err(Error::Backend(BackendError::InvalidInput {
                operation: OperationKind::Bincount,
                reason: "bincount index is not a whole number inside the bin range",
            }));
        }
        counts[value as usize] += 1;
        if !t.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &t.shape);
        }
    }
    Ok(CpuStorage::from_contiguous(
        CpuBuffer::I64(counts),
        alloc::vec![bins],
    ))
}

/// Row-major coordinates of every non-zero element, one `[count, rank]` i64
/// result.
///
/// The count is not known until the scan finishes, which is why the output
/// extent is `DataDependent` and the public API reads the shape back from the
/// returned storage rather than pre-asserting one. A scalar (rank 0) yields
/// `[count, 0]`: there are positions (zero or one) but no coordinates to name
/// them with, which is the same degenerate form a zero-extent axis produces.
///
/// No tape entry is pushed: the coordinates are addresses, not values, the
/// same reason `one_hot` and the rest of the `GradientRule::None` family
/// record nothing.
pub(crate) fn nonzero_storage(t: &CpuStorage) -> Result<CpuStorage> {
    let rank = t.shape.len();
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    let mut indices: alloc::vec::Vec<i64> = alloc::vec::Vec::new();
    let mut count = 0usize;
    let mut idx = alloc::vec![0usize; rank];
    for _ in 0..total {
        if t.get(&idx) != 0.0 {
            for &coordinate in &idx {
                indices.push(coordinate as i64);
            }
            count += 1;
        }
        if rank > 0 {
            crate::cpu::storage::increment_index(&mut idx, &t.shape);
        }
    }
    Ok(CpuStorage::from_contiguous(
        CpuBuffer::I64(indices),
        alloc::vec![count, rank],
    ))
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
