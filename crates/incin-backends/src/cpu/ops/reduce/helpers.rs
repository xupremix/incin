use super::*;

// ---------------------------------------------------------------------------
// Internal axis-reduce helpers (independent of tape.rs's private equivalents)
// ---------------------------------------------------------------------------

/// Compute the flat row-major index of `idx` within `shape`.
pub(super) fn flatten_index(idx: &[usize], shape: &[usize]) -> usize {
    let strides = contiguous_strides(shape);
    idx.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

/// Inverse of `flatten_index`: recover the multi-index within `shape`
/// corresponding to flat row-major index `flat`. Used by `argmax`/`argmin`
/// to resolve a winning FLAT source index into its coordinate along the
/// reduced axis.
pub(super) fn unflatten_index(flat: usize, shape: &[usize]) -> Vec<usize> {
    let strides = contiguous_strides(shape);
    let mut remaining = flat;
    let mut idx = vec![0usize; shape.len()];
    for i in 0..shape.len() {
        if let Some(q) = remaining.checked_div(strides[i]) {
            idx[i] = q;
            remaining %= strides[i];
        } else {
            idx[i] = 0;
        }
    }
    idx
}

/// Builds an index buffer in the dtype the caller asked for.
///
/// `argmax`, `argmin`, `argsort` and `topk` each take an index dtype as a type
/// parameter. All four used to declare it and then hardcode a buffer variant,
/// `I64` for the first two and `U32` for the other two, so asking `argmax` for
/// `u32` indices returned `i64` storage with nothing reporting the difference,
/// and the backend contradicted itself about which integer an index tensor
/// holds.
///
/// Indices are non-negative and bounded by the axis they index, so the only
/// question a narrower dtype raises is whether they fit. They are checked
/// rather than truncated: a silently wrapped index is the failure this
/// function exists to remove, not one to reintroduce in a narrower type.
pub(super) fn index_buffer<KInt: DType>(op: &'static str, indices: &[i64]) -> Result<CpuBuffer> {
    fn narrow<T: TryFrom<i64>>(
        op: &'static str,
        indices: &[i64],
        dtype: DTypeDescriptor,
    ) -> Result<Vec<T>> {
        indices
            .iter()
            .map(|&index| {
                T::try_from(index).map_err(|_| Error::UnsupportedDType {
                    dtype,
                    backend: "cpu",
                    op,
                })
            })
            .collect()
    }

    let descriptor = KInt::descriptor(&Default::default());
    let builtin_id = descriptor.builtin_id();
    match builtin_id {
        Some(DTypeId::I64) => Ok(CpuBuffer::I64(indices.to_vec())),
        Some(DTypeId::U32) => Ok(CpuBuffer::U32(narrow::<u32>(op, indices, descriptor)?)),
        Some(DTypeId::U8) => Ok(CpuBuffer::U8(narrow::<u8>(op, indices, descriptor)?)),
        _ => Err(Error::UnsupportedDType {
            dtype: descriptor,
            backend: "cpu",
            op,
        }),
    }
}

/// Sum-reduce `storage` over `axis`, *keeping* that axis as size 1
/// (e.g. `[4, 3]` over axis 0 → `[1, 3]`).
pub(crate) fn sum_axis_keepdim(storage: &CpuStorage, axis: usize) -> Result<CpuStorage> {
    let mut out_shape = storage.shape.to_vec();
    out_shape[axis] = 1;
    let total: usize = crate::cpu::stride::validated_numel(&(out_shape));

    macro_rules! reduce_variant {
        ($variant:ident, $to_ty:expr) => {{
            let mut out = vec![Default::default(); total];
            let mut idx = vec![0usize; storage.shape.len()];
            let src_total: usize = crate::cpu::stride::validated_numel(&(storage.shape));
            for _ in 0..src_total {
                let mut out_idx = idx.clone();
                out_idx[axis] = 0;
                let flat_out = flatten_index(&out_idx, &out_shape);
                out[flat_out] += $to_ty(storage.get(&idx));
                increment_index(&mut idx, &storage.shape);
            }
            CpuBuffer::$variant(out)
        }};
    }

    let new_buffer = match &*storage.buffer {
        CpuBuffer::F32(values) => {
            if crate::cpu::stride::is_contiguous(&storage.shape, &storage.strides) {
                let start = storage.offset_elements;
                let src_total = crate::cpu::stride::validated_numel(&storage.shape);
                let dense_slice = &values[start..start + src_total];
                if axis == storage.shape.len().saturating_sub(1) {
                    let row_len = storage.shape[axis];
                    let mut out = vec![0.0f32; total];
                    for (r, chunk) in dense_slice.chunks_exact(row_len).enumerate() {
                        out[r] = crate::simd::vectorize_reduce_sum_f32(chunk);
                    }
                    CpuBuffer::F32(out)
                } else if axis == 0 {
                    let dim_len = storage.shape[0];
                    let inner_len = total;
                    let mut out = vec![0.0f32; inner_len];
                    for r in 0..dim_len {
                        let chunk = &dense_slice[r * inner_len..(r + 1) * inner_len];
                        crate::simd::vectorize_add_into_f32(&mut out, chunk);
                    }
                    CpuBuffer::F32(out)
                } else {
                    reduce_variant!(F32, |v: f64| v as f32)
                }
            } else {
                reduce_variant!(F32, |v: f64| v as f32)
            }
        }
        CpuBuffer::F64(_) => reduce_variant!(F64, |v: f64| v),
        CpuBuffer::U8(_) => reduce_variant!(U8, |v: f64| v as u8),
        CpuBuffer::Bool(_) => reduce_variant!(Bool, |v: f64| v as u8),
        CpuBuffer::U32(_) => reduce_variant!(U32, |v: f64| v as u32),
        CpuBuffer::I64(_) => reduce_variant!(I64, |v: f64| v as i64),
        CpuBuffer::F16(_) => reduce_variant!(F16, |v: f64| half::f16::from_f64(v)),
        CpuBuffer::BF16(_) => reduce_variant!(BF16, |v: f64| half::bf16::from_f64(v)),

        CpuBuffer::Q8_0(_) => {
            return Err(Error::UnsupportedDType {
                dtype: DTypeId::Q8_0.descriptor(),
                backend: "cpu",
                op: "sum_axis_keepdim",
            });
        }
    };

    Ok(CpuStorage::from_contiguous(new_buffer, &out_shape))
}

/// Sum-reduce `storage` over `axis`, *removing* that axis from the shape
/// entirely (e.g. `[4, 3]` over axis 0 → `[3]`).
pub(crate) fn sum_axis_squeeze(storage: &CpuStorage, axis: usize) -> Result<CpuStorage> {
    let reduced = sum_axis_keepdim(storage, axis)?;
    let mut new_shape = reduced.shape.to_vec();
    new_shape.remove(axis);
    // Squeezing a size-1 keepdim result is a pure metadata reshape (no data
    // movement) since the output is already contiguous.
    reduced
        .reshape(&new_shape)
        .map_err(|_| Error::InternalInvariant {
            operation: "sum_axis_squeeze",
            reason: "validated keepdim reduction could not be reshaped",
        })
}

/// Build a contiguous `CpuStorage` of `shape` where every element equals
/// `scalar_value`, matching the dtype variant of `like`. Used by `sum_all` and
/// `mean_all` backward closures to broadcast the incoming scalar gradient back
/// to the full original shape.
pub(super) fn fill_like(
    like: &CpuStorage,
    shape: &[usize],
    scalar_value: f64,
) -> Result<CpuStorage> {
    let total: usize = crate::cpu::stride::validated_numel(shape);
    let new_buffer = match &*like.buffer {
        CpuBuffer::F32(_) => CpuBuffer::F32(vec![scalar_value as f32; total]),
        CpuBuffer::F64(_) => CpuBuffer::F64(vec![scalar_value; total]),
        CpuBuffer::U8(_) => CpuBuffer::U8(vec![scalar_value as u8; total]),
        CpuBuffer::Bool(_) => CpuBuffer::Bool(vec![scalar_value as u8; total]),
        CpuBuffer::U32(_) => CpuBuffer::U32(vec![scalar_value as u32; total]),
        CpuBuffer::I64(_) => CpuBuffer::I64(vec![scalar_value as i64; total]),
        CpuBuffer::F16(_) => CpuBuffer::F16(vec![half::f16::from_f64(scalar_value); total]),
        CpuBuffer::BF16(_) => CpuBuffer::BF16(vec![half::bf16::from_f64(scalar_value); total]),

        CpuBuffer::Q8_0(_) => {
            return Err(Error::UnsupportedDType {
                dtype: DTypeId::Q8_0.descriptor(),
                backend: "cpu",
                op: "reduction gradient fill",
            });
        }
    };
    Ok(CpuStorage::from_contiguous(new_buffer, shape))
}

/// Reduce along `axis`, tracking the WINNING flat-index-into-source at each
/// output position (needed for backward's gradient-routing scatter).
/// Ties: strict `>` (not `>=`) naturally picks first-encountered winner
/// during forward iteration (Pitfall 3 mitigation, T-02-07).
///
/// The extremum keeps the operand's dtype. It used to be written into an
/// `F32` buffer whatever was read, which made this the one reduction here
/// that narrowed: `sum_axis_keepdim` beside it already converts through the
/// operand's own buffer. `log_softmax` opens with `max_keepdim`, so `softmax`
/// and through it `scaled_dot_product_attention` inherited the narrowing and
/// answered in `f32` for every operand dtype.
pub(super) fn max_axis_with_indices(
    storage: &CpuStorage,
    axis: usize,
) -> Result<(CpuStorage, Vec<usize>)> {
    let mut out_shape = storage.shape.to_vec();
    out_shape[axis] = 1;
    let out_total: usize = crate::cpu::stride::validated_numel(&(out_shape));
    let mut best_val = vec![f64::NEG_INFINITY; out_total];
    let mut best_flat_src_idx = vec![0usize; out_total];

    if let CpuBuffer::F32(ref values) = *storage.buffer
        && crate::cpu::stride::is_contiguous(&storage.shape, &storage.strides)
        && axis == storage.shape.len().saturating_sub(1)
    {
        let row_len = storage.shape[axis];
        let start = storage.offset_elements;
        let src_total = crate::cpu::stride::validated_numel(&storage.shape);
        let dense_slice = &values[start..start + src_total];
        for (r, row) in dense_slice.chunks_exact(row_len).enumerate() {
            let max_v = crate::simd::vectorize_reduce_max_f32(row, f32::NEG_INFINITY);
            let local_idx = row
                .iter()
                .position(|&v| v == max_v || (v.is_nan() && max_v.is_nan()))
                .unwrap_or(0);
            best_val[r] = max_v as f64;
            best_flat_src_idx[r] = r * row_len + local_idx;
        }
        let out = CpuStorage::from_contiguous(storage.buffer.from_f64_values(best_val)?, out_shape);
        return Ok((out, best_flat_src_idx));
    }

    let mut idx = vec![0usize; storage.shape.len()];
    let src_total: usize = crate::cpu::stride::validated_numel(&(storage.shape));
    for _ in 0..src_total {
        let mut out_idx = idx.clone();
        out_idx[axis] = 0;
        let flat_out = flatten_index(&out_idx, &out_shape);
        let v = storage.get(&idx);
        if v > best_val[flat_out] {
            best_val[flat_out] = v;
            best_flat_src_idx[flat_out] = flatten_index(&idx, &storage.shape);
        }
        increment_index(&mut idx, &storage.shape);
    }
    let out = CpuStorage::from_contiguous(storage.buffer.from_f64_values(best_val)?, out_shape);
    Ok((out, best_flat_src_idx))
}

/// Mirror of `max_axis_with_indices`, seeded with `f64::INFINITY` and a
/// strict `<` comparison - same first-encountered-winner convention, and the
/// same dtype preservation.
pub(super) fn min_axis_with_indices(
    storage: &CpuStorage,
    axis: usize,
) -> Result<(CpuStorage, Vec<usize>)> {
    let mut out_shape = storage.shape.to_vec();
    out_shape[axis] = 1;
    let out_total: usize = crate::cpu::stride::validated_numel(&(out_shape));
    let mut best_val = vec![f64::INFINITY; out_total];
    let mut best_flat_src_idx = vec![0usize; out_total];

    if let CpuBuffer::F32(ref values) = *storage.buffer
        && crate::cpu::stride::is_contiguous(&storage.shape, &storage.strides)
        && axis == storage.shape.len().saturating_sub(1)
    {
        let row_len = storage.shape[axis];
        let start = storage.offset_elements;
        let src_total = crate::cpu::stride::validated_numel(&storage.shape);
        let dense_slice = &values[start..start + src_total];
        for (r, row) in dense_slice.chunks_exact(row_len).enumerate() {
            let min_v = crate::simd::vectorize_reduce_min_f32(row, f32::INFINITY);
            let local_idx = row
                .iter()
                .position(|&v| v == min_v || (v.is_nan() && min_v.is_nan()))
                .unwrap_or(0);
            best_val[r] = min_v as f64;
            best_flat_src_idx[r] = r * row_len + local_idx;
        }
        let out = CpuStorage::from_contiguous(storage.buffer.from_f64_values(best_val)?, out_shape);
        return Ok((out, best_flat_src_idx));
    }

    let mut idx = vec![0usize; storage.shape.len()];
    let src_total: usize = crate::cpu::stride::validated_numel(&(storage.shape));
    for _ in 0..src_total {
        let mut out_idx = idx.clone();
        out_idx[axis] = 0;
        let flat_out = flatten_index(&out_idx, &out_shape);
        let v = storage.get(&idx);
        if v < best_val[flat_out] {
            best_val[flat_out] = v;
            best_flat_src_idx[flat_out] = flatten_index(&idx, &storage.shape);
        }
        increment_index(&mut idx, &storage.shape);
    }
    let out = CpuStorage::from_contiguous(storage.buffer.from_f64_values(best_val)?, out_shape);
    Ok((out, best_flat_src_idx))
}

/// Backward helper shared by `max_dim`/`min_dim`/`max_keepdim`/`min_keepdim`:
/// build a zero-filled buffer sized to `original_shape`, then scatter
/// `grad_out`'s per-output-position value into ONLY the recorded winning
/// flat index for that position (T-02-08 mitigation - reuses `flatten_index`
/// rather than hand-deriving a new index-resolution formula).
pub(super) fn scatter_axis_grad(
    grad_out: &CpuStorage,
    winning_flat_src_idx: &[usize],
    original_shape: &[usize],
) -> CpuStorage {
    let total: usize = crate::cpu::stride::validated_numel(original_shape);
    let mut vals = vec![0.0f32; total];
    let out_total: usize = crate::cpu::stride::validated_numel(&(grad_out.shape));
    let mut out_idx = vec![0usize; grad_out.shape.len()];
    for flat_out in 0..out_total {
        let g = grad_out.get(&out_idx);
        vals[winning_flat_src_idx[flat_out]] = g as f32;
        increment_index(&mut out_idx, &grad_out.shape);
    }
    CpuStorage::from_contiguous(CpuBuffer::F32(vals), original_shape)
}

// ---------------------------------------------------------------------------
// Concrete reduction kernels
// ---------------------------------------------------------------------------

/// Sum every element of `t` into a single-element scalar storage (shape
/// `[]`). Pushes a `TapeEntry` whose backward broadcasts the incoming
/// scalar gradient uniformly back across `t`'s original shape.
/// A contiguous typed buffer, readable one element at a time as `f64`.
///
/// The whole-tensor reducers walked a logical odometer and read each element
/// through [`CpuStorage::get`]: a stride dot product over a heap-allocated index
/// vector, then a match on the buffer behind an `Arc`, for every element. That
/// is roughly twenty cycles to fetch one number against one to add it, and it is
/// why a 1024-element `sum_all` measured 6.8 ns per element.
///
/// Reading through this instead leaves one small match per element whose
/// discriminant stays in a register, which the optimiser hoists out of the loop.
/// The accumulator type and the traversal order are unchanged in every reducer
/// below, so the dense and odometer paths produce bit-identical results.
/// Accumulating `f32` in `f32` would be faster still and would silently change
/// every reduced value.
pub(super) enum DenseReader<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
    U8(&'a [u8]),
    U32(&'a [u32]),
    I64(&'a [i64]),
    F16(&'a [half::f16]),
    BF16(&'a [half::bf16]),
}

impl DenseReader<'_> {
    /// The element at contiguous position `index`, widened to `f64` exactly as
    /// [`CpuStorage::get`] widens it.
    #[inline]
    pub(super) fn at(&self, index: usize) -> f64 {
        match self {
            Self::F32(values) => f64::from(values[index]),
            Self::F64(values) => values[index],
            Self::U8(values) => f64::from(values[index]),
            Self::U32(values) => f64::from(values[index]),
            Self::I64(values) => values[index] as f64,
            Self::F16(values) => f64::from(values[index].to_f32()),
            Self::BF16(values) => f64::from(values[index].to_f32()),
        }
    }
}

/// A [`DenseReader`] over `t`, or `None` when the general path must run.
///
/// `None` means a non-contiguous view, or a block-quantised buffer whose
/// elements are not addressable individually.
pub(super) fn dense_reader(t: &CpuStorage) -> Option<DenseReader<'_>> {
    if !crate::cpu::stride::is_contiguous(&t.shape, &t.strides) {
        return None;
    }
    let total = crate::cpu::stride::validated_numel(&t.shape);
    let start = t.offset_elements;
    let end = start.checked_add(total)?;
    Some(match &*t.buffer {
        CpuBuffer::F32(values) => DenseReader::F32(values.get(start..end)?),
        CpuBuffer::F64(values) => DenseReader::F64(values.get(start..end)?),
        CpuBuffer::U8(values) => DenseReader::U8(values.get(start..end)?),
        CpuBuffer::U32(values) => DenseReader::U32(values.get(start..end)?),
        CpuBuffer::I64(values) => DenseReader::I64(values.get(start..end)?),
        CpuBuffer::F16(values) => DenseReader::F16(values.get(start..end)?),
        CpuBuffer::BF16(values) => DenseReader::BF16(values.get(start..end)?),
        _ => return None,
    })
}

/// Fold every element of `t` in traversal order, taking the dense path when the
/// storage allows it.
///
/// The fold closure is shared by both paths, which is what makes them agree by
/// construction rather than by inspection. `f` receives the accumulator, the
/// flat position in traversal order, and the widened value; for a contiguous
/// tensor that position is also the memory index, which is what the `max_all`
/// and `min_all` gradients record.
pub(super) fn fold_all_f64<A>(t: &CpuStorage, init: A, mut f: impl FnMut(A, usize, f64) -> A) -> A {
    let total = crate::cpu::stride::validated_numel(&t.shape);
    let mut accumulator = init;
    if let Some(reader) = dense_reader(t) {
        for index in 0..total {
            accumulator = f(accumulator, index, reader.at(index));
        }
        return accumulator;
    }
    let mut idx = vec![0usize; t.shape.len()];
    for index in 0..total {
        accumulator = f(accumulator, index, t.get(&idx));
        if !t.shape.is_empty() {
            increment_index(&mut idx, &t.shape);
        }
    }
    accumulator
}

/// The `f64` sum of every element of `t`.
pub(super) fn total_sum_f64(t: &CpuStorage) -> f64 {
    fold_all_f64(t, 0f64, |sum, _, value| sum + value)
}
