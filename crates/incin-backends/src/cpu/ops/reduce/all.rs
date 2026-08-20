use super::*;

pub(crate) fn sum_all(t: &CpuStorage) -> Result<CpuStorage> {
    let sum = if let Some(DenseReader::F32(values)) = dense_reader(t) {
        crate::simd::vectorize_reduce_sum_f32(values) as f64
    } else {
        total_sum_f64(t)
    };
    let out = CpuStorage::from_contiguous(CpuBuffer::F32(vec![sum as f32]), vec![]);

    let original_shape = t.shape.to_vec();
    let t_clone = t.clone(); // dtype reference for fill_like
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            // grad_out is a scalar []; broadcast it to every element of
            // the original shape (the backward of sum is "distribute
            // everywhere").
            let scalar_grad = grad_out.get(&vec![0usize; grad_out.shape.len()]);
            Ok(vec![fill_like(&t_clone, &original_shape, scalar_grad)?])
        }),
    });

    Ok(out)
}

/// Mean of every element of `t`. Backward scales the incoming scalar
/// gradient by `1/n` before broadcasting back to the original shape.
pub(crate) fn mean_all(t: &CpuStorage) -> Result<CpuStorage> {
    let total: usize = crate::cpu::stride::validated_numel(&(t.shape));
    let sum = if let Some(DenseReader::F32(values)) = dense_reader(t) {
        crate::simd::vectorize_reduce_sum_f32(values) as f64
    } else {
        total_sum_f64(t)
    };
    let mean = if total > 0 { sum / total as f64 } else { 0.0 };
    let out = CpuStorage::from_contiguous(CpuBuffer::F32(vec![mean as f32]), vec![]);

    let original_shape = t.shape.to_vec();
    let t_clone = t.clone();
    let n = total as f64;
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let scalar_grad = grad_out.get(&vec![0usize; grad_out.shape.len()]);
            // d(mean)/d(x_i) = 1/n for each element.
            let scaled = if n > 0.0 { scalar_grad / n } else { 0.0 };
            Ok(vec![fill_like(&t_clone, &original_shape, scaled)?])
        }),
    });

    Ok(out)
}

/// Maximum over every element of `t`, as a scalar (shape `[]`).
/// Independent flat iteration (mirrors `sum_all`'s structure, NOT a
/// reshape-then-axis-reduce composition, per RESEARCH.md Open Question 2).
/// Backward scatters the incoming scalar gradient to ONLY the single
/// global winning flat index, zero everywhere else.
pub(crate) fn max_all(t: &CpuStorage) -> Result<CpuStorage> {
    // Strict `>`, so the first of equal maxima wins and the recorded gradient
    // position is the same one the odometer would have chosen.
    let (best_val, best_flat_idx) = if let Some(DenseReader::F32(values)) = dense_reader(t) {
        let max_v = crate::simd::vectorize_reduce_max_f32(values, f32::NEG_INFINITY);
        let idx = values
            .iter()
            .position(|&v| v == max_v || (v.is_nan() && max_v.is_nan()))
            .unwrap_or(0);
        (max_v as f64, idx)
    } else {
        fold_all_f64(
            t,
            (f64::NEG_INFINITY, 0usize),
            |(best, best_index), index, value| {
                if value > best {
                    (value, index)
                } else {
                    (best, best_index)
                }
            },
        )
    };
    let out = CpuStorage::from_contiguous(CpuBuffer::F32(vec![best_val as f32]), vec![]);

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let scalar_grad = grad_out.get(&vec![0usize; grad_out.shape.len()]);
            let total: usize = crate::cpu::stride::validated_numel(&(original_shape));
            let mut vals = vec![0.0f32; total];
            vals[best_flat_idx] = scalar_grad as f32;
            Ok(vec![CpuStorage::from_contiguous(
                CpuBuffer::F32(vals),
                &original_shape,
            )])
        }),
    });

    Ok(out)
}

/// Minimum over every element of `t`, as a scalar (shape `[]`). Mirror of
/// `max_all` with strict `<` comparison.
pub(crate) fn min_all(t: &CpuStorage) -> Result<CpuStorage> {
    let (best_val, best_flat_idx) = if let Some(DenseReader::F32(values)) = dense_reader(t) {
        let min_v = crate::simd::vectorize_reduce_min_f32(values, f32::INFINITY);
        let idx = values
            .iter()
            .position(|&v| v == min_v || (v.is_nan() && min_v.is_nan()))
            .unwrap_or(0);
        (min_v as f64, idx)
    } else {
        fold_all_f64(
            t,
            (f64::INFINITY, 0usize),
            |(best, best_index), index, value| {
                if value < best {
                    (value, index)
                } else {
                    (best, best_index)
                }
            },
        )
    };
    let out = CpuStorage::from_contiguous(CpuBuffer::F32(vec![best_val as f32]), vec![]);

    let original_shape = t.shape.to_vec();
    let (t_id, out_id) = (t.id, out.id);
    tape::push_with(|| TapeEntry {
        output_id: out_id,
        input_ids: vec![t_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            let scalar_grad = grad_out.get(&vec![0usize; grad_out.shape.len()]);
            let total: usize = crate::cpu::stride::validated_numel(&(original_shape));
            let mut vals = vec![0.0f32; total];
            vals[best_flat_idx] = scalar_grad as f32;
            Ok(vec![CpuStorage::from_contiguous(
                CpuBuffer::F32(vals),
                &original_shape,
            )])
        }),
    });

    Ok(out)
}

pub(crate) fn prod_all(t: &CpuStorage) -> Result<CpuStorage> {
    let prod = fold_all_f64(t, 1.0f64, |product, _, value| product * value);
    let buffer = t.buffer.from_f64_values(vec![prod])?;
    Ok(CpuStorage::from_contiguous(buffer, vec![]))
}
