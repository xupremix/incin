use super::*;

pub(crate) fn lerp_storage(
    start: &CpuStorage,
    end: &CpuStorage,
    weight: f64,
) -> Result<CpuStorage> {
    let total = crate::cpu::stride::checked_numel(&start.shape)?;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; start.shape.len()];
    for _ in 0..total {
        let start_value = start.get(&idx);
        let end_value = end.get(&idx);
        out.push(start_value + weight * (end_value - start_value));
        if !start.shape.is_empty() {
            crate::cpu::storage::increment_index(&mut idx, &start.shape);
        }
    }
    Ok(CpuStorage::from_contiguous(
        start.buffer.from_f64_values(out)?,
        &start.shape,
    ))
}

pub(crate) fn addmm_storage(
    mat: &CpuStorage,
    mat1: &CpuStorage,
    mat2: &CpuStorage,
    beta: f64,
    alpha: f64,
) -> Result<CpuStorage> {
    let product = matmul_storage(mat1, mat2)?;
    let scaled_product = canonical_mul_scalar(&product, alpha)?;
    let scaled_mat = canonical_mul_scalar(mat, beta)?;
    add_storage(&scaled_mat, &scaled_product)
}

/// Plain or batched matrix multiplication, chosen by operand rank.
pub(crate) fn matmul_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    if lhs.shape.len() == 2 && rhs.shape.len() == 2 {
        matmul_impl(lhs, rhs)
    } else {
        batched_matmul_impl(lhs, rhs)
    }
}

pub(crate) fn scaled_dot_product_attention_storage<D: Device>(
    q: &CpuStorage,
    k: &CpuStorage,
    v: &CpuStorage,
    mask: Option<&CpuStorage>,
    scale: Option<f64>,
) -> Result<CpuStorage> {
    let k_t = if k.shape.len() >= 2 {
        transpose_storage(k, k.shape.len() - 2, k.shape.len() - 1)?
    } else {
        k.clone()
    };
    let scores = matmul_storage(q, &k_t)?;
    let d_k = *q.shape.last().unwrap_or(&1) as f64;
    let scaled_scores = canonical_mul_scalar(&scores, scale.unwrap_or_else(|| 1.0 / d_k.sqrt()))?;
    let masked_scores = match mask {
        Some(mask) => add_storage(&scaled_scores, mask)?,
        None => scaled_scores,
    };
    let attention = canonical_softmax::<D>(&masked_scores, scores.shape.len() - 1)?;
    matmul_storage(&attention, v)
}
