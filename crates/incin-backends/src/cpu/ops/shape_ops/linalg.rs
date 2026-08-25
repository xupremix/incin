use super::*;

pub(crate) fn lerp_storage(
    start: &CpuStorage,
    end: &CpuStorage,
    weight: f64,
) -> Result<CpuStorage> {
    // Composed from tape-tracked primitives rather than a raw walk so the
    // BinaryBroadcast gradient the catalog declares actually arrives:
    // d/dstart = 1 - weight and d/dend = weight, per element, both through
    // the wrappers' own unbroadcast handling.
    let diff = crate::cpu::ops::elementwise::sub_storage(end, start)?;
    let scaled = crate::cpu::ops::elementwise::canonical_mul_scalar(&diff, weight)?;
    crate::cpu::ops::elementwise::add_storage(start, &scaled)
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
