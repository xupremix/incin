use super::*;

pub(crate) fn canonical_softmax<D: Device>(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    let log_values = log_softmax::<D, f32>(t, dim)?;
    canonical_exp(&log_values)
}

// ---------------------------------------------------------------------------
// Shared log-softmax kernel (D-02)
// ---------------------------------------------------------------------------

/// `log_softmax(x, dim) = (x - max) - log(sum_keepdim(exp(x - max), dim))`
///
/// Matches `candle-nn-0.9.1/src/ops.rs` lines 31-38 exactly:
/// ```text
/// let max = xs.max_keepdim(d)?;
/// let diff = xs.broadcast_sub(&max)?;
/// let sum_exp = diff.exp()?.sum_keepdim(d)?;
/// let log_sm = diff.broadcast_sub(&sum_exp.log()?)?
/// ```
///
/// Composed entirely from already-tape-tracked primitives - zero new backward
/// code is written here; the composed tape entries from `max_keepdim` / `sub`
/// / `exp` / `sum_keepdim` / `log` / `sub` already implement the correct
/// backward chain automatically (Plan 04-01 D-02 rationale).
///
/// Called by both `::softmax` (as `exp(log_softmax(x, dim))`) and
/// `cross_entropy_loss` (as `-log_softmax(x, 1)[target]`), so the
/// numerically-stable kernel is shared rather than duplicated.
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn log_softmax<D: incin_core::tensor::device::Device, K: DType>(
    t: &CpuStorage,
    dim: usize,
) -> Result<CpuStorage> {
    let max = crate::cpu::ops::reduce::max_keepdim(t, dim)?;
    let diff = sub_storage(t, &max)?;
    let exp_diff = canonical_exp(&diff)?;
    let sum_exp = crate::cpu::ops::reduce::sum_keepdim(&exp_diff, dim)?;
    let log_sum_exp = canonical_log(&sum_exp)?;
    sub_storage(&diff, &log_sum_exp)
}
