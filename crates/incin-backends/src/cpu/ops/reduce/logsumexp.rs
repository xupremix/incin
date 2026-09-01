use super::*;

use crate::cpu::ops::elementwise::{add_storage, canonical_exp, canonical_log, sub_storage};
use crate::cpu::ops::shape_ops::squeeze_storage;

/// `logsumexp(x, dim) = max + log(sum_keepdim(exp(x - max), dim))`
///
/// The shift by the axis maximum is the whole point. Written directly,
/// `log(sum(exp(x)))` overflows to infinity as soon as any entry exceeds about
/// 88 in f32, and underflows to negative infinity when every entry is far
/// enough below zero; both are ordinary magnitudes for a router logit.
/// Subtracting the maximum first bounds every exponential to `(0, 1]`, so the
/// sum is at least one and at most the axis length, and adding the maximum back
/// restores the value exactly.
///
/// Composed from `max_keepdim`, `sub`, `exp`, `sum_keepdim`, `log` and `add`,
/// each of which already pushes its own tape entry, so the backward is the
/// replay over those entries rather than new hand-derived math. This is the
/// same argument, and the same set of primitives, that
/// [`crate::cpu::ops::elementwise::log_softmax`] rests on; the two differ only
/// in which of the two subtractions is kept.
pub(crate) fn logsumexp_keepdim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "logsumexp_keepdim",
            expected: t.shape.to_vec(),
            got: vec![dim],
            msg: format!(
                "logsumexp_keepdim: axis {dim} out of range for shape {:?}",
                t.shape
            ),
        });
    }
    let max = max_keepdim(t, dim)?;
    let shifted = sub_storage(t, &max)?;
    let sum_exp = sum_keepdim(&canonical_exp(&shifted)?, dim)?;
    add_storage(&max, &canonical_log(&sum_exp)?)
}

/// [`logsumexp_keepdim`] with the reduced axis removed.
///
/// The squeeze goes through `squeeze_storage` rather than a bare reshape
/// because a bare reshape records nothing: the composition above owes its
/// backward entirely to the tape entries its steps push, and a final step that
/// pushes none would leave the chain ending at the keepdim result while the
/// gradient arrives with the squeezed shape.
pub(crate) fn logsumexp_dim(t: &CpuStorage, dim: usize) -> Result<CpuStorage> {
    if dim >= t.shape.len() {
        return Err(Error::ShapeMismatch {
            op: "logsumexp_dim",
            expected: t.shape.to_vec(),
            got: vec![dim],
            msg: format!(
                "logsumexp_dim: axis {dim} out of range for shape {:?}",
                t.shape
            ),
        });
    }
    squeeze_storage(&logsumexp_keepdim(t, dim)?, dim)
}
