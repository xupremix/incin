//! Shared backward-composition helpers (plain, non-tape-tracked — operate on
//! already-materialized `CpuStorage` values inside the hand-composed
//! backward closures in `conv1d`/`conv2d`/`conv_transpose2d`, mirroring the
//! forward's own per-group narrow/concat convention).

use incin_core::error::Result;
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;

use crate::cpu::storage::{CpuBuffer, CpuStorage, increment_index};

/// Sum a `[B, M, N]` storage over its leading batch axis, producing `[M, N]`.
/// Used by both `conv1d_impl`/`conv2d_windowed_impl`'s backward to reduce
/// `grad_weight`'s per-batch matmul contributions down to weight's own
/// (batch-independent) shape.
pub(super) fn sum_batch_dim(t: &CpuStorage) -> Result<CpuStorage> {
    let (b, m, n) = (t.shape[0], t.shape[1], t.shape[2]);
    let out_total = ShapeBuf::from_slice(&[m, n]).checked_numel(OperationKind::Reduction)?;
    let mut out = vec![0.0f32; out_total];
    for bi in 0..b {
        for mi in 0..m {
            for ni in 0..n {
                out[mi * n + ni] += t.get(&[bi, mi, ni]) as f32;
            }
        }
    }
    CpuStorage::try_from_contiguous(CpuBuffer::F32(out), vec![m, n])
}

/// Plain (non-tape-tracked) concat along axis 1 of a list of same-rank
/// storages, used to re-assemble `grad_input`'s per-group channel slices
/// inside the hand-composed backward closures above.
pub(super) fn concat_along_dim1(parts: &[CpuStorage]) -> Result<CpuStorage> {
    concat_along_dim(parts, 1)
}

/// Plain (non-tape-tracked) concat along axis 0, used to re-assemble
/// `grad_weight`'s per-group output-channel slices.
pub(super) fn concat_along_dim0(parts: &[CpuStorage]) -> Result<CpuStorage> {
    concat_along_dim(parts, 0)
}

/// `concat_along_dim`.
fn concat_along_dim(parts: &[CpuStorage], dim: usize) -> Result<CpuStorage> {
    let rank = parts[0].shape.len();
    let mut out_shape = parts[0].shape.to_vec();
    out_shape[dim] = parts.iter().try_fold(0usize, |total, part| {
        total.checked_add(part.shape[dim]).ok_or(
            incin_core::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Concat,
                expression: "sum of convolution gradient group dimensions",
            },
        )
    })?;
    let out_strides = crate::cpu::stride::contiguous_strides(&out_shape.to_vec());
    let total = crate::cpu::stride::checked_numel(&out_shape)?;
    let mut out = vec![0.0f32; total];

    let mut offset = 0usize;
    for part in parts {
        let value_count: usize = crate::cpu::stride::validated_numel(&(part.shape));
        let mut multi_idx = vec![0usize; rank];
        for _ in 0..value_count {
            let mut flat_dest = 0usize;
            for (axis, &i) in multi_idx.iter().enumerate() {
                let dest_i = if axis == dim { i + offset } else { i };
                flat_dest += dest_i * out_strides[axis];
            }
            out[flat_dest] = part.get(&multi_idx) as f32;
            increment_index(&mut multi_idx, &part.shape);
        }
        offset = offset.checked_add(part.shape[dim]).ok_or(
            incin_core::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Concat,
                expression: "cumulative convolution gradient group offset",
            },
        )?;
    }

    Ok(CpuStorage::from_contiguous(CpuBuffer::F32(out), &out_shape))
}
