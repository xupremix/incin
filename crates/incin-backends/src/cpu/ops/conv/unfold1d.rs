//! `im2col_1d`/`col2im_1d`: the materializing gather/scatter-ADD pair
//! `conv1d_impl` unfolds through.

use incin_core::error::Result;
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;

use crate::cpu::storage::{CpuBuffer, CpuStorage};

use super::helpers::out_size;

// ---------------------------------------------------------------------------
// im2col_1d / col2im_1d
// ---------------------------------------------------------------------------

/// Materializing gather loop producing a `[B, L_out, Cin*K]` column matrix
/// from a `[B, Cin, L]` input. For every gathered element whose computed
/// source position falls outside `[0, L)`, substitutes `0.0` (Pitfall 2).
pub(super) fn im2col_1d(
    input: &CpuStorage,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<CpuStorage> {
    let (b, cin, len) = (input.shape[0], input.shape[1], input.shape[2]);
    let l_out = out_size(len, kernel_size, stride, padding, dilation)?;

    let columns = ShapeBuf::from_slice(&[cin, kernel_size]).checked_numel(OperationKind::Conv1d)?;
    let capacity =
        ShapeBuf::from_slice(&[b, l_out, columns]).checked_numel(OperationKind::Conv1d)?;
    let mut out = Vec::with_capacity(capacity);
    for bi in 0..b {
        for oi in 0..l_out {
            for ci in 0..cin {
                for ki in 0..kernel_size {
                    let src = oi * stride + ki * dilation;
                    let val = if src >= padding && src - padding < len {
                        input.get(&[bi, ci, src - padding])
                    } else {
                        0.0
                    };
                    out.push(val as f32);
                }
            }
        }
    }

    CpuStorage::try_from_contiguous(CpuBuffer::F32(out), vec![b, l_out, columns])
}

/// Exact inverse of `im2col_1d`: scatter-ADD (never overwrite) each
/// contribution of a `[B, L_out, Cin*K]`-shaped gradient back into a
/// zero-initialized `[B, Cin, L]` buffer, skipping any destination position
/// that fell in the padded region during the forward unfold.
pub(super) fn col2im_1d(
    cols_grad: &CpuStorage,
    input_shape: &[usize],
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<CpuStorage> {
    let (b, cin, len) = (input_shape[0], input_shape[1], input_shape[2]);
    let l_out = cols_grad.shape[1];

    let out_total = ShapeBuf::from_slice(input_shape).checked_numel(OperationKind::Conv1d)?;
    let mut out = vec![0.0f32; out_total];
    let out_strides = crate::cpu::stride::contiguous_strides(input_shape);

    for bi in 0..b {
        for oi in 0..l_out {
            for ci in 0..cin {
                for ki in 0..kernel_size {
                    let src = oi * stride + ki * dilation;
                    if src >= padding && src - padding < len {
                        let dst_l = src - padding;
                        let flat =
                            bi * out_strides[0] + ci * out_strides[1] + dst_l * out_strides[2];
                        let g = cols_grad.get(&[bi, oi, ci * kernel_size + ki]);
                        out[flat] += g as f32;
                    }
                }
            }
        }
    }

    CpuStorage::try_from_contiguous(CpuBuffer::F32(out), input_shape)
}
