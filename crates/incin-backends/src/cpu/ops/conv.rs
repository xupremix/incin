//! `conv1d`/`conv2d` for CPU storage via im2col (window-unfold into
//! a column matrix) + `ops::matmul::batched_matmul_impl` for the actual
//! multiply-accumulate (D-01).
//!
//! `im2col_1d`/`im2col_2d` are materializing gather loops (NOT a
//! `CpuStorage` view op, per Pitfall 2): every gathered element whose
//! computed source position falls outside the (unpadded) input range is
//! substituted with `0.0` rather than erroring or reading adjacent unrelated
//! buffer memory. `col2im_1d`/`col2im_2d` are the exact inverse scatter-ADD
//! (`+=`, never `=`) fold, so overlapping output windows (stride <
//! kernel_size) correctly accumulate their gradient contributions (Pitfall 5's
//! discipline, applied to conv's own backward).
//!
//! `groups` support (including the depthwise `groups == Cin` degenerate case)
//! is implemented as a single generic `0..groups` loop — narrow the input's
//! channel axis and the weight's output-channel axis into per-group slices,
//! `im2col` + `batched_matmul_impl` + concat — with NO special-casing branch
//! for any particular `groups` value (Pitfall 7).
//!
//! `conv1d_impl`/`conv2d_impl` each push exactly ONE top-level `TapeEntry`:
//! the im2col unfold/col2im fold steps are plain (non-tape-tracked) helper
//! functions operating on already-materialized `CpuStorage` values, so
//! their OWN backward is hand-composed here (reusing `batched_matmul_impl`'s
//! already-gradcheck-verified backward only for the INTERNAL
//! multiply-accumulate step, per RESEARCH.md Pattern 3). Bias, when present,
//! is broadcast-added via the canonical storage helper AFTER
//! the hand-composed conv math, so `grad_bias` falls out of that op's own
//! existing backward + `unbroadcast` for free — it is never hand-derived
//! inside `conv1d_impl`/`conv2d_impl`'s own closure.
//!
//! `conv_transpose2d_impl` (Plan 04-07, RESEARCH.md Pattern 4) reuses
//! `col2im_2d` VERBATIM as its own forward fold subroutine — transposed
//! convolution's forward pass is exactly `conv2d`'s own backward-data
//! (grad-w.r.t.-input) formula applied to `input` directly instead of to a
//! gradient. Its own backward, symmetrically, reuses `im2col_2d` +
//! `batched_matmul_impl` (i.e. `conv2d`'s FORWARD formula) to recover
//! `grad_input`. `output_padding` is handled as a separate final
//! allocate-larger-then-copy-into-leading-sub-region step (via
//! `scatter_into_zeros`), never folded into `padding`'s own symmetric
//! offset arithmetic (Pitfall 4). Only `groups == 1` is supported, matching
//! `CandleBackend::conv_transpose2d`'s own confirmed effective behavior.

use incin_core::prelude::Error;
use incin_core::prelude::{BackwardError, OperationKind, ShapeBuf, ShapeError};
use incin_core::prelude::{DType, Result};

use crate::cpu::ops::elementwise::add_storage;
use crate::cpu::ops::matmul::{batched_matmul_impl, transpose_last2};
use crate::cpu::ops::shape_ops::concat_storage;
use crate::cpu::storage::{CpuBuffer, CpuStorage, increment_index, scatter_into_zeros};

use crate::cpu::tape::{self, TapeEntry};

// ---------------------------------------------------------------------------
// Shared output-size arithmetic (T-04-11: saturating_sub, never raw subtraction)
// ---------------------------------------------------------------------------

/// `L_out = (L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`,
/// using `saturating_sub` throughout (matching RESEARCH.md's exact formula)
/// so a pathological small-input/large-kernel combination produces `0`
/// (an empty output) rather than panicking on integer underflow.
fn out_size(
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    if kernel_size == 0 || stride == 0 || dilation == 0 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Conv2d,
            parameter: "kernel, stride, and dilation must be nonzero",
            value: 0,
        }
        .into());
    }
    let padded = padding
        .checked_mul(2)
        .and_then(|twice| len.checked_add(twice))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Conv2d,
            expression: "convolution padded input dimension",
        })?;
    let effective_kernel = dilation
        .checked_mul(kernel_size - 1)
        .and_then(|span| span.checked_add(1))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Conv2d,
            expression: "convolution effective kernel",
        })?;
    Ok(padded.saturating_sub(effective_kernel) / stride + 1)
}

/// The "natural" (no `output_padding`) `conv_transpose2d` output size:
/// `(len - 1) * stride - 2*padding + dilation*(kernel_size-1) + 1`, i.e.
/// `conv2d`'s own forward-shape formula (`out_size` above) inverted. Uses
/// `saturating_sub` throughout (T-04-11) so a pathological small-input
/// combination underflows to `0` rather than panicking. `output_padding` is
/// deliberately NOT part of this formula (Pitfall 4) — it is applied as a
/// separate final allocate-larger step by the caller.
fn natural_transpose_out_size(
    len: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    if kernel_size == 0 || stride == 0 || dilation == 0 {
        return Err(ShapeError::InvalidParameter {
            operation: OperationKind::Conv2d,
            parameter: "kernel, stride, and dilation must be nonzero",
            value: 0,
        }
        .into());
    }
    let unpadded = len
        .saturating_sub(1)
        .checked_mul(stride)
        .and_then(|span| {
            dilation
                .checked_mul(kernel_size - 1)
                .and_then(|kernel| span.checked_add(kernel))
        })
        .and_then(|span| span.checked_add(1))
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Conv2d,
            expression: "transposed-convolution output dimension",
        })?;
    let twice_padding = padding
        .checked_mul(2)
        .ok_or(ShapeError::ArithmeticOverflow {
            operation: OperationKind::Conv2d,
            expression: "transposed-convolution padding",
        })?;
    Ok(unpadded.saturating_sub(twice_padding))
}

/// Validate that `groups` evenly divides both `cin`/`cout`, returning
/// `Error::ShapeMismatch` (never panicking on an integer-division remainder)
/// otherwise (T-04-11).
fn validate_groups(op: &'static str, cin: usize, cout: usize, groups: usize) -> Result<()> {
    if groups == 0 || !cin.is_multiple_of(groups) || !cout.is_multiple_of(groups) {
        return Err(Error::ShapeMismatch {
            op,
            expected: vec![groups],
            got: vec![cin, cout],
            msg: format!("{op}: groups={groups} must evenly divide both Cin={cin} and Cout={cout}"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// im2col_1d / col2im_1d
// ---------------------------------------------------------------------------

/// Materializing gather loop producing a `[B, L_out, Cin*K]` column matrix
/// from a `[B, Cin, L]` input. For every gathered element whose computed
/// source position falls outside `[0, L)`, substitutes `0.0` (Pitfall 2).
fn im2col_1d(
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
fn col2im_1d(
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

    CpuStorage::try_from_contiguous(CpuBuffer::F32(out), input_shape.to_vec())
}

// ---------------------------------------------------------------------------
// im2col_2d / col2im_2d
// ---------------------------------------------------------------------------

/// A convolution window, stated once per spatial axis.
///
/// The historical module-family contract takes one extent for both axes, while the descriptor
/// that routes to it carries one per axis. An anisotropic window was therefore
/// refused outright rather than applying the first axis' extent to both, which
/// was honest but left a real request unanswerable. Carrying the pair this far
/// down is what makes it answerable, and the isotropic case is the degenerate
/// one rather than a separate path.
#[derive(Clone, Copy)]
pub(crate) struct Window2d {
    /// Row and column stride.
    pub(crate) stride: [usize; 2],
    /// Row and column zero padding, applied to both ends of each axis.
    pub(crate) padding: [usize; 2],
    /// Row and column spacing between kernel taps.
    pub(crate) dilation: [usize; 2],
}

impl Window2d {
    /// The same extent on both axes, which is all the legacy signature can say.
    pub(crate) fn isotropic(stride: usize, padding: usize, dilation: usize) -> Self {
        Self {
            stride: [stride; 2],
            padding: [padding; 2],
            dilation: [dilation; 2],
        }
    }
}

/// 2D generalization of `im2col_1d`: gathers a `[B, Cin, H, W]` input into a
/// `[B, H_out*W_out, Cin*Kh*Kw]` column matrix.
fn im2col_2d(
    input: &CpuStorage,
    kernel_h: usize,
    kernel_w: usize,
    window: Window2d,
) -> Result<CpuStorage> {
    let (b, cin, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let h_out = out_size(
        h,
        kernel_h,
        window.stride[0],
        window.padding[0],
        window.dilation[0],
    )?;
    let w_out = out_size(
        w,
        kernel_w,
        window.stride[1],
        window.padding[1],
        window.dilation[1],
    )?;

    let spatial = ShapeBuf::from_slice(&[h_out, w_out]).checked_numel(OperationKind::Conv2d)?;
    let columns =
        ShapeBuf::from_slice(&[cin, kernel_h, kernel_w]).checked_numel(OperationKind::Conv2d)?;
    let capacity =
        ShapeBuf::from_slice(&[b, spatial, columns]).checked_numel(OperationKind::Conv2d)?;
    let mut out = Vec::with_capacity(capacity);
    for bi in 0..b {
        for oh in 0..h_out {
            for ow in 0..w_out {
                for ci in 0..cin {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let src_h = oh * window.stride[0] + kh * window.dilation[0];
                            let src_w = ow * window.stride[1] + kw * window.dilation[1];
                            let val = if src_h >= window.padding[0]
                                && src_h - window.padding[0] < h
                                && src_w >= window.padding[1]
                                && src_w - window.padding[1] < w
                            {
                                input.get(&[
                                    bi,
                                    ci,
                                    src_h - window.padding[0],
                                    src_w - window.padding[1],
                                ])
                            } else {
                                0.0
                            };
                            out.push(val as f32);
                        }
                    }
                }
            }
        }
    }

    CpuStorage::try_from_contiguous(CpuBuffer::F32(out), vec![b, spatial, columns])
}

/// Exact inverse of `im2col_2d`: scatter-ADD each contribution of a
/// `[B, H_out*W_out, Cin*Kh*Kw]`-shaped gradient back into a
/// zero-initialized `[B, Cin, H, W]` buffer, skipping padded-region
/// destinations.
fn col2im_2d(
    cols_grad: &CpuStorage,
    input_shape: &[usize],
    kernel_h: usize,
    kernel_w: usize,
    window: Window2d,
) -> Result<CpuStorage> {
    let (b, cin, h, w) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );
    let h_out = out_size(
        h,
        kernel_h,
        window.stride[0],
        window.padding[0],
        window.dilation[0],
    )?;
    let w_out = out_size(
        w,
        kernel_w,
        window.stride[1],
        window.padding[1],
        window.dilation[1],
    )?;

    let out_total = ShapeBuf::from_slice(input_shape).checked_numel(OperationKind::Conv2d)?;
    let mut out = vec![0.0f32; out_total];
    let out_strides = crate::cpu::stride::contiguous_strides(input_shape);

    for bi in 0..b {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let o_flat = oh * w_out + ow;
                for ci in 0..cin {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let src_h = oh * window.stride[0] + kh * window.dilation[0];
                            let src_w = ow * window.stride[1] + kw * window.dilation[1];
                            if src_h >= window.padding[0]
                                && src_h - window.padding[0] < h
                                && src_w >= window.padding[1]
                                && src_w - window.padding[1] < w
                            {
                                let dst_h = src_h - window.padding[0];
                                let dst_w = src_w - window.padding[1];
                                let flat = bi * out_strides[0]
                                    + ci * out_strides[1]
                                    + dst_h * out_strides[2]
                                    + dst_w * out_strides[3];
                                let col_idx = ci * kernel_h * kernel_w + kh * kernel_w + kw;
                                let g = cols_grad.get(&[bi, o_flat, col_idx]);
                                out[flat] += g as f32;
                            }
                        }
                    }
                }
            }
        }
    }

    CpuStorage::try_from_contiguous(CpuBuffer::F32(out), input_shape.to_vec())
}

// ---------------------------------------------------------------------------
// conv1d_impl
// ---------------------------------------------------------------------------

/// Canonical conv1d implementation: im2col + per-group
/// `batched_matmul_impl` + concat forward, hand-composed backward for
/// grad_input (col2im fold) and grad_weight (per-group matmul), with bias
/// broadcast-added via the canonical storage helper (so
/// `grad_bias` is free via composition, per this file's module doc).
pub(crate) fn conv1d_impl<D: incin_core::prelude::Device, K: DType>(
    input: &CpuStorage,
    weight: &CpuStorage,
    bias: Option<&CpuStorage>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<CpuStorage> {
    let (b, cin, len) = (input.shape[0], input.shape[1], input.shape[2]);
    let (cout, cin_g, kernel_size) = (weight.shape[0], weight.shape[1], weight.shape[2]);
    validate_groups("conv1d", cin, cout, groups)?;
    if cin / groups != cin_g {
        return Err(Error::ShapeMismatch {
            op: "conv1d",
            expected: vec![cin / groups],
            got: vec![cin_g],
            msg: format!(
                "conv1d: weight's Cin/groups ({cin_g}) does not match input Cin/groups ({})",
                cin / groups
            ),
        });
    }
    let cout_g = cout / groups;
    let l_out = out_size(len, kernel_size, stride, padding, dilation)?;
    let input_columns =
        ShapeBuf::from_slice(&[cin_g, kernel_size]).checked_numel(OperationKind::Conv1d)?;

    let mut group_outputs: Vec<CpuStorage> = Vec::with_capacity(groups);
    for g in 0..groups {
        let input_g = input.narrow(1, g * cin_g, cin_g)?;
        let weight_g = weight.narrow(0, g * cout_g, cout_g)?;
        let cols = im2col_1d(&input_g, kernel_size, stride, padding, dilation)?;
        let weight_mat = weight_g.reshape(&[cout_g, input_columns])?;
        let out_g = batched_matmul_impl(&cols, &transpose_last2(&weight_mat))?;
        group_outputs.push(out_g);
    }
    let refs: Vec<&CpuStorage> = group_outputs.iter().collect();
    let matmul_out = if groups == 1 {
        group_outputs[0].clone()
    } else {
        concat_storage(&refs, 2)?
    };
    // matmul_out: [B, L_out, Cout] -> canonical [B, Cout, L_out]
    let conv_out = matmul_out.transpose(1, 2)?.reshape(&[b, cout, l_out])?;

    let (input_capture, weight_capture) = (input.clone(), weight.clone());
    let (input_id, weight_id, out_id) = (input.id, weight.id, conv_out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![input_id, weight_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            // grad_out: [B, Cout, L_out] -> [B, L_out, Cout]
            let grad_out_t = grad_out.transpose(1, 2)?;

            let mut grad_input_groups: Vec<CpuStorage> = Vec::with_capacity(groups);
            let mut grad_weight_groups: Vec<CpuStorage> = Vec::with_capacity(groups);
            for g in 0..groups {
                let input_g = input_capture.narrow(1, g * cin_g, cin_g)?;
                let weight_g = weight_capture.narrow(0, g * cout_g, cout_g)?;
                let grad_out_g = grad_out_t.narrow(2, g * cout_g, cout_g)?;

                let weight_mat = weight_g.reshape(&[cout_g, input_columns])?;

                // grad_cols = grad_out_g @ weight_mat : [B, L_out, Cout_g] @ [Cout_g, Cin_g*K]
                let grad_cols = batched_matmul_impl(&grad_out_g, &weight_mat)?;
                let grad_input_g = col2im_1d(
                    &grad_cols,
                    &[b, cin_g, len],
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                )?;
                grad_input_groups.push(grad_input_g);

                // grad_weight_mat = grad_out_g^T @ cols : [Cout_g, B*L_out] view via batched matmul
                let cols = im2col_1d(&input_g, kernel_size, stride, padding, dilation)?;
                let grad_weight_mat = batched_matmul_impl(&transpose_last2(&grad_out_g), &cols)?;
                // grad_weight_mat: [B, Cout_g, Cin_g*K] -> sum over batch -> [Cout_g, Cin_g*K]
                let grad_weight_summed = sum_batch_dim(&grad_weight_mat)?;
                let grad_weight_g = grad_weight_summed.reshape(&[cout_g, cin_g, kernel_size])?;
                grad_weight_groups.push(grad_weight_g);
            }

            let grad_input = if groups == 1 {
                grad_input_groups
                    .into_iter()
                    .next()
                    .ok_or(BackwardError::Recipe {
                        operation: OperationKind::Conv2d,
                        reason: "the single convolution group produced no input gradient",
                    })?
            } else {
                concat_along_dim1(&grad_input_groups)?
            };
            let grad_weight = if groups == 1 {
                grad_weight_groups
                    .into_iter()
                    .next()
                    .ok_or(BackwardError::Recipe {
                        operation: OperationKind::Conv2d,
                        reason: "the single convolution group produced no weight gradient",
                    })?
            } else {
                concat_along_dim0(&grad_weight_groups)?
            };

            Ok(vec![grad_input, grad_weight])
        }),
    });

    match bias {
        Some(bias) => {
            let bias_shaped = bias.reshape(&[1, cout, 1])?;
            add_storage(&conv_out, &bias_shaped)
        }
        None => Ok(conv_out),
    }
}

// ---------------------------------------------------------------------------
// conv2d_impl
// ---------------------------------------------------------------------------

/// Canonical conv2d implementation, mirroring
/// `conv1d_impl`'s exact structure generalized to two spatial axes.
///
/// The legacy signature states one extent for both axes, so this is the
/// isotropic case of [`conv2d_windowed_impl`] rather than a kernel of its own.
pub(crate) fn conv2d_impl<D: incin_core::prelude::Device, K: DType>(
    input: &CpuStorage,
    weight: &CpuStorage,
    bias: Option<&CpuStorage>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<CpuStorage> {
    conv2d_windowed_impl::<D, K>(
        input,
        weight,
        bias,
        Window2d::isotropic(stride, padding, dilation),
        groups,
    )
}

/// `conv2d` with a window stated once per spatial axis.
///
/// The descriptor carries a stride, a padding and a dilation for each axis,
/// and an anisotropic one used to be refused because the routed kernel took a
/// single extent for both. Nothing about the algorithm needed them equal: the
/// row and column extents are used in separate expressions throughout, and
/// making them separate parameters is the whole of the change.
pub(crate) fn conv2d_windowed_impl<D: incin_core::prelude::Device, K: DType>(
    input: &CpuStorage,
    weight: &CpuStorage,
    bias: Option<&CpuStorage>,
    window: Window2d,
    groups: usize,
) -> Result<CpuStorage> {
    let (b, cin, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let (cout, cin_g, kh, kw) = (
        weight.shape[0],
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    );
    validate_groups("conv2d", cin, cout, groups)?;
    if cin / groups != cin_g {
        return Err(Error::ShapeMismatch {
            op: "conv2d",
            expected: vec![cin / groups],
            got: vec![cin_g],
            msg: format!(
                "conv2d: weight's Cin/groups ({cin_g}) does not match input Cin/groups ({})",
                cin / groups
            ),
        });
    }
    let cout_g = cout / groups;
    let h_out = out_size(
        h,
        kh,
        window.stride[0],
        window.padding[0],
        window.dilation[0],
    )?;
    let w_out = out_size(
        w,
        kw,
        window.stride[1],
        window.padding[1],
        window.dilation[1],
    )?;
    let spatial = ShapeBuf::from_slice(&[h_out, w_out]).checked_numel(OperationKind::Conv2d)?;
    let input_columns =
        ShapeBuf::from_slice(&[cin_g, kh, kw]).checked_numel(OperationKind::Conv2d)?;

    let mut group_outputs: Vec<CpuStorage> = Vec::with_capacity(groups);
    for g in 0..groups {
        let input_g = input.narrow(1, g * cin_g, cin_g)?;
        let weight_g = weight.narrow(0, g * cout_g, cout_g)?;
        let cols = im2col_2d(&input_g, kh, kw, window)?;
        let weight_mat = weight_g.reshape(&[cout_g, input_columns])?;
        let out_g = batched_matmul_impl(&cols, &transpose_last2(&weight_mat))?;
        group_outputs.push(out_g);
    }
    let refs: Vec<&CpuStorage> = group_outputs.iter().collect();
    let matmul_out = if groups == 1 {
        group_outputs[0].clone()
    } else {
        concat_storage(&refs, 2)?
    };
    // matmul_out: [B, H_out*W_out, Cout] -> canonical [B, Cout, H_out, W_out]
    let conv_out = matmul_out
        .transpose(1, 2)?
        .reshape(&[b, cout, h_out, w_out])?;

    let (input_capture, weight_capture) = (input.clone(), weight.clone());
    let (input_id, weight_id, out_id) = (input.id, weight.id, conv_out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![input_id, weight_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            // grad_out: [B, Cout, H_out, W_out] -> [B, Cout, H_out*W_out] -> [B, H_out*W_out, Cout]
            let grad_out_flat = grad_out.reshape(&[b, cout, spatial])?;
            let grad_out_t = grad_out_flat.transpose(1, 2)?;

            let mut grad_input_groups: Vec<CpuStorage> = Vec::with_capacity(groups);
            let mut grad_weight_groups: Vec<CpuStorage> = Vec::with_capacity(groups);
            for g in 0..groups {
                let input_g = input_capture.narrow(1, g * cin_g, cin_g)?;
                let weight_g = weight_capture.narrow(0, g * cout_g, cout_g)?;
                let grad_out_g = grad_out_t.narrow(2, g * cout_g, cout_g)?;

                let weight_mat = weight_g.reshape(&[cout_g, input_columns])?;

                let grad_cols = batched_matmul_impl(&grad_out_g, &weight_mat)?;
                let grad_input_g = col2im_2d(&grad_cols, &[b, cin_g, h, w], kh, kw, window)?;
                grad_input_groups.push(grad_input_g);

                let cols = im2col_2d(&input_g, kh, kw, window)?;
                let grad_weight_mat = batched_matmul_impl(&transpose_last2(&grad_out_g), &cols)?;
                let grad_weight_summed = sum_batch_dim(&grad_weight_mat)?;
                let grad_weight_g = grad_weight_summed.reshape(&[cout_g, cin_g, kh, kw])?;
                grad_weight_groups.push(grad_weight_g);
            }

            let grad_input = if groups == 1 {
                grad_input_groups
                    .into_iter()
                    .next()
                    .ok_or(BackwardError::Recipe {
                        operation: OperationKind::Conv2d,
                        reason: "the single convolution group produced no input gradient",
                    })?
            } else {
                concat_along_dim1(&grad_input_groups)?
            };
            let grad_weight = if groups == 1 {
                grad_weight_groups
                    .into_iter()
                    .next()
                    .ok_or(BackwardError::Recipe {
                        operation: OperationKind::Conv2d,
                        reason: "the single convolution group produced no weight gradient",
                    })?
            } else {
                concat_along_dim0(&grad_weight_groups)?
            };

            Ok(vec![grad_input, grad_weight])
        }),
    });

    match bias {
        Some(bias) => {
            let bias_shaped = bias.reshape(&[1, cout, 1, 1])?;
            add_storage(&conv_out, &bias_shaped)
        }
        None => Ok(conv_out),
    }
}

// ---------------------------------------------------------------------------
// conv_transpose2d_impl
// ---------------------------------------------------------------------------

/// Canonical conv-transpose2d implementation (RESEARCH.md
/// Pattern 4): transposed convolution's forward pass is exactly `conv2d`'s
/// own backward-data (grad-w.r.t.-input) formula applied directly to
/// `input` (renamed "output" in transposed-conv terminology) instead of to a
/// gradient — so this reuses `col2im_2d` (built in Plan 04-05 for
/// `conv2d_impl`'s backward) VERBATIM as its forward fold subroutine,
/// rather than a separate im2col-style forward.
///
/// `weight` arrives in Candle's confirmed `conv_transpose2d` layout
/// `[Cin, Cout, Kh, Kw]` — already the "transposed channel order" relative
/// to `conv2d`'s own `[Cout, Cin, Kh, Kw]` convention that the backward-data
/// formula needs, so no additional channel-axis transpose is required.
///
/// `output_padding` (Pitfall 4) is handled as its OWN final step, separate
/// from `padding`'s symmetric fold-size arithmetic: the natural
/// (no-`output_padding`) fold output is computed first via
/// `natural_transpose_out_size`, then — only if `output_padding > 0` — the
/// final output buffer is allocated `output_padding` larger (added once, not
/// doubled) in H and W via `scatter_into_zeros`, copying the natural result
/// into the leading `[0..H_nat, 0..W_nat]` sub-region and leaving the
/// trailing rows/columns at exactly `0.0`.
///
/// Only `groups == 1` is supported (an accepted narrower-scope
/// simplification matching `CandleBackend::conv_transpose2d`'s own
/// confirmed behavior, which likewise ignores `groups`); a `groups != 1`
/// call returns a typed `Error::ShapeMismatch` rather than silently ignoring
/// the parameter or asserting via `debug_assert_eq!`.
pub(crate) fn conv_transpose2d_impl<D: incin_core::prelude::Device, K: DType>(
    input: &CpuStorage,
    weight: &CpuStorage,
    bias: Option<&CpuStorage>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<CpuStorage> {
    if groups != 1 {
        return Err(Error::ShapeMismatch {
            op: "conv_transpose2d",
            expected: vec![1],
            got: vec![groups],
            msg: format!(
                "conv_transpose2d: only groups == 1 is supported on CpuBackendImpl, got groups={groups}"
            ),
        });
    }

    let (b, cin, h, w) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    // weight: [Cin, Cout, Kh, Kw] (Candle's conv_transpose2d convention).
    let (w_cin, cout, kh, kw) = (
        weight.shape[0],
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    );
    if w_cin != cin {
        return Err(Error::ShapeMismatch {
            op: "conv_transpose2d",
            expected: vec![cin],
            got: vec![w_cin],
            msg: format!(
                "conv_transpose2d: weight's Cin ({w_cin}) does not match input Cin ({cin})"
            ),
        });
    }

    let h_nat = natural_transpose_out_size(h, kh, stride, padding, dilation)?;
    let w_nat = natural_transpose_out_size(w, kw, stride, padding, dilation)?;
    let input_spatial = ShapeBuf::from_slice(&[h, w]).checked_numel(OperationKind::Conv2d)?;
    let weight_columns =
        ShapeBuf::from_slice(&[cout, kh, kw]).checked_numel(OperationKind::Conv2d)?;

    // input: [B, Cin, H, W] -> [B, Cin, H*W] -> [B, H*W, Cin] (mirrors
    // conv2d_impl's backward: "grad_out_t" role, but played by `input` here).
    let input_flat = input.reshape(&[b, cin, input_spatial])?;
    let input_t = input_flat.transpose(1, 2)?;
    // weight: [Cin, Cout, Kh, Kw] -> [Cin, Cout*Kh*Kw] ("weight_mat" role).
    let weight_mat = weight.reshape(&[cin, weight_columns])?;
    // cols = input_t @ weight_mat : [B, H*W, Cin] @ [Cin, Cout*Kh*Kw] -> [B, H*W, Cout*Kh*Kw]
    let cols = batched_matmul_impl(&input_t, &weight_mat)?;

    // Fold cols into the natural (no output_padding) [B, Cout, H_nat, W_nat]
    // output, reusing col2im_2d verbatim (Pattern 4).
    let natural_out = col2im_2d(
        &cols,
        &[b, cout, h_nat, w_nat],
        kh,
        kw,
        Window2d::isotropic(stride, padding, dilation),
    )?;

    let conv_out = if output_padding == 0 {
        natural_out
    } else {
        let final_h = h_nat
            .checked_add(output_padding)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Conv2d,
                expression: "transposed-convolution height plus output padding",
            })?;
        let final_w = w_nat
            .checked_add(output_padding)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Conv2d,
                expression: "transposed-convolution width plus output padding",
            })?;
        ShapeBuf::from_slice(&[b, cout, final_h, final_w]).checked_numel(OperationKind::Conv2d)?;
        let final_shape = vec![b, cout, final_h, final_w];
        scatter_into_zeros(&final_shape, &[0, 0, 0, 0], &natural_out)?
    };

    let (input_capture, weight_capture) = (input.clone(), weight.clone());
    let (input_id, weight_id, out_id) = (input.id, weight.id, conv_out.id);
    tape::push(TapeEntry {
        output_id: out_id,
        input_ids: vec![input_id, weight_id],
        backward: Box::new(move |grad_out: &CpuStorage| {
            // grad_out is shaped like conv_out ([B, Cout, H_nat+op, W_nat+op]);
            // narrow away any trailing output_padding rows/columns first so
            // every downstream step operates on the natural [B, Cout, H_nat,
            // W_nat] fold shape (output_padding contributes nothing to
            // grad_input/grad_weight, matching its forward role as a
            // trailing-zero region only).
            let grad_out_nat = if output_padding == 0 {
                grad_out.clone()
            } else {
                grad_out
                    .narrow(2, 0, h_nat)
                    .and_then(|t| t.narrow(3, 0, w_nat))?
            };

            // conv_transpose2d's OWN backward w.r.t. its input is exactly
            // conv2d's FORWARD formula (im2col_2d + matmul) applied to
            // grad_out_nat against the same weight (the exact inverse
            // relationship of this op's forward being conv2d's
            // backward-data formula): im2col_2d unfolds grad_out_nat using
            // the SAME window geometry the forward fold used, then matmuls
            // against weight_mat (transposed relative to the forward's own
            // orientation) to recover grad_input.
            let grad_out_cols = im2col_2d(
                &grad_out_nat,
                kh,
                kw,
                Window2d::isotropic(stride, padding, dilation),
            )?;
            // grad_out_cols: [B, H*W, Cout*Kh*Kw] @ weight_mat^T: [Cout*Kh*Kw, Cin] -> [B, H*W, Cin]
            let weight_mat = weight_capture.reshape(&[cin, weight_columns])?;
            let grad_input_flat =
                batched_matmul_impl(&grad_out_cols, &transpose_last2(&weight_mat))?;
            // [B, H*W, Cin] -> [B, Cin, H*W] -> [B, Cin, H, W]
            let grad_input = grad_input_flat.transpose(1, 2)?.reshape(&[b, cin, h, w])?;

            // grad_weight follows the same per-position outer-product-and-sum
            // structure conv2d_impl's own grad_weight closure uses, with
            // `input` and `grad_out` swapped relative to conv2d's own
            // convention (conv_transpose2d's forward played `input`'s role
            // where conv2d's backward played `grad_out`'s role, and vice
            // versa) — this swap is the least-obvious part of this reuse:
            // grad_weight_mat = input_t^T @ grad_out_cols :
            // [Cin, B*H*W] view via batched matmul against [B, H*W, Cout*Kh*Kw].
            let input_flat = input_capture.reshape(&[b, cin, input_spatial])?;
            let input_t = input_flat.transpose(1, 2)?;
            let grad_weight_mat = batched_matmul_impl(&transpose_last2(&input_t), &grad_out_cols)?;
            // grad_weight_mat: [B, Cin, Cout*Kh*Kw] -> sum over batch -> [Cin, Cout*Kh*Kw]
            let grad_weight_summed = sum_batch_dim(&grad_weight_mat)?;
            let grad_weight = grad_weight_summed.reshape(&[cin, cout, kh, kw])?;

            Ok(vec![grad_input, grad_weight])
        }),
    });

    match bias {
        Some(bias) => {
            let bias_shaped = bias.reshape(&[1, cout, 1, 1])?;
            add_storage(&conv_out, &bias_shaped)
        }
        None => Ok(conv_out),
    }
}

// ---------------------------------------------------------------------------
// Shared backward-composition helpers (plain, non-tape-tracked — operate on
// already-materialized CpuStorage values inside the hand-composed
// backward closures above, mirroring the forward's own per-group narrow/
// concat convention).
// ---------------------------------------------------------------------------

/// Sum a `[B, M, N]` storage over its leading batch axis, producing `[M, N]`.
/// Used by both `conv1d_impl`/`conv2d_impl`'s backward to reduce
/// `grad_weight`'s per-batch matmul contributions down to weight's own
/// (batch-independent) shape.
fn sum_batch_dim(t: &CpuStorage) -> Result<CpuStorage> {
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
fn concat_along_dim1(parts: &[CpuStorage]) -> Result<CpuStorage> {
    concat_along_dim(parts, 1)
}

/// Plain (non-tape-tracked) concat along axis 0, used to re-assemble
/// `grad_weight`'s per-group output-channel slices.
fn concat_along_dim0(parts: &[CpuStorage]) -> Result<CpuStorage> {
    concat_along_dim(parts, 0)
}

/// `concat_along_dim`.
fn concat_along_dim(parts: &[CpuStorage], dim: usize) -> Result<CpuStorage> {
    let rank = parts[0].shape.len();
    let mut out_shape = parts[0].shape.to_vec();
    out_shape[dim] = parts.iter().try_fold(0usize, |total, part| {
        total.checked_add(part.shape[dim]).ok_or(
            incin_core::prelude::ShapeError::ArithmeticOverflow {
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
            incin_core::prelude::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Concat,
                expression: "cumulative convolution gradient group offset",
            },
        )?;
    }

    Ok(CpuStorage::from_contiguous(
        CpuBuffer::F32(out),
        out_shape.to_vec(),
    ))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;
    use crate::cpu::CpuBackendImpl;
    use crate::cpu::gradcheck::gradcheck;
    use incin_core::prelude::Cpu;

    /// `TestBackend`.
    type TestBackend = CpuBackendImpl<Cpu>;

    /// `tensor`.
    fn tensor(v: Vec<f32>, shape: Vec<usize>) -> CpuStorage {
        CpuStorage::from_contiguous(CpuBuffer::F32(v), shape)
    }

    /// `f32_vec`.
    fn f32_vec(s: &CpuStorage) -> Vec<f32> {
        match &*s.buffer {
            CpuBuffer::F32(v) => v.clone(),
            _ => panic!("expected F32 buffer"),
        }
    }

    // --- conv1d forward ---

    /// Forward test (groups=1, stride=1, padding=0, dilation=1): a small
    /// hand-computable [1,1,4] input convolved with a [1,1,2] kernel produces
    /// a [1,1,3] output matching manual sliding-window dot products.
    #[test]
    fn conv1d_forward_hand_computed_no_padding() {
        // input = [1,2,3,4], kernel = [10,1]
        let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
        let weight = tensor(vec![10.0, 1.0], vec![1, 1, 2]);
        let out = conv1d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3]);
        // window0 = [1,2] . [10,1] = 10+2=12
        // window1 = [2,3] . [10,1] = 20+3=23
        // window2 = [3,4] . [10,1] = 30+4=34
        assert_eq!(f32_vec(&out), vec![12.0, 23.0, 34.0]);
    }

    /// Forward test (padding>0, Pitfall 2): a [1,1,3] input with padding=1
    /// and a [1,1,3] kernel produces the correct zero-padded-boundary
    /// output.
    #[test]
    fn conv1d_forward_with_padding_zero_fills_boundary() {
        // input = [1,2,3], kernel = [1,1,1], padding=1 -> padded = [0,1,2,3,0]
        let input = tensor(vec![1.0, 2.0, 3.0], vec![1, 1, 3]);
        let weight = tensor(vec![1.0, 1.0, 1.0], vec![1, 1, 3]);
        let out = conv1d_impl::<Cpu, f32>(&input, &weight, None, 1, 1, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3]);
        // windows over padded [0,1,2,3,0]: [0,1,2]->3, [1,2,3]->6, [2,3,0]->5
        assert_eq!(f32_vec(&out), vec![3.0, 6.0, 5.0]);
    }

    /// Forward test (groups>1, Pitfall 7): a [1,4,5] input (Cin=4) with
    /// groups=2 and a [2,2,2] weight (Cout=2, Cin/groups=2) matches two
    /// independent single-group convolutions concatenated along the
    /// output-channel axis.
    #[test]
    fn conv1d_forward_groups_matches_two_independent_convs() {
        let input_data: Vec<f32> = (1..=20).map(|x| x as f32).collect(); // [1,4,5]
        let input = tensor(input_data.clone(), vec![1, 4, 5]);
        let weight_data: Vec<f32> = (1..=8).map(|x| x as f32 * 0.1).collect(); // [2,2,2]
        let weight = tensor(weight_data.clone(), vec![2, 2, 2]);

        let out = conv1d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 1, 2).unwrap();
        assert_eq!(out.shape, vec![1, 2, 4]);

        // group 0: input channels [0,1] (rows 0-1 of input, each len 5),
        // weight channel 0 (shape [1,2,2])
        let g0_input = tensor(input_data[0..10].to_vec(), vec![1, 2, 5]);
        let g0_weight = tensor(weight_data[0..4].to_vec(), vec![1, 2, 2]);
        let g0_out = conv1d_impl::<Cpu, f32>(&g0_input, &g0_weight, None, 1, 0, 1, 1).unwrap();

        let g1_input = tensor(input_data[10..20].to_vec(), vec![1, 2, 5]);
        let g1_weight = tensor(weight_data[4..8].to_vec(), vec![1, 2, 2]);
        let g1_out = conv1d_impl::<Cpu, f32>(&g1_input, &g1_weight, None, 1, 0, 1, 1).unwrap();

        let combined = f32_vec(&out);
        assert_eq!(&combined[0..4], &f32_vec(&g0_out)[..]);
        assert_eq!(&combined[4..8], &f32_vec(&g1_out)[..]);
    }

    /// Forward test (bias): providing `Some(bias)` adds the per-output-channel
    /// bias value to every spatial position of that channel.
    #[test]
    fn conv1d_forward_with_bias_adds_per_channel_constant() {
        let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
        let weight = tensor(vec![10.0, 1.0], vec![1, 1, 2]);
        let bias = tensor(vec![100.0], vec![1]);
        let out = conv1d_impl::<Cpu, f32>(&input, &weight, Some(&bias), 1, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3]);
        assert_eq!(f32_vec(&out), vec![112.0, 123.0, 134.0]);
    }

    // --- conv1d backward ---

    /// `conv1d_sum_op`.
    fn conv1d_sum_op(inputs: &[CpuStorage]) -> CpuStorage {
        let out = conv1d_impl::<Cpu, f32>(&inputs[0], &inputs[1], None, 1, 0, 1, 1).unwrap();
        crate::cpu::ops::reduce::sum_all(&out).unwrap()
    }

    /// Backward test (gradcheck against input AND weight): a small
    /// [1,1,4]/[1,1,2] pair, wrapped in `sum_all`, gradchecked with
    /// `max_relative_error < 1e-2` for BOTH the input and the weight tensor.
    #[test]
    fn conv1d_gradcheck_input_and_weight() {
        let input = tensor(vec![0.1, 0.2, 0.3, 0.4], vec![1, 1, 4]);
        let weight = tensor(vec![0.5, 0.6], vec![1, 1, 2]);
        let max_rel_err = gradcheck(conv1d_sum_op, &[input, weight], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "gradcheck max relative error too high: {max_rel_err}"
        );
    }

    /// Backward test (overlapping windows, stride < kernel_size): confirms
    /// col2im's fold uses `+=` accumulation.
    #[test]
    fn conv1d_backward_overlapping_windows_accumulates_grad_input() {
        // kernel_size=2, stride=1 on a length-3 input -> 2 output positions,
        // each input position (except the first/last) touched by 2 windows.
        let input = tensor(vec![1.0, 2.0, 3.0], vec![1, 1, 3]);
        let weight = tensor(vec![1.0, 1.0], vec![1, 1, 2]);
        let out = conv1d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 1, 1).unwrap();
        let loss = crate::cpu::ops::reduce::sum_all(&out).unwrap();
        let grads = tape::backward(&loss).unwrap();
        let grad_input = grads.get(input.id).expect("grad_input should exist");
        // grad w.r.t weight[k] summed over both output positions = 1 each,
        // and grad w.r.t input[i] = sum of weight values whose window covers i.
        // window0 covers input[0],input[1]; window1 covers input[1],input[2].
        // grad_input[0] = weight[0] (only window0) = 1
        // grad_input[1] = weight[1] (window0) + weight[0] (window1) = 1+1=2
        // grad_input[2] = weight[1] (only window1) = 1
        assert_eq!(f32_vec(grad_input), vec![1.0, 2.0, 1.0]);
    }

    // --- conv2d forward ---

    /// Forward test (2D, groups=1): a [1,1,4,4] input convolved with a
    /// [1,1,3,3] kernel, stride=1, padding=0, dilation=1 -> [1,1,2,2] output
    /// matching hand-computed sliding-window sums.
    #[test]
    fn conv2d_forward_hand_computed_no_padding() {
        let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect(); // [1,1,4,4]
        let input = tensor(input_data, vec![1, 1, 4, 4]);
        let weight = tensor(vec![1.0; 9], vec![1, 1, 3, 3]); // sum-of-window kernel
        let out = conv2d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        // input matrix:
        //  1  2  3  4
        //  5  6  7  8
        //  9 10 11 12
        // 13 14 15 16
        // window(0,0) = rows0-2,cols0-2 = 1+2+3+5+6+7+9+10+11=54
        // window(0,1) = rows0-2,cols1-3 = 2+3+4+6+7+8+10+11+12=63
        // window(1,0) = rows1-3,cols0-2 = 5+6+7+9+10+11+13+14+15=90
        // window(1,1) = rows1-3,cols1-3 = 6+7+8+10+11+12+14+15+16=99
        assert_eq!(f32_vec(&out), vec![54.0, 63.0, 90.0, 99.0]);
    }

    /// Forward test (groups>1, Pitfall 7, 2D case): a [1,4,5,5] input
    /// (Cin=4) with groups=2 and weight [2,2,3,3] matches two independent
    /// single-group conv2d calls concatenated along the output-channel axis.
    #[test]
    fn conv2d_forward_groups_matches_two_independent_convs() {
        let input_data: Vec<f32> = (1..=100).map(|x| x as f32 * 0.01).collect(); // [1,4,5,5]
        let input = tensor(input_data.clone(), vec![1, 4, 5, 5]);
        let weight_data: Vec<f32> = (1..=36).map(|x| x as f32 * 0.01).collect(); // [2,2,3,3]
        let weight = tensor(weight_data.clone(), vec![2, 2, 3, 3]);

        let out = conv2d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 1, 2).unwrap();
        assert_eq!(out.shape, vec![1, 2, 3, 3]);

        let g0_input = tensor(input_data[0..50].to_vec(), vec![1, 2, 5, 5]);
        let g0_weight = tensor(weight_data[0..18].to_vec(), vec![1, 2, 3, 3]);
        let g0_out = conv2d_impl::<Cpu, f32>(&g0_input, &g0_weight, None, 1, 0, 1, 1).unwrap();

        let g1_input = tensor(input_data[50..100].to_vec(), vec![1, 2, 5, 5]);
        let g1_weight = tensor(weight_data[18..36].to_vec(), vec![1, 2, 3, 3]);
        let g1_out = conv2d_impl::<Cpu, f32>(&g1_input, &g1_weight, None, 1, 0, 1, 1).unwrap();

        let combined = f32_vec(&out);
        assert_eq!(&combined[0..9], &f32_vec(&g0_out)[..]);
        assert_eq!(&combined[9..18], &f32_vec(&g1_out)[..]);
    }

    /// Forward test (depthwise, groups==Cin): a [1,3,5,5] input with
    /// groups=3 and weight [3,1,3,3] runs through the SAME code path as
    /// groups=2 above (no special branch), producing correct
    /// per-channel-independent output.
    #[test]
    fn conv2d_forward_depthwise_groups_equal_cin() {
        let input_data: Vec<f32> = (1..=75).map(|x| x as f32 * 0.01).collect(); // [1,3,5,5]
        let input = tensor(input_data.clone(), vec![1, 3, 5, 5]);
        let weight_data: Vec<f32> = (1..=27).map(|x| x as f32 * 0.01).collect(); // [3,1,3,3]
        let weight = tensor(weight_data.clone(), vec![3, 1, 3, 3]);

        let out = conv2d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 1, 3).unwrap();
        assert_eq!(out.shape, vec![1, 3, 3, 3]);

        // Verify each channel independently against a groups=1 conv on just
        // that channel's own input/weight slice.
        for c in 0..3 {
            let ch_input = tensor(input_data[c * 25..(c + 1) * 25].to_vec(), vec![1, 1, 5, 5]);
            let ch_weight = tensor(weight_data[c * 9..(c + 1) * 9].to_vec(), vec![1, 1, 3, 3]);
            let ch_out = conv2d_impl::<Cpu, f32>(&ch_input, &ch_weight, None, 1, 0, 1, 1).unwrap();
            let combined = f32_vec(&out);
            assert_eq!(&combined[c * 9..(c + 1) * 9], &f32_vec(&ch_out)[..]);
        }
    }

    // --- conv2d backward ---

    /// `conv2d_sum_op`.
    fn conv2d_sum_op(inputs: &[CpuStorage]) -> CpuStorage {
        let out = conv2d_impl::<Cpu, f32>(&inputs[0], &inputs[1], None, 1, 0, 1, 1).unwrap();
        crate::cpu::ops::reduce::sum_all(&out).unwrap()
    }

    /// Backward test: gradcheck on a small [1,1,4,4]/[1,1,2,2] pair
    /// (stride=1,padding=0,dilation=1,groups=1), max_relative_error < 1e-2
    /// for both grad_input and grad_weight.
    #[test]
    fn conv2d_gradcheck_input_and_weight() {
        let input_data: Vec<f32> = (1..=16).map(|x| x as f32 * 0.01).collect();
        let input = tensor(input_data, vec![1, 1, 4, 4]);
        let weight_data: Vec<f32> = (1..=4).map(|x| x as f32 * 0.1).collect();
        let weight = tensor(weight_data, vec![1, 1, 2, 2]);
        let max_rel_err = gradcheck(conv2d_sum_op, &[input, weight], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "gradcheck max relative error too high: {max_rel_err}"
        );
    }

    // --- conv_transpose2d forward ---

    /// Forward test (basic, stride=1, padding=0, output_padding=0,
    /// dilation=1, groups=1): a small [1,1,2,2] input with a [1,1,2,2]
    /// weight (Cin=1,Cout=1) produces the hand-computed transposed-conv
    /// output (verified against a manually-derived scatter-add-of-weighted-
    /// patches reference, not just shape).
    #[test]
    fn conv_transpose2d_forward_hand_computed_basic() {
        // input = [[1,2],[3,4]], weight = [[1,1],[1,1]]
        let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
        let out = conv_transpose2d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
        // Hand-computed scatter-add of weighted 2x2 patches:
        // out[i+kh, j+kw] += input[i,j] * weight[kh,kw] for i,j,kh,kw in 0..2
        assert_eq!(
            f32_vec(&out),
            vec![1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]
        );
    }

    /// Forward test (stride>1, the common upsampling case): a [1,1,2,2]
    /// input with stride=2 produces an output shape matching Candle's exact
    /// formula `(i_h - 1) * stride + dilation*(k_h-1) + output_padding + 1 -
    /// 2*padding` for both H and W, with hand-computed values.
    #[test]
    fn conv_transpose2d_forward_stride_upsamples() {
        let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
        let out = conv_transpose2d_impl::<Cpu, f32>(&input, &weight, None, 2, 0, 0, 1, 1).unwrap();
        // (2-1)*2 + 1*(2-1) + 1 - 0 = 4
        assert_eq!(out.shape, vec![1, 1, 4, 4]);
        assert_eq!(
            f32_vec(&out),
            vec![
                1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0
            ]
        );
    }

    /// Forward test (output_padding>0, Pitfall 4): explicitly constructs a
    /// case with non-zero `output_padding` and confirms the extra
    /// rows/columns are allocated on the correct (bottom/right) side ONLY,
    /// at exactly value 0.0 — confirming the natural fold-output size is
    /// computed first using `padding` symmetrically, THEN `output_padding`
    /// extra rows/columns are appended afterward (not folded into the same
    /// offset arithmetic as `padding`).
    #[test]
    fn conv_transpose2d_forward_output_padding_appends_trailing_zeros_only() {
        let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
        let out = conv_transpose2d_impl::<Cpu, f32>(&input, &weight, None, 2, 0, 1, 1, 1).unwrap();
        // natural (output_padding=0) shape was [1,1,4,4]; output_padding=1
        // appends ONE extra trailing row and column -> [1,1,5,5].
        assert_eq!(out.shape, vec![1, 1, 5, 5]);
        let vals = f32_vec(&out);
        let natural = [
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ];
        // Leading [0..4, 0..4] sub-region matches the natural (no
        // output_padding) result exactly.
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(vals[row * 5 + col], natural[row * 4 + col]);
            }
        }
        // The trailing row (row=4) and trailing column (col=4) are exactly
        // 0.0 for every position.
        for col in 0..5 {
            assert_eq!(vals[4 * 5 + col], 0.0, "trailing row must be zero");
        }
        for row in 0..5 {
            assert_eq!(vals[row * 5 + 4], 0.0, "trailing column must be zero");
        }
    }

    /// Forward test (groups != 1 rejected): calling with `groups=2` returns
    /// a typed `Error::ShapeMismatch` rather than silently ignoring the
    /// parameter or panicking via `debug_assert_eq!`.
    #[test]
    fn conv_transpose2d_rejects_groups_other_than_one() {
        let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let weight = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
        let result = conv_transpose2d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 0, 1, 2);
        assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
    }

    // --- conv_transpose2d backward ---

    /// `conv_transpose2d_sum_op`.
    fn conv_transpose2d_sum_op(inputs: &[CpuStorage]) -> CpuStorage {
        let out =
            conv_transpose2d_impl::<Cpu, f32>(&inputs[0], &inputs[1], None, 1, 0, 0, 1, 1).unwrap();
        crate::cpu::ops::reduce::sum_all(&out).unwrap()
    }

    /// Backward test: gradcheck on the basic [1,1,2,2]/[1,1,2,2] case
    /// (stride=1, padding=0, output_padding=0, dilation=1),
    /// max_relative_error < 1e-2 for both grad_input and grad_weight.
    #[test]
    fn conv_transpose2d_gradcheck_input_and_weight() {
        let input = tensor(vec![0.1, 0.2, 0.3, 0.4], vec![1, 1, 2, 2]);
        let weight = tensor(vec![0.5, 0.6, 0.7, 0.8], vec![1, 1, 2, 2]);
        let max_rel_err = gradcheck(conv_transpose2d_sum_op, &[input, weight], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "gradcheck max relative error too high: {max_rel_err}"
        );
    }
}
