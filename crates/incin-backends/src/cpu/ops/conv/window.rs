//! The 2D window descriptor and `im2col_2d`/`col2im_2d`, the 2D
//! generalization of `unfold1d`'s gather/scatter-ADD pair.

use incin_core::error::Result;
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;

use crate::cpu::storage::{CpuBuffer, CpuStorage};

use super::helpers::out_size;

// ---------------------------------------------------------------------------
// im2col_2d / col2im_2d
// ---------------------------------------------------------------------------

/// A convolution window, stated once per spatial axis.
///
/// The descriptor that routes here carries a stride, a padding and a dilation
/// per axis, so the kernel takes the pair rather than one extent applied to
/// both. The row and column values are used in separate expressions
/// throughout, which makes an anisotropic window an ordinary case rather than
/// a separate path.
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
    /// The same extent on both axes.
    ///
    /// `conv_transpose2d_impl` still states one extent for both axes, and its
    /// unfold and fold subroutines take a window, so it spreads the scalar
    /// here rather than each call site writing the pair out.
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
pub(super) fn im2col_2d(
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
pub(super) fn col2im_2d(
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

    CpuStorage::try_from_contiguous(CpuBuffer::F32(out), input_shape)
}
