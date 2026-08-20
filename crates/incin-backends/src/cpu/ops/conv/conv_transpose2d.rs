//! `conv_transpose2d_impl` (RESEARCH.md Pattern 4): transposed convolution's
//! forward pass reuses `col2im_2d` verbatim as its own forward fold, and its
//! backward reuses `im2col_2d` (i.e. `conv2d`'s forward formula).

use incin_core::error::{Error, Result};
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::ShapeError;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DType;

use crate::cpu::ops::elementwise::add_storage;
use crate::cpu::ops::matmul::{batched_matmul_impl, transpose_last2};
use crate::cpu::storage::{CpuStorage, scatter_into_zeros};
use crate::cpu::tape::{self, TapeEntry};

use super::combine::sum_batch_dim;
use super::helpers::natural_transpose_out_size;
use super::window::{Window2d, col2im_2d, im2col_2d};

// ---------------------------------------------------------------------------
// conv_transpose2d_impl
// ---------------------------------------------------------------------------

/// Canonical conv-transpose2d implementation (RESEARCH.md
/// Pattern 4): transposed convolution's forward pass is exactly `conv2d`'s
/// own backward-data (grad-w.r.t.-input) formula applied directly to
/// `input` (renamed "output" in transposed-conv terminology) instead of to a
/// gradient - so this reuses `col2im_2d` (built in Plan 04-05 for
/// `conv2d_windowed_impl`'s backward) VERBATIM as its forward fold subroutine,
/// rather than a separate im2col-style forward.
///
/// `weight` arrives in Candle's confirmed `conv_transpose2d` layout
/// `[Cin, Cout, Kh, Kw]` - already the "transposed channel order" relative
/// to `conv2d`'s own `[Cout, Cin, Kh, Kw]` convention that the backward-data
/// formula needs, so no additional channel-axis transpose is required.
///
/// `output_padding` (Pitfall 4) is handled as its OWN final step, separate
/// from `padding`'s symmetric fold-size arithmetic: the natural
/// (no-`output_padding`) fold output is computed first via
/// `natural_transpose_out_size`, then - only if `output_padding > 0` - the
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
#[allow(clippy::extra_unused_type_parameters, clippy::too_many_arguments)]
pub(crate) fn conv_transpose2d_impl<D: incin_core::tensor::device::Device, K: DType>(
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
    // conv2d_windowed_impl's backward: "grad_out_t" role, but played by `input` here).
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
    tape::push_with(|| TapeEntry {
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
            // structure conv2d_windowed_impl's own grad_weight closure uses, with
            // `input` and `grad_out` swapped relative to conv2d's own
            // convention (conv_transpose2d's forward played `input`'s role
            // where conv2d's backward played `grad_out`'s role, and vice
            // versa) - this swap is the least-obvious part of this reuse:
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
