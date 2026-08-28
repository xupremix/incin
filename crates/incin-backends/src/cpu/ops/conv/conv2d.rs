//! `conv2d_windowed_impl`, mirroring `conv1d`'s exact structure generalized
//! to two spatial axes.

use incin_core::error::{BackwardError, Error, Result};
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::DType;

use crate::cpu::ops::elementwise::add_storage;
use crate::cpu::ops::matmul::{batched_matmul_impl, transpose_last2};
use crate::cpu::ops::shape_ops::concat_storage;
use crate::cpu::storage::CpuStorage;
use crate::cpu::tape::{self, TapeEntry};

use super::combine::{concat_along_dim0, concat_along_dim1, sum_batch_dim};
use super::helpers::{out_size, validate_groups, with_batch_axis, without_batch_axis};
use super::window::{Window2d, col2im_2d, im2col_2d};

// ---------------------------------------------------------------------------
// conv2d_windowed_impl
// ---------------------------------------------------------------------------

/// Canonical conv2d implementation, mirroring `conv1d_impl`'s exact structure
/// generalized to two spatial axes.
///
/// The window is stated once per spatial axis. The descriptor carries a
/// stride, a padding and a dilation for each axis, and an anisotropic one used
/// to be refused because the routed kernel took a single extent for both.
/// Nothing about the algorithm needed them equal: the row and column extents
/// are used in separate expressions throughout, and making them separate
/// parameters is the whole of the change.
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn conv2d_windowed_impl<D: incin_core::tensor::device::Device, K: DType>(
    input: &CpuStorage,
    weight: &CpuStorage,
    bias: Option<&CpuStorage>,
    window: Window2d,
    groups: usize,
) -> Result<CpuStorage> {
    // The unbatched `[Cin, H, W]` activation the catalog admits is given a
    // batch of one here and has it taken back off below, so the im2col
    // arithmetic and the tape recipe stay written against one shape.
    let (activation, unbatched) = with_batch_axis("conv2d", input, 4)?;
    let (b, cin, h, w) = (
        activation.shape[0],
        activation.shape[1],
        activation.shape[2],
        activation.shape[3],
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
        let input_g = activation.narrow(1, g * cin_g, cin_g)?;
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
    // Unwrapped before the tape entry is pushed rather than after, so the
    // output the caller receives is the one the recipe is keyed on.
    let conv_out = without_batch_axis(conv_out, unbatched)?;

    let (input_capture, weight_capture) = (activation.clone(), weight.clone());
    let (input_id, weight_id, out_id) = (input.id, weight.id, conv_out.id);
    let input_shape = input.shape.to_vec();
    tape::push_with(|| TapeEntry {
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
            // The gradient is shaped like the activation that was asked about,
            // not like the batched one the kernel worked on.
            let grad_input = grad_input.reshape(&input_shape)?;
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
            // One broadcast axis per output axis. Right-aligned broadcasting
            // would otherwise put the batch axis back on an unbatched result.
            let bias_shaped = if unbatched {
                bias.reshape(&[cout, 1, 1])?
            } else {
                bias.reshape(&[1, cout, 1, 1])?
            };
            add_storage(&conv_out, &bias_shaped)
        }
        None => Ok(conv_out),
    }
}
