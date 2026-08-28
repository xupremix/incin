//! `conv1d_impl`: im2col + per-group `batched_matmul_impl` + concat forward,
//! hand-composed backward.

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
use super::unfold1d::{col2im_1d, im2col_1d};

// ---------------------------------------------------------------------------
// conv1d_impl
// ---------------------------------------------------------------------------

/// Canonical conv1d implementation: im2col + per-group
/// `batched_matmul_impl` + concat forward, hand-composed backward for
/// grad_input (col2im fold) and grad_weight (per-group matmul), with bias
/// broadcast-added via the canonical storage helper (so
/// `grad_bias` is free via composition, per this file's module doc).
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn conv1d_impl<D: incin_core::tensor::device::Device, K: DType>(
    input: &CpuStorage,
    weight: &CpuStorage,
    bias: Option<&CpuStorage>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<CpuStorage> {
    // The unbatched `[Cin, L]` activation the catalog admits is given a batch
    // of one here and has it taken back off below, so the im2col arithmetic and
    // the tape recipe stay written against one shape.
    let (activation, unbatched) = with_batch_axis("conv1d", input, 3)?;
    let (b, cin, len) = (
        activation.shape[0],
        activation.shape[1],
        activation.shape[2],
    );
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
        let input_g = activation.narrow(1, g * cin_g, cin_g)?;
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
            // grad_out: [B, Cout, L_out] -> [B, L_out, Cout], with the batch
            // axis put back on first when the activation had none, since the
            // gradient is shaped like the output and the output lost it.
            let grad_out_t = with_batch_axis("conv1d", grad_out, 3)?.0.transpose(1, 2)?;

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
                bias.reshape(&[cout, 1])?
            } else {
                bias.reshape(&[1, cout, 1])?
            };
            add_storage(&conv_out, &bias_shaped)
        }
        None => Ok(conv_out),
    }
}
