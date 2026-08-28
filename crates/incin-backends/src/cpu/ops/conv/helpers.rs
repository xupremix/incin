//! Shared output-size arithmetic and group validation used by every conv
//! variant below.

use incin_core::error::{Error, Result};
use incin_core::shapes::ShapeError;
use incin_core::shapes::error::OperationKind;

use crate::cpu::storage::CpuStorage;

// ---------------------------------------------------------------------------
// Shared output-size arithmetic (T-04-11: saturating_sub, never raw subtraction)
// ---------------------------------------------------------------------------

/// `L_out = (L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`,
/// using `saturating_sub` throughout (matching RESEARCH.md's exact formula)
/// so a pathological small-input/large-kernel combination produces `0`
/// (an empty output) rather than panicking on integer underflow.
pub(super) fn out_size(
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
/// deliberately NOT part of this formula (Pitfall 4) - it is applied as a
/// separate final allocate-larger step by the caller.
pub(super) fn natural_transpose_out_size(
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
pub(super) fn validate_groups(
    op: &'static str,
    cin: usize,
    cout: usize,
    groups: usize,
) -> Result<()> {
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

/// The activation a convolution kernel works on, given the one it was handed.
///
/// The catalog admits an unbatched activation: `accepted_ranks` opens at three
/// for the two-dimensional convolutions and at two for `conv1d`, and
/// `inference.rs` infers the output of that form by rewriting only the trailing
/// axes and reading the channel extent at `input[len - 1 - spatial]`. Every
/// kernel here is written against the batched form throughout, from the
/// `narrow(1, ..)` that splits the channel groups to the `reshape` that
/// restores the canonical output layout.
///
/// So the unbatched form is served by giving it a batch of one and taking that
/// axis back off the result, rather than by a second code path through the
/// same im2col arithmetic. The second return says whether the axis was added,
/// which the caller needs twice: once to unwrap the output, and once to return
/// a gradient shaped like the activation it was asked about.
///
/// A rank outside those two is refused rather than folded. The descriptor
/// pins the range before a kernel is reached, so this is the arm that keeps a
/// wrong answer from being a panic if a row ever widens past what the kernel
/// can address.
pub(super) fn with_batch_axis(
    op: &'static str,
    input: &CpuStorage,
    batched_rank: usize,
) -> Result<(CpuStorage, bool)> {
    let rank = input.shape.len();
    if rank == batched_rank {
        return Ok((input.clone(), false));
    }
    if rank + 1 == batched_rank {
        let mut batched = vec![1];
        batched.extend_from_slice(&input.shape);
        return Ok((input.reshape(&batched)?, true));
    }
    Err(Error::ShapeMismatch {
        op,
        expected: vec![batched_rank - 1, batched_rank],
        got: input.shape.to_vec(),
        msg: format!(
            "{op}: an activation must be batched or unbatched, and this one is rank {rank}"
        ),
    })
}

/// Drop the batch axis `with_batch_axis` added, when it added one.
pub(super) fn without_batch_axis(storage: CpuStorage, added: bool) -> Result<CpuStorage> {
    if added {
        let unbatched = storage.shape[1..].to_vec();
        storage.reshape(&unbatched)
    } else {
        Ok(storage)
    }
}
